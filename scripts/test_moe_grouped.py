"""MoE 分组GEMM路径的正确性检查 / correctness checks for the grouped-GEMM MoE path.

无第三方测试框架（本仓库没有 pytest / CI），直接 `python test_moe_grouped.py` 运行，失败返回非零。

多数用例把模块级的 `_grouped_mm` 替换成纯 torch 的参考实现，因此排序、偏移、
token映射、零专家剔除、scatter 等全部逻辑都能在 **CPU** 上对着原循环校验；
只有 test_cuda_real_kernel 需要真实的 CUDA + bf16 环境，否则自动 SKIP。
"""
import os, sys, traceback
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import model.model_minimind as mm
from model.model_minimind import MiniMindConfig, MOEFeedForward


class Skip(Exception): pass


# ============================== 参考实现 / references ==============================

def ref_grouped_mm(x, w, offs):
    """纯 torch 的分组矩阵乘参考实现（可求导），语义对齐 torch._grouped_mm。"""
    out, s = [], 0
    for i, e in enumerate(offs.tolist()):
        out.append(x[s:e] @ w[i]); s = e
    return torch.cat(out, 0) if out else x.new_zeros(0, w.shape[-1])


def reference_forward(mod, x):
    """上游 master 的专家循环，逐字复制，作为不变量基准。"""
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.view(-1, hidden_dim)
    scores = F.softmax(mod.gate(x_flat), dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=mod.config.num_experts_per_tok, dim=-1, sorted=False)
    if mod.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    y = torch.zeros_like(x_flat)
    for i, expert in enumerate(mod.experts):
        mask = (topk_idx == i)
        if mask.any():
            token_idx = mask.any(dim=-1).nonzero().flatten()
            weight = topk_weight[mask].view(-1, 1)
            y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
    return y.view(batch_size, seq_len, hidden_dim)


# ============================== 工具 / helpers ==============================

def cfg(**kw):
    base = dict(hidden_size=64, num_hidden_layers=1, use_moe=True, num_experts=4,
                num_experts_per_tok=1, moe_intermediate_size=128)
    base.update(kw)
    return MiniMindConfig(**base)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return MOEFeedForward(cfg(**kw))


def force_routing(mod, assign, hidden_size, k=1):
    """构造 x 使得 token t 被路由到 assign[t]：gate 权重取单位阵，x 在目标维上取大值。"""
    e = mod.config.num_experts
    with torch.no_grad():
        mod.gate.weight.zero_()
        for i in range(e): mod.gate.weight[i, i] = 1.0
    n = len(assign)
    x = torch.zeros(1, n, hidden_size)
    for t, a in enumerate(assign): x[0, t, a] = 10.0
    return x


class patched:
    """把分组路径强制打开，并换成 CPU 参考 kernel。"""
    def __enter__(self):
        self.saved = (mm._grouped_mm, mm._grouped_ok, mm._GROUPED_MIN_ROWS)
        mm._grouped_mm = ref_grouped_mm
        mm._grouped_ok = lambda x, config: getattr(config, 'moe_grouped_gemm', False)
        mm._GROUPED_MIN_ROWS = 0
        return self
    def __exit__(self, *a):
        mm._grouped_mm, mm._grouped_ok, mm._GROUPED_MIN_ROWS = self.saved


def pair(seed=3, **kw):
    """一对权重完全相同的模块：a=上游循环(开关关)，b=融合布局(开关开)，经 state_dict 同步。"""
    a = build(seed=seed, **kw)
    b = build(seed=seed, moe_grouped_gemm=True, **kw)
    b.load_state_dict(a.state_dict())  # 走 load 前置钩子，本身就是一次往返验证
    a.train(); b.train()
    return a, b


def fused_expert_w(b, e, name):
    """从融合参数里取出第 e 个专家的等价 nn.Linear 权重。"""
    i = b.config.moe_intermediate_size
    if name == 'down': return b.down_proj[e].t()
    return b.gate_up_proj[e][:, :i].t() if name == 'gate' else b.gate_up_proj[e][:, i:].t()


# ============================== 用例 / tests ==============================

def test_flag_off_default_and_structure():
    """C4: 新开关默认关闭，且模块结构与 state_dict 键名完全未变。"""
    c = cfg()
    assert c.moe_grouped_gemm is False, 'moe_grouped_gemm 必须默认关闭'
    assert MiniMindConfig(use_moe=True).moe_grouped_gemm is False
    keys = set(build().state_dict().keys())
    expect = {'gate.weight'} | {f'experts.{e}.{p}_proj.weight' for e in range(4) for p in ('gate', 'up', 'down')}
    assert keys == expect, f'state_dict 键名变化: {keys ^ expect}'
    m = build()
    assert m.experts[0].gate_proj.weight.shape == (128, 64)
    assert m.experts[0].down_proj.weight.shape == (64, 128)


def test_flag_off_bit_identical():
    """C4: 开关关闭时，输出与上游循环逐位一致（train 与 eval 都验）。"""
    for training in (True, False):
        m = build(); m.train(training)
        torch.manual_seed(7)
        x = torch.randn(2, 16, 64)
        got, want = m(x), reference_forward(m, x)
        assert torch.equal(got, want), f'training={training} 下与上游输出不一致'


def test_flag_off_keeps_ddp_dummy_grad():
    """循环路径的补零梯度分支仍然存在：零token专家也必须拿到（全零）梯度。"""
    m = build(); m.train()
    x = force_routing(m, [0, 0, 0, 0], 64)
    m(x).sum().backward()
    g = m.experts[3].gate_proj.weight.grad
    assert g is not None, '零token专家梯度为 None → DDP(find_unused_parameters=False) 会报错'
    assert torch.equal(g, torch.zeros_like(g)), '零token专家梯度应当恰好为零'


def test_checkpoint_roundtrip_both_directions():
    """开关开/关的 checkpoint 必须双向互换：磁盘格式始终是上游逐专家的键名。"""
    a, b = pair(seed=3)
    assert sorted(a.state_dict().keys()) == sorted(b.state_dict().keys()), '融合模型导出的键名必须与上游一致'
    for k, v in a.state_dict().items():
        assert v.shape == b.state_dict()[k].shape, f'{k} 形状不一致'
        assert torch.equal(v, b.state_dict()[k]), f'{k} 往返后数值不一致'
    for e in range(4):  # 融合张量里切出来的权重必须与上游 Linear 权重逐位相同
        for p in ('gate', 'up', 'down'):
            w = getattr(a.experts[e], f'{p}_proj').weight
            assert torch.equal(w, fused_expert_w(b, e, p)), f'experts.{e}.{p}_proj 折叠/展开不一致'
    c = build(seed=99)  # 反向：关闭开关的模型加载融合模型导出的 state_dict
    r = c.load_state_dict(b.state_dict(), strict=True)
    assert not r.missing_keys and not r.unexpected_keys, f'反向加载有缺失/多余键: {r}'
    assert torch.equal(c.experts[2].up_proj.weight, a.experts[2].up_proj.weight)


def test_partial_expert_keys_raise():
    """checkpoint 残缺时必须显式报错，而不是静默留在随机初始化上（init_model 用的是 strict=False）。"""
    a, b = pair(seed=3)
    sd = a.state_dict(); sd.pop('experts.1.up_proj.weight')
    for strict in (True, False):
        try:
            build(seed=1, moe_grouped_gemm=True).load_state_dict(sd, strict=strict)
            assert False, f'strict={strict} 时残缺 checkpoint 应当报错'
        except RuntimeError as ex:
            assert '不完整' in str(ex) or 'Missing' in str(ex), f'报错信息不明确: {ex}'


def test_grouped_matches_loop():
    """C1/C6: 分组路径与循环路径在 k 与 norm_topk_prob 各组合下数值一致，aux_loss 完全相同。"""
    for k in (1, 2, 4):
        for norm in (True, False):
            a, b = pair(seed=3, num_experts_per_tok=k, norm_topk_prob=norm)
            torch.manual_seed(11)
            x = torch.randn(2, 32, 64)
            ya = a(x)
            with patched(): yb = b(x)
            assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), \
                f'k={k} norm={norm} 输出不符, max|Δ|={(ya - yb).abs().max():.3e}'
            assert torch.equal(a.aux_loss, b.aux_loss), f'k={k} norm={norm} aux_loss 不一致'


def test_fused_loop_fallback_matches_loop():
    """开关打开但环境不支持分组kernel时的回退路径，也必须与上游循环一致。"""
    a, b = pair(seed=4)
    torch.manual_seed(11)
    x = torch.randn(2, 32, 64)
    ya, yb = a(x), b(x)  # CPU 上能力检查为 False → 自动走 _fused_loop_forward
    assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), f'回退路径不符, max|Δ|={(ya - yb).abs().max():.3e}'


def test_edge_cases():
    """C0: 零token专家 / 全部token到同一专家 / batch 小于专家数。"""
    cases = {
        '零token专家': [0, 1, 0, 1, 0, 1],          # 专家 2、3 拿不到 token
        '全部到专家0': [0, 0, 0, 0, 0, 0],
        'batch<专家数': [1, 2],                      # N=2 < E=4
    }
    for name, assign in cases.items():
        a, b = pair(seed=5)
        x = force_routing(a, assign, 64)
        force_routing(b, assign, 64)
        ya = a(x)
        with patched(): yb = b(x)
        assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), \
            f'{name}: 输出不符, max|Δ|={(ya - yb).abs().max():.3e}'


def test_grad_reachability_on_grouped_path():
    """融合参数是无条件参与运算的，零token专家也必然拿到（全零）梯度，否则 DDP 会挂。"""
    b = build(seed=5, moe_grouped_gemm=True); b.train()
    x = force_routing(b, [0, 0, 0, 0, 0, 0], 64)  # 只有专家0收到 token
    with patched(): b(x).sum().backward()
    for name in ('gate_up_proj', 'down_proj'):
        g = getattr(b, name).grad
        assert g is not None, f'{name} 梯度为 None → DDP(find_unused_parameters=False) 会报错'
        assert g.shape == getattr(b, name).shape, f'{name} 梯度形状不对'
    g = b.gate_up_proj.grad
    assert g[0].abs().sum() > 0, '收到 token 的专家梯度不应为零'
    for e in (1, 2, 3):
        assert torch.equal(g[e], torch.zeros_like(g[e])), f'未收到 token 的专家 {e} 梯度应恰好为零'


def test_capability_gate_falls_back():
    """C2: 开关打开但环境不支持（CPU / 无 _grouped_mm）时，必须安静回退且结果与循环一致。"""
    a, m = pair(seed=9)
    torch.manual_seed(13)
    x = torch.randn(2, 16, 64)
    assert mm._grouped_ok(x.view(-1, 64), m.config) is False, 'CPU 上能力检查必须返回 False'
    assert torch.allclose(m(x), a(x), atol=1e-5), '回退路径输出与上游循环不一致'

    saved = torch._grouped_mm  # 模拟老版本 torch 没有该私有算子
    try:
        del torch._grouped_mm
        assert mm._grouped_ok(x.view(-1, 64), m.config) is False
        assert torch.allclose(m(x), a(x), atol=1e-5)
    finally:
        torch._grouped_mm = saved


def test_min_rows_threshold_keeps_decode_on_loop():
    """decode 时 N=1，行数远低于阈值，必须留在循环路径（否则每 token 都要堆一次权重）。"""
    assert mm._GROUPED_MIN_ROWS >= 256, '阈值过低会让单 token 解码走排序/分组的固定开销'
    a, m = pair(seed=9)
    a.eval(); m.eval()
    x = torch.randn(1, 1, 64)
    assert (1 * 1 * m.config.num_experts_per_tok) < mm._GROUPED_MIN_ROWS
    with patched():  # 即使能力检查通过，行数不足也必须留在循环路径
        assert torch.allclose(m(x), a(x), atol=1e-5)


def test_cuda_real_kernel():
    """C1/C2 在真实 torch._grouped_mm 上的版本；无 CUDA/bf16 则跳过。

    前向要求逐位一致（torch.equal）。反向做不到逐位一致，原因明确：grad_w = xᵀ@go 是
    在 token 维上的归约，分组kernel 的 tiling/split-K 与逐专家 cuBLAS 不同，bf16 下
    累加顺序不同必然产生末位差异。因此这里不比绝对容差，而是比一个更有意义的量：
    分组与循环之间的差，必须远小于循环自身相对 fp32 的 bf16 量化误差——
    即分组路径引入的噪声被现有实现已有的误差淹没。
    """
    if not torch.cuda.is_available(): raise Skip('无 CUDA')
    dev = 'cuda'
    a = build(seed=17, hidden_size=768, moe_intermediate_size=2432).to(dev)
    b = build(seed=17, hidden_size=768, moe_intermediate_size=2432, moe_grouped_gemm=True).to(dev)
    b.load_state_dict(a.state_dict())
    a.train(); b.train()
    torch.manual_seed(19)
    x = torch.randn(4, 512, 768, device=dev)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if not mm._grouped_ok(x.view(-1, 768), b.config): raise Skip('本机 torch._grouped_mm 不可用（能力探测失败）')
        ya, yb = a(x), b(x)
    assert torch.equal(a.aux_loss, b.aux_loss), 'aux_loss 必须完全一致'
    assert torch.equal(ya, yb), f'前向必须逐位一致, max|Δ|={(ya - yb).abs().max().item():.3e}'
    print(f'    [cuda] 前向 bit-exact={torch.equal(ya, yb)}')

    def grads(m, autocast):
        m.zero_grad(set_to_none=True)
        if autocast:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16): m(x).sum().backward()
        else: m(x).sum().backward()
        return {f'{e}.{p}': getattr(m.experts[e], f'{p}_proj').weight.grad.detach().float().clone()
                for e in range(4) for p in ('gate', 'up', 'down')}

    ref = grads(build(seed=17, hidden_size=768, moe_intermediate_size=2432).to(dev), False)  # fp32 循环 = 基准
    gl = grads(a, True)
    b.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16): b(x).sum().backward()
    i = b.config.moe_intermediate_size
    gg = {f'{e}.gate': b.gate_up_proj.grad[e][:, :i].t().float().clone() for e in range(4)}
    gg.update({f'{e}.up': b.gate_up_proj.grad[e][:, i:].t().float().clone() for e in range(4)})
    gg.update({f'{e}.down': b.down_proj.grad[e].t().float().clone() for e in range(4)})
    scale = {k: max(v.abs().max().item(), 1e-12) for k, v in ref.items()}
    e_loop = max((gl[k] - ref[k]).abs().max().item() / scale[k] for k in ref)
    e_grp = max((gg[k] - ref[k]).abs().max().item() / scale[k] for k in ref)
    e_diff = max((gl[k] - gg[k]).abs().max().item() / scale[k] for k in ref)
    print(f'    [cuda] 反向相对 fp32: 循环={e_loop:.3e} 分组={e_grp:.3e}; 两者之差={e_diff:.3e}')
    assert b.gate_up_proj.grad is not None and b.down_proj.grad is not None, '融合参数梯度为 None'
    assert e_grp <= e_loop * 1.5, f'分组路径反向误差显著大于循环: {e_grp:.3e} vs {e_loop:.3e}'
    assert e_diff < e_loop, f'两条路径之差({e_diff:.3e})应远小于 bf16 自身误差({e_loop:.3e})'


# ============================== 排序分发 / sorted dispatch ==============================

def sorted_pair(seed=3, **kw):
    """一对权重相同的模块：a=上游循环，b=排序分发。二者应逐位相同（不是"在容差内"）。"""
    a = build(seed=seed, **kw)
    b = build(seed=seed, moe_sorted_dispatch=True, **kw)
    b.load_state_dict(a.state_dict())  # 排序分发不改变 state_dict 结构，strict 加载必须成功
    return a, b


def test_sorted_default_off_and_structure():
    """新开关默认关闭；打开后模块结构与 state_dict 键名仍与上游完全一致。"""
    assert MiniMindConfig(use_moe=True).moe_sorted_dispatch is False, 'moe_sorted_dispatch 必须默认关闭'
    expect = {'gate.weight'} | {f'experts.{e}.{p}_proj.weight' for e in range(4) for p in ('gate', 'up', 'down')}
    keys = set(build(moe_sorted_dispatch=True).state_dict().keys())
    assert keys == expect, f'排序分发不应改变 state_dict: {keys ^ expect}'


def test_sorted_bit_identical_to_loop():
    """排序分发与上游循环逐位相同：前向、输入梯度、每个参数梯度，train/eval × k=1,2 全覆盖。"""
    for e, k in ((1, 1), (2, 1), (4, 1), (4, 2), (8, 2)):
        a, b = sorted_pair(seed=3, num_experts=e, num_experts_per_tok=k)
        torch.manual_seed(21)
        x = torch.randn(3, 11, 64)
        for train in (True, False):
            a.train(train); b.train(train)
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya, yb = a(xa), b(xb)
            assert torch.equal(ya, yb), f'E={e} k={k} train={train} 前向不是逐位相同'
            assert torch.equal(a.aux_loss, b.aux_loss), f'E={e} k={k} aux_loss 不一致'
            (ya.sum() + a.aux_loss).backward(); (yb.sum() + b.aux_loss).backward()
            assert torch.equal(xa.grad, xb.grad), f'E={e} k={k} train={train} 输入梯度不是逐位相同'
            for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                assert (pa.grad is None) == (pb.grad is None), f'{n}: 梯度有无不一致'
                if pa.grad is not None:
                    assert torch.equal(pa.grad, pb.grad), f'E={e} k={k} {n} 梯度不是逐位相同'
            a.zero_grad(); b.zero_grad()


def test_sorted_edge_cases_and_ddp_grad():
    """零token专家 / 全部到一个专家 / batch<专家数；且空专家仍须拿到梯度，否则 DDP 会挂。"""
    for name, assign in {'零token专家': [0, 1, 0, 1, 0, 1], '全部到专家0': [0] * 6, 'batch<专家数': [1, 2]}.items():
        a, b = sorted_pair(seed=5)
        a.train(); b.train()
        x = force_routing(a, assign, 64); force_routing(b, assign, 64)
        assert torch.equal(a(x), b(x)), f'{name}: 排序分发输出与循环不是逐位相同'
    b = build(seed=5, moe_sorted_dispatch=True); b.train()
    x = force_routing(b, [0] * 6, 64)  # 只有专家0收到 token
    b(x).sum().backward()
    missing = [n for n, p in b.named_parameters() if p.grad is None]
    assert not missing, f'{missing} 梯度为 None → DDP(find_unused_parameters=False) 会报错'


def test_sorted_host_sync_count():
    """本贡献的核心断言：上游循环每层约 3E 次 device→host 同步，排序分发恒为 1 次。"""
    if not torch.cuda.is_available(): raise Skip('需要 CUDA')
    import warnings
    def syncs(mod, x):
        mod(x); torch.cuda.synchronize()  # 预热，避免把首次分配算进去
        torch.cuda.set_sync_debug_mode('warn')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always'); mod(x)
        torch.cuda.set_sync_debug_mode('default')
        return sum('sync' in str(r.message).lower() for r in w)
    for e in (4, 8):
        a, b = sorted_pair(seed=3, num_experts=e)
        a, b = a.cuda().train(), b.cuda().train()
        x = torch.randn(2, 64, 64, device='cuda')
        na, nb = syncs(a, x), syncs(b, x)
        assert na >= 3 * e, f'E={e}: 上游循环同步数 {na}，预期约 3E={3 * e}'
        assert nb == 1, f'E={e}: 排序分发同步数应恒为 1，实测 {nb}'


def test_sorted_cuda_bit_identical():
    """CUDA 上（fp32 与 bf16 autocast）同样必须逐位相同——排序改变了 kernel 的分块方式。"""
    if not torch.cuda.is_available(): raise Skip('需要 CUDA')
    for e, k in ((4, 1), (8, 2)):
        a, b = sorted_pair(seed=3, num_experts=e, num_experts_per_tok=k)
        a, b = a.cuda().train(), b.cuda().train()
        torch.manual_seed(21)
        x = torch.randn(4, 128, 64, device='cuda')
        assert torch.equal(a(x), b(x)), f'E={e} k={k} CUDA fp32 前向不是逐位相同'
        with torch.autocast('cuda', dtype=torch.bfloat16):
            assert torch.equal(a(x), b(x)), f'E={e} k={k} CUDA bf16 前向不是逐位相同'


# ============================== runner ==============================

def test_init_bit_identical_from_scratch():
    """同一随机种子下，开关开/关必须产生逐位相同的初始权重（融合分支的 RNG 消耗顺序要一致）。"""
    a, b = build(seed=1234), build(seed=1234, moe_grouped_gemm=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(sa) == sorted(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f'{k} 从零初始化不一致 → 融合分支消耗 RNG 的顺序与上游不同'


TESTS = [test_flag_off_default_and_structure, test_flag_off_bit_identical,
         test_flag_off_keeps_ddp_dummy_grad, test_init_bit_identical_from_scratch,
         test_checkpoint_roundtrip_both_directions, test_partial_expert_keys_raise,
         test_grouped_matches_loop, test_fused_loop_fallback_matches_loop, test_edge_cases,
         test_grad_reachability_on_grouped_path, test_capability_gate_falls_back,
         test_min_rows_threshold_keeps_decode_on_loop, test_cuda_real_kernel,
         test_sorted_default_off_and_structure, test_sorted_bit_identical_to_loop,
         test_sorted_edge_cases_and_ddp_grad, test_sorted_host_sync_count,
         test_sorted_cuda_bit_identical]


def main():
    print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  '
          f'_grouped_mm={hasattr(torch, "_grouped_mm")}\n')
    fails = skips = 0
    for t in TESTS:
        try:
            t(); print(f'  PASS  {t.__name__}')
        except Skip as s:
            skips += 1; print(f'  SKIP  {t.__name__} ({s})')
        except Exception:
            fails += 1; print(f'  FAIL  {t.__name__}'); traceback.print_exc()
    print(f'\n{len(TESTS) - fails - skips} passed, {fails} failed, {skips} skipped')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
