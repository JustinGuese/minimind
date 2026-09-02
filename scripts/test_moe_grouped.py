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


def test_grouped_matches_loop():
    """C1/C6: 分组路径与循环路径在 k 与 norm_topk_prob 各组合下数值一致，aux_loss 完全相同。"""
    for k in (1, 2, 4):
        for norm in (True, False):
            a = build(seed=3, num_experts_per_tok=k, norm_topk_prob=norm)
            b = build(seed=3, num_experts_per_tok=k, norm_topk_prob=norm, moe_grouped_gemm=True)
            b.load_state_dict(a.state_dict())
            a.train(); b.train()
            torch.manual_seed(11)
            x = torch.randn(2, 32, 64)
            ya = a(x)
            with patched(): yb = b(x)
            assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), \
                f'k={k} norm={norm} 输出不符, max|Δ|={(ya - yb).abs().max():.3e}'
            assert torch.equal(a.aux_loss, b.aux_loss), f'k={k} norm={norm} aux_loss 不一致'


def test_edge_cases():
    """C0: 零token专家 / 全部token到同一专家 / batch 小于专家数。"""
    cases = {
        '零token专家': [0, 1, 0, 1, 0, 1],          # 专家 2、3 拿不到 token
        '全部到专家0': [0, 0, 0, 0, 0, 0],
        'batch<专家数': [1, 2],                      # N=2 < E=4
    }
    for name, assign in cases.items():
        a = build(seed=5)
        b = build(seed=5, moe_grouped_gemm=True); b.load_state_dict(a.state_dict())
        a.train(); b.train()
        x = force_routing(a, assign, 64)
        force_routing(b, assign, 64)
        ya = a(x)
        with patched(): yb = b(x)
        assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), \
            f'{name}: 输出不符, max|Δ|={(ya - yb).abs().max():.3e}'


def test_grad_reachability_on_grouped_path():
    """分组路径下所有专家参数都必须拿到梯度（stack 无条件遍历全部专家），否则 DDP 会挂。"""
    b = build(seed=5, moe_grouped_gemm=True); b.train()
    x = force_routing(b, [0, 0, 0, 0, 0, 0], 64)  # 只有专家0收到 token
    with patched(): b(x).sum().backward()
    for e in range(4):
        for p in ('gate', 'up', 'down'):
            g = getattr(b.experts[e], f'{p}_proj').weight.grad
            assert g is not None, f'experts.{e}.{p}_proj 梯度为 None → DDP 会报错'
    nz = b.experts[0].gate_proj.weight.grad.abs().sum()
    assert nz > 0, '收到 token 的专家梯度不应为零'
    for e in (1, 2, 3):
        g = b.experts[e].gate_proj.weight.grad
        assert torch.equal(g, torch.zeros_like(g)), f'未收到 token 的 experts.{e} 梯度应恰好为零'


def test_capability_gate_falls_back():
    """C2: 开关打开但环境不支持（CPU / 无 _grouped_mm）时，必须安静回退且结果与循环一致。"""
    m = build(seed=9, moe_grouped_gemm=True); m.train()
    torch.manual_seed(13)
    x = torch.randn(2, 16, 64)
    assert mm._grouped_ok(x.view(-1, 64), m.config) is False, 'CPU 上能力检查必须返回 False'
    assert torch.equal(m(x), reference_forward(m, x)), '回退路径输出与上游循环不一致'

    saved = torch._grouped_mm  # 模拟老版本 torch 没有该私有算子
    try:
        del torch._grouped_mm
        assert mm._grouped_ok(x.view(-1, 64), m.config) is False
        assert torch.equal(m(x), reference_forward(m, x))
    finally:
        torch._grouped_mm = saved


def test_min_rows_threshold_keeps_decode_on_loop():
    """decode 时 N=1，行数远低于阈值，必须留在循环路径（否则每 token 都要堆一次权重）。"""
    assert mm._GROUPED_MIN_ROWS >= 256, '阈值过低会让单 token 解码付出权重堆叠开销'
    m = build(seed=9, moe_grouped_gemm=True); m.eval()
    x = torch.randn(1, 1, 64)
    assert (1 * 1 * m.config.num_experts_per_tok) < mm._GROUPED_MIN_ROWS
    assert torch.equal(m(x), reference_forward(m, x))


def test_cuda_real_kernel():
    """C1/C2 在真实 torch._grouped_mm 上的版本；无 CUDA/bf16 则跳过。"""
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
    scale = ya.abs().max().item()
    rel = (ya - yb).abs().max().item() / max(scale, 1e-12)
    print(f'    [cuda] max|Δ|/max|y| = {rel:.3e}  bit-exact={torch.equal(ya, yb)}')
    assert torch.equal(a.aux_loss, b.aux_loss), 'aux_loss 必须完全一致'
    assert rel < 2e-2, f'相对误差过大: {rel:.3e}'
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        a(x).sum().backward(); b(x).sum().backward()
    for e in range(4):
        for p in ('gate', 'up', 'down'):
            ga = getattr(a.experts[e], f'{p}_proj').weight.grad
            gb = getattr(b.experts[e], f'{p}_proj').weight.grad
            assert gb is not None, f'experts.{e}.{p}_proj 梯度为 None'
            r = (ga - gb).abs().max().item() / max(ga.abs().max().item(), 1e-12)
            assert r < 5e-2, f'experts.{e}.{p}_proj 梯度相对误差过大: {r:.3e}'


# ============================== runner ==============================

TESTS = [test_flag_off_default_and_structure, test_flag_off_bit_identical,
         test_flag_off_keeps_ddp_dummy_grad, test_grouped_matches_loop, test_edge_cases,
         test_grad_reachability_on_grouped_path, test_capability_gate_falls_back,
         test_min_rows_threshold_keeps_decode_on_loop, test_cuda_real_kernel]


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
