"""MoE 前馈层分发性能基准 / MoE feed-forward dispatch benchmark.

自包含脚本，不引入任何新依赖。模式：
  compile-gate  对照 dense / loop / loop+torch.compile，判断 torch.compile 是否已经消除了差距
  sweep         完整扫描：专家数 × 形状 × 各实现（整步训练）
  layer         单个 MoE 层隔离测量
  sync-probe    直接测本机的 device→host 同步开销，用来解释各机器之间的加速比差异

三个实现分支（消融用，三者数学等价）：
  loop     上游实现，每个专家 mask.any()/nonzero()/布尔索引，每层约 3E 次同步
  sorted   先按专家排序再循环，每层 1 次同步；纯 PyTorch，无硬件要求（moe_sorted_dispatch）
  grouped  排序 + torch._grouped_mm 单 kernel，需 sm_90+ 与 bf16（moe_grouped_gemm）
sorted 分离了"去同步"与"换 kernel"两个变量：loop→sorted 是同步开销，sorted→grouped 是 kernel。

计时协议：≥10 次预热，每个计时区间前后 torch.cuda.synchronize()，
≥20 次独立测量取中位数并报告 min/max，每次测量固定种子、测量间变化种子。
"""
import argparse, json, os, statistics, sys, time
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ============================== 计时核心 / timing core ==============================

def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def timed(step_fn, warmup=10, reps=20, steps_per_rep=5, seeds=None, divisor=1):
    """返回 {median, min, max, reps, seeds, peak_mem_mb}，单位为秒/步。

    step_fn(seed) 执行一次训练步。预热不计时；每次测量前 sync，测量 steps_per_rep 步后再 sync。
    """
    seeds = seeds or [1234 + i for i in range(reps)]
    for i in range(warmup): step_fn(seeds[i % len(seeds)])
    _sync()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    samples = []
    for r in range(reps):
        _sync()
        t0 = time.perf_counter()
        for _ in range(steps_per_rep): step_fn(seeds[r])
        _sync()
        samples.append((time.perf_counter() - t0) / (steps_per_rep * divisor))
    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    return {
        'median': statistics.median(samples), 'min': min(samples), 'max': max(samples),
        'reps': reps, 'steps_per_rep': steps_per_rep, 'seeds': seeds, 'peak_mem_mb': peak,
    }


# ============================== 构建 / build ==============================

VARIANTS = ('loop', 'sorted', 'grouped')
VARIANT_KW = {'loop': {}, 'sorted': {'moe_sorted_dispatch': True}, 'grouped': {'moe_grouped_gemm': True}}


def build_model(args, use_moe, num_experts=None, variant='loop', device='cuda'):
    """按 minimind 默认配置构建模型；MoE 超参仅在 use_moe 时生效。"""
    kw = dict(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=use_moe)
    if use_moe:
        kw.update(num_experts=num_experts, num_experts_per_tok=args.top_k)
        kw.update(VARIANT_KW[variant])
    torch.manual_seed(0)  # 各实现权重一致
    model = MiniMindForCausalLM(MiniMindConfig(**kw)).to(device)
    model.train()
    return model


def make_step(model, args, device='cuda'):
    """一次完整训练步：fwd + loss(+aux) + bwd + AdamW.step，对齐 train_pretrain.py 的结构。

    注意：这里 accumulation_steps=1（train_pretrain 默认为 8），因此 "一步" 含一次优化器更新。
    """
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    vocab = model.config.vocab_size

    accum = max(1, args.accum)

    def step(seed):  # 计时口径 = 一个 micro-batch；优化器按 accumulation_steps 摊薄，与 train_pretrain 一致
        torch.manual_seed(seed)
        for j in range(accum):
            ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(ids, labels=ids)
                loss = (res.loss + res.aux_loss) / accum
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    return (lambda seed: step(seed)) if accum == 1 else step


# ============================== 模式：compile-gate ==============================

def mode_compile_gate(args):
    """Phase 0 杀死门：torch.compile 是否已经补上了 README 所称的 ~50% 差距？

    若 compiled loop 在 4 experts 下已接近 dense（默认阈值 15%），则本贡献是多余的。
    """
    out = {'mode': 'compile-gate', 'env': env_info(), 'results': {}}

    dense = build_model(args, use_moe=False)
    out['results']['dense'] = timed(make_step(dense, args), args.warmup, args.reps, divisor=max(1, args.accum))
    del dense; torch.cuda.empty_cache()
    print(f"dense                : {fmt(out['results']['dense'])}")

    for e in args.experts:
        m = build_model(args, use_moe=True, num_experts=e)
        key = f'loop_e{e}'
        out['results'][key] = timed(make_step(m, args), args.warmup, args.reps, divisor=max(1, args.accum))
        del m; torch.cuda.empty_cache()
        print(f"moe loop   e={e:<3d}    : {fmt(out['results'][key])}")

        m = build_model(args, use_moe=True, num_experts=e)
        torch._dynamo.reset()
        _clear_graph_break_counter()
        ckey = f'compile_e{e}'
        try:
            cm = torch.compile(m)
            out['results'][ckey] = timed(make_step(cm, args), args.warmup, args.reps, divisor=max(1, args.accum))
            out['results'][ckey]['graph_breaks'] = _graph_break_count()
            print(f"moe compile e={e:<3d}   : {fmt(out['results'][ckey])}  graph_breaks={out['results'][ckey]['graph_breaks']}")
        except Exception as ex:  # 编译失败本身就是结果，如实记录
            out['results'][ckey] = {'error': f'{type(ex).__name__}: {ex}'}
            print(f"moe compile e={e:<3d}   : FAILED {type(ex).__name__}: {ex}")
        del m; torch.cuda.empty_cache()

    # 判定 / verdict
    d = out['results']['dense']['median']
    verdict = {}
    for e in args.experts:
        for tag in ('loop', 'compile'):
            r = out['results'].get(f'{tag}_e{e}', {})
            if 'median' in r: verdict[f'{tag}_e{e}_vs_dense'] = r['median'] / d
    out['verdict'] = verdict
    gate_e = args.experts[0]
    c = verdict.get(f'compile_e{gate_e}_vs_dense')
    out['kill_gate'] = {
        'threshold': 1.0 + args.gate_margin, 'compiled_vs_dense': c,
        'killed': bool(c is not None and c <= 1.0 + args.gate_margin),
        'note': f'compiled MoE loop within {args.gate_margin:.0%} of dense at e={gate_e} => contribution redundant',
    }
    print('\n--- vs dense (lower is better, 1.00 = dense parity) ---')
    for k, v in verdict.items(): print(f'  {k:<28s} {v:.3f}x')
    print(f"\nKILL GATE: {'TRIGGERED — stop' if out['kill_gate']['killed'] else 'not triggered — proceed'}")
    return out


# ============================== 模式：sweep（整步训练） ==============================

def _assert_path(model, args, variant):
    """确认真的走到了预期的分支，避免把同一条路径测了两遍。"""
    from model.model_minimind import MOEFeedForward, _grouped_ok, _GROUPED_MIN_ROWS
    mlp = [m for m in model.modules() if isinstance(m, MOEFeedForward)][0]
    x = torch.zeros(8, args.hidden_size, device='cuda', dtype=torch.bfloat16)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        ok = _grouped_ok(x, mlp.config)
    rows = args.batch_size * args.seq_len * args.top_k
    assert ok == (variant == 'grouped'), f'分组路径生效状态与预期不符: got {ok}, want {variant == "grouped"}'
    assert mlp.sorted_dispatch == (variant == 'sorted'), f'排序路径生效状态与预期不符: want {variant}'
    if variant == 'grouped': assert rows >= _GROUPED_MIN_ROWS, f'行数 {rows} 低于阈值 {_GROUPED_MIN_ROWS}，实际会走循环'


def mode_sweep(args):
    """(b) 整步训练时间 + (c) tokens/s + 峰值显存，扫描专家数 × 形状。"""
    out = {'mode': 'sweep', 'env': env_info(), 'results': {}}
    shapes = args.shapes
    for (bs, sl) in shapes:
        a = argparse.Namespace(**{**vars(args), 'batch_size': bs, 'seq_len': sl})
        dm = build_model(a, use_moe=False)
        r = timed(make_step(dm, a), args.warmup, args.reps, divisor=max(1, args.accum))
        r['tokens_per_s'] = bs * sl / r['median']
        out['results'][f'dense|b{bs}s{sl}'] = r
        del dm; torch.cuda.empty_cache()
        print(f"b{bs}xs{sl} dense           : {fmt(r)}  {r['tokens_per_s']:>9.0f} tok/s")
        for e in args.experts:
            for variant in args.variants:
                key = f"{variant}|e{e}|b{bs}s{sl}"
                try:
                    m = build_model(a, use_moe=True, num_experts=e, variant=variant)
                    _assert_path(m, a, variant)
                    r = timed(make_step(m, a), args.warmup, args.reps, divisor=max(1, args.accum))
                    r['tokens_per_s'] = bs * sl / r['median']
                    out['results'][key] = r
                    print(f"b{bs}xs{sl} {variant:<7s} e={e:<3d}: {fmt(r)}  {r['tokens_per_s']:>9.0f} tok/s")
                except (torch.cuda.OutOfMemoryError, AssertionError) as ex:
                    out['results'][key] = {'error': f'{type(ex).__name__}: {ex}'}
                    print(f"b{bs}xs{sl} {variant:<7s} e={e:<3d}: SKIP {type(ex).__name__}: {ex}")
                finally:
                    m = None; torch.cuda.empty_cache()
    _speedups(out, args, shapes)
    return out


def _speedups(out, args, shapes):
    """消融口径：sorted/loop 是去掉同步的收益，grouped/sorted 是换 kernel 的额外收益。"""
    sp = {}
    for (bs, sl) in shapes:
        for e in args.experts:
            g = lambda v: out['results'].get(f'{v}|e{e}|b{bs}s{sl}', {})
            lo, so, gr, dn = g('loop'), g('sorted'), g('grouped'), out['results'].get(f'dense|b{bs}s{sl}', {})
            if 'median' not in lo: continue
            rec = {'loop_vs_dense': lo['median'] / dn['median'] if 'median' in dn else None}
            if 'median' in so:
                rec['sorted_speedup_vs_loop'] = lo['median'] / so['median']
                rec['sorted_peak_mem_ratio'] = so['peak_mem_mb'] / lo['peak_mem_mb'] if lo.get('peak_mem_mb') else None
            if 'median' in gr:
                rec['grouped_speedup_vs_loop'] = lo['median'] / gr['median']
                rec['grouped_peak_mem_ratio'] = gr['peak_mem_mb'] / lo['peak_mem_mb'] if lo.get('peak_mem_mb') else None
                if 'median' in so: rec['grouped_speedup_vs_sorted'] = so['median'] / gr['median']
            sp[f'e{e}|b{bs}s{sl}'] = rec
    out['speedups'] = sp
    print('\n--- 整步训练加速比（消融） ---')
    print(f"  {'config':<20s} {'sorted/loop':>12s} {'grouped/loop':>13s} {'grouped/sorted':>15s} {'loop/dense':>11s}")
    for k, v in sp.items():
        f = lambda x: f'{v[x]:.3f}x' if v.get(x) else '   -   '
        print(f"  {k:<20s} {f('sorted_speedup_vs_loop'):>12s} {f('grouped_speedup_vs_loop'):>13s} "
              f"{f('grouped_speedup_vs_sorted'):>15s} {f('loop_vs_dense'):>11s}")


# ============================== 模式：layer（单层隔离） ==============================

def mode_layer(args):
    """(a) 单个 MoE 层的前向 / 前向+反向 / 推理，可以扫到整模型放不下的专家数。"""
    from model.model_minimind import MOEFeedForward
    out = {'mode': 'layer', 'env': env_info(), 'results': {}}
    n = args.batch_size * args.seq_len
    for e in args.experts:
        for variant in args.variants:
            kw = dict(hidden_size=args.hidden_size, num_hidden_layers=1, use_moe=True, num_experts=e,
                      num_experts_per_tok=args.top_k, **VARIANT_KW[variant])
            torch.manual_seed(0)
            mlp = MOEFeedForward(MiniMindConfig(**kw)).cuda()
            tag = variant
            for phase in ('fwd', 'fwd_bwd', 'infer'):
                mlp.train(phase != 'infer')
                def step(seed, phase=phase, mlp=mlp):
                    torch.manual_seed(seed)
                    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device='cuda')
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        if phase == 'infer':
                            with torch.no_grad(): mlp(x)
                        elif phase == 'fwd':
                            mlp(x)
                        else:
                            mlp(x).sum().backward(); mlp.zero_grad(set_to_none=True)
                try:
                    r = timed(step, args.warmup, args.reps)
                    r['tokens_per_s'] = n / r['median']
                    out['results'][f'{variant}|e{e}|{phase}'] = r
                    print(f"e={e:<3d} {tag:<7s} {phase:<8s}: {fmt(r)}  {r['tokens_per_s']:>10.0f} tok/s")
                except torch.cuda.OutOfMemoryError:
                    print(f"e={e:<3d} {tag:<7s} {phase:<8s}: OOM")
                    out['results'][f'{variant}|e{e}|{phase}'] = {'error': 'OOM'}
            del mlp; torch.cuda.empty_cache()
    print('\n--- 单层加速比（消融，基准=loop） ---')
    sp = {}
    for e in args.experts:
        for phase in ('fwd', 'fwd_bwd', 'infer'):
            lo = out['results'].get(f'loop|e{e}|{phase}', {})
            if 'median' not in lo: continue
            rec = {}
            for v in ('sorted', 'grouped'):
                r = out['results'].get(f'{v}|e{e}|{phase}', {})
                if 'median' in r: rec[v] = lo['median'] / r['median']
            if rec:
                sp[f'e{e}|{phase}'] = rec
                print(f"  e={e:<3d} {phase:<8s} " + '  '.join(f'{v}={x:.3f}x' for v, x in rec.items()))
    out['speedups'] = sp
    return out


# ============================== 模式：sync-probe ==============================

def mode_sync_probe(args):
    """直接测本机一次 device→host 同步的往返开销。

    上游循环每层要做约 3E 次这样的往返，所以 loop 基准的耗时（进而 sorted/grouped 的加速比）
    与本机 CPU/PCIe 延迟强相关。报告这个数值，读者就能预估自己机器上的收益，
    而不是只看到两台机器之间无法解释的差异。
    """
    out = {'mode': 'sync-probe', 'env': env_info(), 'results': {}}
    x = torch.randn(1024, 1024, device='cuda')
    flag = torch.zeros(1, dtype=torch.bool, device='cuda')

    def probe(fn, n=200):
        for _ in range(20): fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / n)
        return {'median': statistics.median(samples), 'min': min(samples), 'max': max(samples), 'n': n}

    # 空 kernel 连发（不同步）：纯 launch 开销
    out['results']['launch_only'] = probe(lambda: x.add_(0.0))
    # .item()：一次完整的 device→host 往返，等价于循环里的 mask.any()
    out['results']['sync_item'] = probe(lambda: flag.any().item(), n=100)
    # 上游循环真正的三件套：mask.any() / nonzero() / 布尔索引
    idx = torch.randint(0, max(2, args.experts[0]), (args.batch_size * args.seq_len, args.top_k), device='cuda')
    w = torch.randn_like(idx, dtype=torch.float)

    def triple():
        m = (idx == 0)
        if m.any(): m.any(dim=-1).nonzero().flatten(); w[m]
    out['results']['loop_triple'] = probe(triple, n=100)

    for k, v in out['results'].items():
        print(f"  {k:<14s} median {v['median'] * 1e6:8.2f} us  [min {v['min'] * 1e6:7.2f}, max {v['max'] * 1e6:7.2f}]  (n={v['n']})")
    s = out['results']['loop_triple']['median']
    print(f"\n  上游循环每个专家约 {s * 1e6:.1f} us 的主机往返 → 每层 E 个专家 ≈ {s * 1e6:.1f}×E us，"
          f"8 层 ≈ {s * 8 * 1e6:.1f}×E us/步")
    out['per_expert_host_cost_us'] = s * 1e6
    return out


def _clear_graph_break_counter():
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception: pass


def _graph_break_count():
    try:
        from torch._dynamo.utils import counters
        return sum(counters.get('graph_break', {}).values())
    except Exception: return None


# ============================== 辅助 / helpers ==============================

def fmt(r):
    if 'error' in r: return r['error']
    return (f"median {r['median'] * 1e3:8.2f} ms  [min {r['min'] * 1e3:7.2f}, max {r['max'] * 1e3:7.2f}]  "
            f"peak {r['peak_mem_mb']:7.0f} MB")


def env_info():
    info = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
            'has_grouped_mm': hasattr(torch, '_grouped_mm')}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    cuda=torch.version.cuda)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='compile-gate', choices=['compile-gate', 'sweep', 'layer', 'sync-probe'])
    p.add_argument('--hidden_size', default=768, type=int)
    p.add_argument('--num_hidden_layers', default=8, type=int)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--seq_len', default=512, type=int)
    p.add_argument('--top_k', default=1, type=int)
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
    p.add_argument('--experts', default='4,16', type=lambda s: [int(x) for x in s.split(',')])
    p.add_argument('--shapes', default='8x512,16x512,16x1024',
                   type=lambda s: [tuple(int(v) for v in x.split('x')) for x in s.split(',')])
    p.add_argument('--variants', default='loop,sorted,grouped',
                   type=lambda s: [v for v in s.split(',')], help=f'实现分支，可选 {VARIANTS}')
    p.add_argument('--warmup', default=10, type=int)
    p.add_argument('--reps', default=20, type=int)
    p.add_argument('--accum', default=1, type=int, help='梯度累积步数；train_pretrain 默认为 8')
    p.add_argument('--gate_margin', default=0.15, type=float)
    p.add_argument('--out', default=None, help='写入 JSON 结果的路径')
    args = p.parse_args()
    bad = [v for v in args.variants if v not in VARIANTS]
    if bad: p.error(f'未知实现分支 {bad}，可选 {VARIANTS}')

    print(json.dumps(env_info(), indent=2))
    if not torch.cuda.is_available():
        print('\nERROR: 本脚本需要 CUDA。/ This benchmark requires CUDA.'); sys.exit(1)

    out = {'compile-gate': mode_compile_gate, 'sweep': mode_sweep, 'layer': mode_layer,
           'sync-probe': mode_sync_probe}[args.mode](args)
    out['args'] = vars(args)
    if args.out:
        with open(args.out, 'w') as f: json.dump(out, f, indent=2)
        print(f'\n结果已写入 {args.out}')


if __name__ == '__main__':
    main()
