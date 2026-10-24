"""[opus-pushsm] cost-only exact virtual pricing (no numerics) of the INCUMBENT young-bank push _mm (S2tr 46e9315d) on a grid of
(m, k, n): warm FLOPs and requests. Same harness as price_push_v1.py.
usage: nice -n 10 <openblas python> price_inc_grid_v1.py <tag> <m list> <k list> <n start> <n stop> <n step>"""
import collections, json, os, sys, time, importlib.util
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS, 0) == 10
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_pushsm_v1'
s = importlib.util.spec_from_file_location('virtual_tariff', O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p = importlib.util.module_from_spec(s); s.loader.exec_module(p)
p.SOURCE = O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py'
p.SOURCE_SHA = '46e9315d1a3e563cf669f27ec499be487ca7a437ee942b7be44ea1fad682ad9f'
M, _ = p.load(); fnp = p.fnp; fs = p.flops
with fs.BudgetContext(flop_budget=2**60, quiet=True):
    arena = fnp.empty((M._PACKED_ELEMENTS,), dtype=fnp.float32)
    M._SCRATCH = [arena, M._cp3_bind_arena_views(arena, M._PACKED_PUSH), M._cp3_bind_arena_views(arena, M._PACKED_READ), None, None, None, None]
    M._bind_push6(arena)
M._WINOGRAD = True; M._AB_ON = True
M._LVF_EAGER[0] = True; M._LVF_KEEP[0] = False


def measure(fn):
    M._BH_HOLD.clear(); M._TR_LAST[0] = None; p.CALLS.clear()
    with fs.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        fn()
    return int(ctx.flops_used), sum(p.CALLS.values())


tag = sys.argv[1]
ms = [int(x) for x in sys.argv[2].split(',')]; ks = [int(x) for x in sys.argv[3].split(',')]
ns = list(range(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])))
rows, t0 = [], time.monotonic()
for m in ms:
    for k in ks:
        for n in ns:
            with fs.BudgetContext(flop_budget=2**60, quiet=True):
                a = fnp.empty((m, k), dtype=fnp.float32); b = fnp.empty((k, n), dtype=fnp.float32)
            measure(lambda: M._mm(a, b))
            f, r = measure(lambda: M._mm(a, b))
            rows.append([m, k, n, f, r])
        print(json.dumps(dict(m=m, k=k, done=len(rows), s=round(time.monotonic() - t0, 1))), flush=True)
(D/f'INC_{tag}.json').write_text(json.dumps(dict(tag=tag, source_sha256=p.SOURCE_SHA, rows=rows)) + '\n')
