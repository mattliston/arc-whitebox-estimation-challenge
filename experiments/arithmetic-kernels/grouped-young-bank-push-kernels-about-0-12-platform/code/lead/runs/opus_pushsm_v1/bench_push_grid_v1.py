"""[opus-pushsm] NATIVE (real numerics, in-process flopscope, OpenBLAS 1 thread) cost of the young-bank push products on a GRID:
incumbent _mm vs Smirnov <3,3,6;40> (signed_executable_0) at inner L5, for every (m, k) in the given lists and n in the given list.
Per variant: FLOPs (context), process instructions / cycles (proc_pid_rusage: load-independent work counts) and wall, median of 2
measured runs after one warm-up. Smirnov families are bound per shape (P = 10) on the module's own arena (the incumbent's scratch);
outer stacks and padding are separate arrays.
usage: nice -n 10 <openblas python> bench_push_grid_v1.py <tag> <m list> <k list> <n list>"""
import hashlib, json, math, os, statistics, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_pushsm_v1'
sys.path.insert(0, str(O/'release')); import rusage
sys.path.insert(0, str(O/'lead/runs/astra_smirnov_cfull_v1')); import smirnov_cfull_v1 as h
SRC = O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py'
assert hashlib.sha256(SRC.read_bytes()).hexdigest().startswith('46e9315d')
assert os.getpriority(os.PRIO_PROCESS, 0) >= 10 and os.environ.get('OPENBLAS_NUM_THREADS') == '1'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
M = types.ModuleType('bench_candidate'); M.__file__ = str(SRC); sys.modules[M.__name__] = M
exec(compile(SRC.read_text(), str(SRC), 'exec', dont_inherit=True), M.__dict__)
M._PB_WALL_S = 1e9
est = M.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
arena = M._SCRATCH[0]
rec = [r for r in json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']
       if r['name'] == 'signed_executable_0'][0]
SLOTS = 400


def group_views(self, kind, shape):
    key = (self.record['name'], kind, tuple(shape))
    cache = self.M._sm40_group_cache
    if key not in cache:
        groups = self.circuits[kind][1]
        sizes = [len(nodes) for _, nodes in groups]
        total = sum(sizes); rows, cols = map(int, shape)
        allnodes = fnp.reshape(arena[:total*rows*cols], (total, rows, cols))   # exact-size node views at the arena's start
        cuts = []; at = 0
        for size in sizes[:-1]:
            at += size; cuts.append(at)
        cache[key] = (tuple(fnp.split(allnodes, cuts, axis=0)), tuple(fnp.unstack(allnodes, axis=0)))
    return cache[key]


h.Kernel.group_views = group_views
kernel = h.Kernel(M, rec)


def bind(f):
    L = f[1]; mr, mk, mc = f[3] << L, f[4] << L, f[5] << L
    h.FAMILIES = (f,)
    M._sm40_owner = None; M._sm40_cache = {}; M._sm40_io_cache = {}; M._sm40_outer_cache = {}; M._sm40_ccrop_cache = {}
    M._sm40_frames = {f[0]: h.bind_family(fnp, M, arena, f)}
    M._sm40_extra_leaf = None; M._sm40_extra_leaf_bytes = 0
    M._sm40_outer = dict(A=fnp.empty((40, mr, mk), dtype=fnp.float32), B=fnp.empty((40, mk, mc), dtype=fnp.float32),
                         C=fnp.empty((40, mr, mc), dtype=fnp.float32))
    M._sm40_outer_bytes = 40 * (mr*mk + mk*mc + mr*mc) * 4
    M._sm40_group_cache = {}
    M._sm40_padding = (fnp.empty((3*mr, 3*mk), dtype=fnp.float32), fnp.empty((3*mk, 6*mc), dtype=fnp.float32))
    M._sm40_padding_grid = {}; M._sm40_padding_cache = {}
    h.bind_c_full(M, arena)
    M._sm40_owner = arena


def run_counts(fn):
    M._BH_HOLD.clear(); M._TR_LAST[0] = None
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        c0, i0 = rusage.raw(); t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter(); c1, i1 = rusage.raw()
    return dict(flops=int(ctx.flops_used), instructions=i1 - i0, cycles=c1 - c0, wall=t1 - t0), out


def measured(fn):
    run_counts(fn)
    runs = [run_counts(fn)[0] for _ in range(2)]
    return {k: (runs[0][k] if k == 'flops' else statistics.median(r[k] for r in runs)) for k in runs[0]}


ms, ks, ns = ([int(x) for x in a.split(',') if x] for a in sys.argv[2:5])
shapes = [(m, k, n) for m in ms for k in ks for n in ns]
rng = np.random.default_rng(7)
rows, t_start = [], time.monotonic()
for (m, k, n) in shapes:
    with flops.BudgetContext(flop_budget=2**60, quiet=True):
        a = fnp.asarray(rng.standard_normal((m, k), dtype=np.float32)); b = fnp.asarray(rng.standard_normal((k, n), dtype=np.float32))
    inc = measured(lambda: M._mm(a, b))
    row = dict(shape=[m, k, n], incumbent=inc)
    for L in (5,):
        u = 1 << L
        f = ('push', L, 10, math.ceil(m/3/u), math.ceil(k/3/u), math.ceil(n/6/u))
        if 7 ** L * 10 * (f[3]*f[4] + f[4]*f[5] + f[3]*f[5]) > int(arena.shape[0]):
            row[f'L{L}'] = None
            continue
        bind(f)
        row[f'L{L}'] = measured(lambda: kernel(a, b, L, 10))
        M._sm40_outer = {}; M._sm40_padding = None
    rows.append(row)
    print(json.dumps(dict(shape=[m, k, n], inc=inc, L5=row.get('L5'), L6=row.get('L6'), s=round(time.monotonic() - t_start, 1))), flush=True)
(D/f'BENCHGRID_{sys.argv[1]}.json').write_text(json.dumps(dict(tag=sys.argv[1], rows=rows), indent=1) + '\n')
