"""[opus-small] NATIVE wall price of the Smirnov winners: per winning shape, the live incumbent route vs its best certified Smirnov
executable (bound exactly as the cost-only harness binds it, on a real arena), warm, median of 3: process INSTRUCTIONS / CYCLES
(release/rusage.py, load-independent) and FLOPs; scaled by each network's logged winning calls. usage: <openblas python> wallprice_v1.py"""
import collections, json, math, os, sys, time, types, glob, statistics
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_small_v1'; sys.path.insert(0, str(O/'release')); import rusage
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
SRC = O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py'
M = types.ModuleType('cand_wall'); M.__file__ = str(SRC); sys.modules[M.__name__] = M
exec(compile(SRC.read_text(), str(SRC), 'exec', dont_inherit=True), M.__dict__)
M._PB_WALL_S = 1e9
est = M.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
class P: pass
p = P(); p.CALLS = collections.Counter()
views = collections.Counter()
def reset():
    M._BH_HOLD.clear(); M._TR_LAST[0] = None
def measure(fn):
    reset()
    with fs.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        c0, i0 = rusage.raw(); fn(); c1, i1 = rusage.raw()
    return dict(flops=int(ctx.flops_used), requests=0, cycles=c1 - c0, instructions=i1 - i0)
rng = np.random.default_rng(1)
def arrays(m, k, n):
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        return fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)), fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
sys.path.insert(0, str(O/'lead/runs/astra_smirnov_cfull_v1')); import smirnov_cfull_v1 as h
SLOTS = 400                                         # linear-circuit node slots (the read's base has 125: enough for its 3x6x3 executables only)


def group_views(self, kind, shape):                 # = h.Kernel.group_views with the slot cap of the bound base
    key = (self.record['name'], kind, tuple(shape))
    cache = self.M._sm40_group_cache
    if key not in cache:
        groups = self.circuits[kind][1]
        sizes = [len(nodes) for _, nodes in groups]
        base = self.M._sm40_group_base[1 if kind == 'R' else 0]
        total = sum(sizes); assert total <= int(base.shape[0])
        rows, cols = map(int, shape)
        assert rows <= int(base.shape[1]) and cols <= int(base.shape[2])
        allnodes = base[:total, :rows, :cols]
        cuts = []; at = 0
        for size in sizes[:-1]:
            at += size; cuts.append(at)
        cache[key] = (tuple(self.fnp.split(allnodes, cuts, axis=0)), tuple(self.fnp.unstack(allnodes, axis=0)))
    return cache[key]


h.Kernel.group_views = group_views
records = json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']
EXEC = {r['name']: r for r in records}            # two certified executables per orientation (3x3x6, 3x6x3, 6x3x3)
ORIENT = {tuple(r['shape']): None for r in records}

def blocks(shape, ori, L):
    (m, k, n), (ar, ak, an), u = shape, ori, 1 << L
    return math.ceil(m/ar/u), math.ceil(k/ak/u), math.ceil(n/an/u)

def packets(n):
    return [4096] * (n // 4096) + ([n % 4096] if n % 4096 else [])


def bind(f):
    """Bind ONE push family f = (name, L, 40, r, k, c) with exactly the blocks one priced product needs (C unwinds use family bounds)."""
    L = f[1]; u = 1 << L
    mr, mk, mc = f[3] << L, f[4] << L, f[5] << L
    side = 7 ** L * f[2] * max(f[3]*f[4], f[4]*f[5], f[3]*f[5])
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        big = fnp.empty((max(3 * side, SLOTS * max(mr, mc) * max(mk, mr, mc)),), dtype=fnp.float32)
        h.FAMILIES = (f,)
        M._sm40_owner = None; M._sm40_cache = {}; M._sm40_io_cache = {}; M._sm40_outer_cache = {}; M._sm40_ccrop_cache = {}
        M._sm40_frames = {f[0]: h.bind_family(fnp, M, big, f)}
        M._sm40_extra_leaf = None; M._sm40_extra_leaf_bytes = 0
        M._sm40_outer = dict(A=fnp.empty((40, mr, mk), dtype=fnp.float32), B=fnp.empty((40, mk, mc), dtype=fnp.float32),
                             C=fnp.empty((40, mr, mc), dtype=fnp.float32))
        M._sm40_outer_bytes = 40 * (mr*mk + mk*mc + mr*mc) * 4
        gr, gc = max(mr, mc), max(mk, mr, mc)
        base = fnp.reshape(big[:SLOTS*gr*gc], (SLOTS, gr, gc))
        M._sm40_group_base = (base, fnp.swapaxes(base, 1, 2)); M._sm40_group_cache = {}
        M._sm40_padding = (fnp.empty((6*mr, 6*mk), dtype=fnp.float32), fnp.empty((6*mk, 6*mc), dtype=fnp.float32))
        M._sm40_padding_grid = {}; M._sm40_padding_cache = {}
        h.bind_c_full(M, big)
        M._sm40_owner = big
    return big


kernels = {name: h.Kernel(M, r) for name, r in EXEC.items()}
own = M._SCRATCH[0]


def smirnov(shape, name, L, split):
    m, k, n = shape
    ori, u = tuple(EXEC[name]['shape']), 1 << L
    parts = packets(n) if split else [n]
    need = [(math.ceil(m/ori[0]/u), math.ceil(k/ori[1]/u), math.ceil(nn/ori[2]/u)) for nn in parts]
    big = bind(('push', L, 40, max(x[0] for x in need), max(x[1] for x in need), max(x[2] for x in need)))
    q = kernels[name]; operands = [arrays(m, k, nn) for nn in parts]
    M._SCRATCH[0] = big
    def run():
        for a, b in operands:
            q(a, b, L, 40)
    try:
        first = measure(run); warm = measure(run)
    finally:
        M._SCRATCH[0] = own
    return dict(first=first, warm=warm)




own = M._SCRATCH[0]
price = json.loads((D/'PRICE_small_v1.json').read_text())['best']
logs = [json.loads(open(f).read()) for f in sorted(glob.glob(str(D/'shapes/shapes_*.json')))]
SRCL = SRC.read_text().split('\n')
UVP = ('_ubc_product', '_ubx_product'); OLD = ('_slr_reconstruct', '_cp_slr_inner_sq', '_astra_bcc_recon_impl', '_astra_bcc_recon_full_impl', '_astra_bcc_lazy')
STAR = ('_flms_cov_star', '_dc_cov_star', '_dc_cov_diagonal')
def fam(r):
    t = SRCL[r['line'] - 1] if 0 < r['line'] <= len(SRCL) else ''
    if r['entry'] in UVP or r['site'] in ('_uvp_recon', '_uvp_inner', '_ubx_recon7', '_astra_bcc_uvrecon'): return 'uvp'
    if r['site'] == '_astra_bcc_push_impl': return 'basis_push' if 'shared_basis' in t else 'join'
    if r['site'] in OLD: return 'oldbank'
    if r['site'] in STAR: return 'starcov'
    return None
def mkn(a, b):
    if len(a) == 2 and len(b) == 2:
        if a[1] == b[0]: return (a[0], a[1], b[1])
        if a[1] == b[1]: return (a[0], a[1], b[0])
    return None
want = collections.defaultdict(list)      # (family, shape) -> [(net, live flops, entry)]
for L in logs:
    for r in L['rows']:
        f, s = fam(r), mkn(r['a'], r['b'])
        if f in ('join', 'oldbank', 'uvp', 'starcov') and s and min(s) > 0:
            b = price.get(str(s))
            if b is not None and r['flops'] - b['warm']['flops'] > 0: want[(f, s)].append((L['seed'], r['flops'], r['entry']))
res = {}
for (f, s), uses in sorted(want.items()):
    b = price[str(s)]; entry = collections.Counter(u[2] for u in uses).most_common(1)[0][0]
    inc_fn = M._shared_old384_mm if entry == '_shared_old384_mm' else M._mm
    a_, b_ = arrays(*s)
    inc = [measure(lambda: inc_fn(a_, b_)) for _ in range(4)][1:]
    sm = None
    try:
        name, L, split = b['name'], b['L'], b['packets']
        m, k, n = s; ori, u = tuple(EXEC[name]['shape']), 1 << L
        parts = packets(n) if split else [n]
        need = [(math.ceil(m/ori[0]/u), math.ceil(k/ori[1]/u), math.ceil(nn/ori[2]/u)) for nn in parts]
        big = bind(('push', L, 40, max(x[0] for x in need), max(x[1] for x in need), max(x[2] for x in need)))
        q = kernels[name]; ops = [arrays(m, k, nn) for nn in parts]
        M._SCRATCH[0] = big
        def run():
            for x, y in ops: q(x, y, L, 40)
        try: sm = [measure(run) for _ in range(4)][1:]
        finally: M._SCRATCH[0] = own
    except Exception as e:
        print('smirnov failed', f, s, repr(e)[:120], flush=True)
    if sm is None: continue
    med = lambda xs, k: statistics.median(x[k] for x in xs)
    res[(f, s)] = dict(inc_flops=med(inc, 'flops'), sm_flops=med(sm, 'flops'), d_instr=med(sm, 'instructions') - med(inc, 'instructions'),
                       d_cycles=med(sm, 'cycles') - med(inc, 'cycles'), inc_instr=med(inc, 'instructions'), uses=uses)
    r = res[(f, s)]; print(f, s, 'dFLOP', r['sm_flops'] - r['inc_flops'], 'dInstr', r['d_instr'], 'dCyc', r['d_cycles'], 'inc_instr', r['inc_instr'], flush=True)
tot = {L['seed']: L['total'] for L in logs}; out = {}
for f in ('join', 'oldbank', 'uvp', 'starcov'):
    per = []
    for net in tot:
        sav = sum(u[1] - price[str(s)]['warm']['flops'] for (ff, s), r in res.items() if ff == f for u in r['uses'] if u[0] == net)
        di = sum(r['d_instr'] for (ff, s), r in res.items() if ff == f for u in r['uses'] if u[0] == net)
        dc = sum(r['d_cycles'] for (ff, s), r in res.items() if ff == f for u in r['uses'] if u[0] == net)
        per.append((100 * sav / tot[net], di, dc))
    if per:
        out[f] = dict(saving_pct=round(statistics.mean(x[0] for x in per), 4), d_instr_per_net=round(statistics.mean(x[1] for x in per)),
                      d_cycles_per_net=round(statistics.mean(x[2] for x in per)))
(D/'WALLPRICE_small_v1.json').write_text(json.dumps(dict(families=out, shapes={f'{k[0]}:{k[1]}': {kk: vv for kk, vv in v.items() if kk != 'uses'} for k, v in res.items()}), indent=1))
print(json.dumps(out, indent=1))
