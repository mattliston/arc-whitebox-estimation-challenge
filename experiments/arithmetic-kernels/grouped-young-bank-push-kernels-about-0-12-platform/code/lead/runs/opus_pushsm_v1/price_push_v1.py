"""[opus-pushsm] cost-only exact virtual pricing (no numerics) of a Smirnov <3,3,6;40> top level (orientations 3x3x6, 3x6x3, 6x3x3;
inner Winograd depth L = 4, 5, 6; padding counted) against the incumbent young-bank push _mm (S2tr 46e9315d), on the push products
logged from two networks (pushlog_v1.jsonl: every _mm call inside _astra_bcc_newborn_impl, by layer). Same harness as
opus_smcov_v1/price_misses_v1.py (virtual tariff: flopscope's own charging, metadata-only backend). No setup constraint: the push
families (P = 40, one batch) are bound on a separate, oversized virtual arena; the incumbent runs on the module's own arena.
Smirnov is priced on the whole product (A-side once) and on 4096-column packets (A-side per packet).
usage: nice -n 10 <openblas python> price_push_v1.py <tag>"""
import collections, functools, json, math, os, sys, time, importlib.util
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
from flopscope.numpy import ndarray
views = collections.Counter(); original_get = ndarray.__getitem__; original_T = ndarray.T
def counted_get(self, key):
    views['getitem'] += 1; return original_get(self, key)
ndarray.__getitem__ = counted_get
def counted_T(self):
    views['T'] += 1; return original_T.__get__(self, type(self))
ndarray.T = property(counted_T)
def reset():
    M._BH_HOLD.clear(); M._TR_LAST[0] = None; views.clear(); p.CALLS.clear()
def measure(fn):
    reset()
    with fs.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
        fn()
    return dict(flops=int(ctx.flops_used), requests=sum(p.CALLS.values()) + sum(views.values()))
def arrays(m, k, n):
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        return fnp.empty((m, k), dtype=fnp.float32), fnp.empty((k, n), dtype=fnp.float32)

calls = [json.loads(x) for x in open(D/'pushlog_v1.jsonl')]
shapes = sorted({tuple(r[1:]) for net in calls for r in net if r[3] > 0})

# incumbent, on the module's own arena (first call = cold view binding, second = warm)
incumbent = {}
for (m, k, n) in shapes:
    a, b = arrays(m, k, n)
    first = measure(lambda: M._mm(a, b)); warm = measure(lambda: M._mm(a, b))
    incumbent[(m, k, n)] = dict(first=first, warm=warm)

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


rows, t0 = [], time.monotonic()
for shape in shapes:
    inc = incumbent[shape]; best = None; choices = []
    for name, rec in EXEC.items():
        for L in (4, 5, 6):
            for split in (False, True):
                if split and shape[2] <= 4096:
                    continue
                r = smirnov(shape, name, L, split)
                c = dict(ori='x'.join(map(str, rec['shape'])), name=name, L=L, packets=split, first=r['first'], warm=r['warm'],
                         warm_saving=inc['warm']['flops'] - r['warm']['flops'], d_req_warm=r['warm']['requests'] - inc['warm']['requests'])
                choices.append(c)
                if best is None or c['warm']['flops'] < best['warm']['flops']:
                    best = c
    rows.append(dict(shape=list(shape), incumbent=inc, best=best, choices=choices))
    print(json.dumps(dict(shape=shape, inc_warm=inc['warm'], best=(best['ori'], best['L'], best['packets'], best['warm']['flops'], best['warm_saving'], best['d_req_warm']), s=round(time.monotonic()-t0, 1))), flush=True)
by = {tuple(r['shape']): r for r in rows}
nets = []
for net in calls:
    inc = sum(by[tuple(r[1:])]['incumbent']['warm']['flops'] for r in net if r[3] > 0)
    sav = sum(max(0, by[tuple(r[1:])]['best']['warm_saving']) for r in net if r[3] > 0)
    dreq = sum(by[tuple(r[1:])]['best']['d_req_warm'] for r in net if r[3] > 0 and by[tuple(r[1:])]['best']['warm_saving'] > 0)
    nets.append(dict(push_flops=inc, best_saving=sav, saving_share_of_push=sav/inc, d_requests=dreq))
# setup cost of binding the winning families (P = 10) on the module's own arena: requests (views) and elements
need = {}
for r in rows:
    b = r['best']
    if b['warm_saving'] <= 0:
        continue
    (m, k, n), ori, L = r['shape'], tuple(map(int, b['ori'].split('x'))), b['L']
    u = 1 << L; x = need.setdefault((ori, L), [0, 0, 0])
    x[0] = max(x[0], math.ceil(m/ori[0]/u)); x[1] = max(x[1], math.ceil(k/ori[1]/u)); x[2] = max(x[2], math.ceil(n/ori[2]/u))
binds = []
for (ori, L), (rr, kk, cc) in need.items():
    f = ('push_%d%d%d_L%d' % (ori + (L,)), L, 10, rr, kk, cc)
    el = 7 ** L * 10 * (rr*kk + kk*cc + rr*cc)
    cost = measure(lambda: h.bind_family(fnp, M, own, f)) if el <= int(own.shape[0]) else None
    binds.append(dict(family=f, elements=el, fits_main_arena=el <= int(own.shape[0]), bind=cost))
out = dict(tag=sys.argv[1], source_sha256=p.SOURCE_SHA, rows=rows, nets=nets, binds=binds)
(D/f'PRICE_push_{sys.argv[1]}.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(nets), flush=True)
