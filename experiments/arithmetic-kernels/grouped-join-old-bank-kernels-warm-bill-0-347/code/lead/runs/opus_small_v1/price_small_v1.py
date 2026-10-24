"""[opus-small] (copy of opus-pushsm price_push_v1.py harness) cost-only exact virtual pricing (no numerics) of a Smirnov <3,3,6;40> top level (orientations 3x3x6, 3x6x3, 6x3x3;
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
D = O/'lead/runs/opus_small_v1'
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



# ---- [opus-small] families from the shape logs (live incumbent cost per call), all six certified executables x L4/5/6 x whole/packets ----
import glob
FAM = {'uvp': ('_ubc_product', '_ubx_product', '_uvp_recon', '_uvp_inner', '_ubx_recon7', '_astra_bcc_uvrecon'),
       'readlambda': ('_astra_read_lambda',), 'births_common': ('_astra_bcc_newborn_impl@common',),
       'oldbank': ('_slr_reconstruct', '_cp_slr_inner_sq', '_astra_bcc_recon_impl', '_astra_bcc_recon_full_impl', '_astra_bcc_lazy'),
       'starcov': ('_flms_cov_star', '_dc_cov_star', '_dc_cov_diagonal'),
       'young_read': ('_cp_read_young_sq',)}
SRC = open(p.SOURCE).read().split('\n')
def family(r):
    text = SRC[r['line'] - 1] if 0 < r['line'] <= len(SRC) else ''
    if r['site'] == '_astra_bcc_newborn_impl':
        return 'births_common' if 'common' in text else 'push'
    if r['entry'] in ('_ubc_product', '_ubx_product'): return 'uvp'
    if r['site'] == '_astra_bcc_push_impl':
        return 'basis_push' if 'shared_basis' in text else 'join'
    for f, names in FAM.items():
        if r['site'] in names: return f
    return 'other:%s@%d' % (r['site'], r['line'])
def mkn(a, b):
    if len(a) == 2 and len(b) == 2:
        if a[1] == b[0]: return (a[0], a[1], b[1])
        if a[1] == b[1]: return (a[0], a[1], b[0])
    return None
logs = [json.loads(open(f).read()) for f in sorted(glob.glob(str(D/'shapes/shapes_*.json')))]
calls = []                                         # (net, family, shape, live warm flops)
for L in logs:
    for r in L['rows']:
        f = family(r); s = mkn(r['a'], r['b'])
        if f and f != 'push' and s and min(s) > 0: calls.append((L['seed'], f, s, r['flops'], r['entry']))
big_other = {f for f in {c[1] for c in calls if c[1].startswith('other:')} if sum(c[3] for c in calls if c[1] == f and c[0] == logs[0]['seed']) >= 0.015 * logs[0]['total']}
calls = [c for c in calls if not c[1].startswith('other:') or c[1] in big_other]
FAM.update({f: () for f in sorted(big_other)}); FAM.update({'join': (), 'basis_push': ()})
shapes = sorted({c[2] for c in calls})
print('families', {f: sum(1 for c in calls if c[1] == f) for f in FAM}, 'unique shapes', len(shapes), flush=True)
inc_req = {}
for (m, k, n) in shapes:
    a, b = arrays(m, k, n); measure(lambda: M._mm(a, b)); inc_req[(m, k, n)] = measure(lambda: M._mm(a, b))
best = {}; t0 = time.monotonic()
for shape in shapes:
    bb = None
    for name, rec in EXEC.items():
        for L in (4, 5, 6):
            for split in (False, True):
                if split and shape[2] <= 4096: continue
                try: r = smirnov(shape, name, L, split)
                except Exception as e: continue
                c = dict(ori='x'.join(map(str, rec['shape'])), name=name, L=L, packets=split, warm=r['warm'])
                if bb is None or c['warm']['flops'] < bb['warm']['flops']: bb = c
    best[shape] = bb
    print(json.dumps(dict(shape=shape, mm_warm=inc_req[shape], best=None if bb is None else (bb['ori'], bb['L'], bb['packets'], bb['warm'])), separators=(',', ':')), flush=True)
nets = {}
for net, f, shape, live, entry in calls:
    b = best.get(shape)
    if b is None: continue
    sav = live - b['warm']['flops']
    x = nets.setdefault(net, {}).setdefault(f, dict(live=0, saving=0, wins=0, calls=0, dreq=0))
    x['live'] += live; x['calls'] += 1
    if sav > 0:
        x['saving'] += sav; x['wins'] += 1; x['dreq'] += b['warm']['requests'] - inc_req[shape]['requests']
tot = {L['seed']: L['total'] for L in logs}
summary = {}
for f in FAM:
    per = [100 * nets[n][f]['saving'] / tot[n] for n in nets if f in nets[n]]
    if per: summary[f] = dict(saving_pct_warm_bill_mean=round(sum(per) / len(per), 4), per_net=[round(x, 4) for x in per],
                              live_share_pct=round(sum(100 * nets[n][f]['live'] / tot[n] for n in nets if f in nets[n]) / len(per), 3),
                              wins_calls=[(nets[n][f]['wins'], nets[n][f]['calls']) for n in nets if f in nets[n]],
                              dreq_mean=round(sum(nets[n][f]['dreq'] for n in nets if f in nets[n]) / len(per), 1))
(D/'PRICE_small_v1.json').write_text(json.dumps(dict(summary=summary, best={str(k): v for k, v in best.items()}), indent=1))
print(json.dumps(summary, indent=1))
