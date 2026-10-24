"""[opus-smtable] v2 = v1 + every shape's best plan by WARM cost as well (best_warm, with its first-call numbers), and n_choices.
[opus-smtable] v1 = the lead's price_misses_v1.py with the incumbent = GOV_v3 0640c0cf (its read path = S2tr's; governor inert
without a platform BudgetContext), a grid shard as input (JSON list; argv[3]/argv[4] = shard index/count), output under opus_smtable_v1.
[opus-smcov] lead: cost-only exact virtual pricing (no numerics, no MSE) of the Smirnov 3x6x3 read plans for young-read shapes
that miss S2tr's table167. Same harness as Astra's astra_smirnov_unseen_shapes_v1/price_unseen_v1.py, but the incumbent read is
S2tr's own (46e9315d). Input: shapes from smcov.jsonl (or a JSON list via argv[1]); output PRICE_<tag>.json.
usage: nice -n 10 python price_misses_v1.py <shapes.json|smcov> <tag>"""
import os, sys, json, time, functools, collections, importlib.util, hashlib
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS, 0) == 10
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O / 'lead/runs/opus_smtable_v1'
s = importlib.util.spec_from_file_location('virtual_tariff', O / 'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p = importlib.util.module_from_spec(s); s.loader.exec_module(p)
p.SOURCE = O / 'lead/runs/opus_governor_v1/candidate_GOV_v3.py'
p.SOURCE_SHA = '0640c0cfa1b8de327f42ebf0503bc13d4db5a37be099e38550b06e0f17d282ed'
M, _ = p.load(); fnp = p.fnp; fs = p.flops
with fs.BudgetContext(flop_budget=2**60, quiet=True):
    arena = fnp.empty((M._PACKED_ELEMENTS,), dtype=fnp.float32)
    M._SCRATCH = [arena, M._cp3_bind_arena_views(arena, M._PACKED_PUSH), M._cp3_bind_arena_views(arena, M._PACKED_READ), None, None, None, None]
    M._bind_push6(arena)
M._YR7_ACTIVE[0] = True; M._WINOGRAD = True; M._AB_ON = True
M._YR7_STATS = dict(calls=0, chunks=0, shapes=[], routed_shapes=[], old_fused=0, max_arena_elements=0)
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
    return dict(flops=int(ctx.flops_used), requests=sum(p.CALLS.values()) + sum(views.values()), charged=len(ctx._op_log))
def arrays(m, k, n):
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        return fnp.empty((m, k), dtype=fnp.float32), fnp.empty((k, n), dtype=fnp.float32), fnp.empty((m, n), dtype=fnp.float32)
@functools.lru_cache(None)
def incumbent(m, k, n, rank):
    old = int(rank > 0)
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        a = fnp.empty((m, old + k), dtype=fnp.float32); b = fnp.empty((n, k), dtype=fnp.float32); lam = fnp.empty((old + k,), dtype=fnp.float32)
        idx = fnp.asarray(p.np.arange(n, dtype='int64')); inner = fnp.empty((m, rank), dtype=fnp.float32); right = fnp.empty((n, rank), dtype=fnp.float32)
        state = dict(columns=old, pre_basis=fnp.empty((1024, rank), dtype=fnp.float32))
    M._cp_slr_inner_sq = lambda *args: inner; M._xsg_rows = lambda *args: right
    measure(lambda: M._yr7_read(a, b, lam, idx, state))
    return measure(lambda: M._yr7_read(a, b, lam, idx, state))
sys.path.insert(0, str(O / 'lead/runs/astra_smirnov_cfull_v1')); import smirnov_cfull_v1 as h
binding = measure(lambda: h.bind_universal(M, M._SCRATCH[0]))
records = [r for r in json.loads((O.parent / 'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records'] if r['shape'] == [3, 6, 3]]
kernels = {r['name']: h.Kernel(M, r) for r in records}

def clear():
    for name in ('_sm40_cache', '_sm40_ccrop_cache', '_sm40_io_cache', '_sm40_outer_cache', '_sm40_group_cache', '_sm40_padding_grid', '_sm40_padding_cache'):
        getattr(M, name).clear()
src_arg, tag = sys.argv[1], sys.argv[2]
if src_arg == 'smcov':
    shapes, per_net = [], []
    for line in open(D / 'smcov.jsonl'):
        r = json.loads(line); keys = [tuple(x['key']) for x in r['reads'] if not x['hit']]
        per_net.append(keys); shapes += keys
    shapes = sorted(set(shapes))
else:
    shapes = sorted(set(tuple(x) for x in json.load(open(src_arg)))); per_net = None
    if len(sys.argv) > 4:
        shapes = shapes[int(sys.argv[3])::int(sys.argv[4])]
rows, start = [], time.monotonic()
for i, key in enumerate(shapes):
    m, k, n, rank = key; base = incumbent(*key); a, b, _ = arrays(m, k, n)
    with fs.BudgetContext(flop_budget=2**60, quiet=True):
        lam = fnp.empty((k,), dtype=fnp.float32)
        old = (fnp.empty((m, rank), dtype=fnp.float32), fnp.empty((rank, n), dtype=fnp.float32)) if rank else None
    choices = []
    for L in (4, 5, 6):
        u = 1 << L; cm = (m + 3*u - 1)//(3*u)*u; ck = (k + rank + 6*u - 1)//(6*u)*u; cn = (n + 3*u - 1)//(3*u)*u
        try:
            family = h.family_for(L, cm, ck, cn)
        except StopIteration:
            continue
        for name, q in kernels.items():
            clear()
            def fn():
                return q(a*lam, b, L, family[2], old) if m < n else q(a, b*lam[:, None], L, family[2], old)
            first = measure(fn); warm = measure(fn)
            choices.append(dict(name=name, L=L, P=family[2], family=list(family), first=first, warm=warm,
                                first_saving=base['flops']-first['flops'], warm_saving=base['flops']-warm['flops']))
    choices.sort(key=lambda x: x['first']['flops'])
    best = choices[0] if choices and choices[0]['first_saving'] > 0 else None
    best_warm = min(choices, key=lambda x: x['warm']['flops']) if choices else None
    rows.append(dict(shape=list(key), baseline=base, best=best, best_warm=best_warm, n_choices=len(choices)))
    print(json.dumps(dict(at=i, shape=key, base=base['flops'], best=(best['name'], best['L'], best['P'], best['first_saving'], best['warm_saving'], best['first']['requests']-base['requests']) if best else None, s=round(time.monotonic()-start, 1))), flush=True)
out = dict(tag=tag, source_sha256=p.SOURCE_SHA, rows=rows)
if per_net is not None:
    by = {tuple(r['shape']): r for r in rows}
    out['per_net'] = [dict(misses=len(keys), priced_positive=sum(1 for kk in keys if by[kk]['best']), first_saving=sum(by[kk]['best']['first_saving'] for kk in keys if by[kk]['best']), warm_saving=sum(by[kk]['best']['warm_saving'] for kk in keys if by[kk]['best'])) for keys in per_net]
(D / f'PRICE_{tag}.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out.get('per_net')), flush=True)
