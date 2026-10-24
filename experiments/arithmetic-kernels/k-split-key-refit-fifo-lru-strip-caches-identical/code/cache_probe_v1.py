"""[opus-tables] read-only cache-pressure probe: one worker session (setup once, then networks in order) on a candidate.
After each network: size of every capped cache (and its cap), puts/evictions through _cache_put/_bh_put, FLOPs and charged-op counts.
Wrappers only count (plain Python); no flopscope op is added, so predictions/FLOPs are those of the candidate.
usage (OpenBLAS python, BLAS 1): cache_probe_v1.py <candidate.py> <comma ranks> <out.json>"""
import collections, hashlib, json, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
cand, ranks, out = Path(sys.argv[1]).resolve(), [int(r) for r in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_cache'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
g = mod.__dict__
CAPS = {n: g['_CAP_BY_ID'].get(id(g[n])) for n in list(g['_CAP_LIMITS']) + ['_READ6_FRAMES', '_IPT_A', '_IPT_B', '_IPT_C', '_IPT_BA', '_IPT_BB', '_IPT_BC', '_IPT_R6', '_AB_VIEWS'] if n in g}
INLINE = dict(_HZ_SIDES='_HZ_SIDE_CAP', _HZ_VIEWS='_HZ_VIEW_CAP', _HZG_VIEWS='_HZG_VIEW_CAP', _RQD_CACHE='_RQD_CAP', _HZ7_VIEWS='_HZ7_VIEW_CAP', _HZD_VIEWS='_HZD_VIEW_CAP', _OT_QV='_OT_QV_CAP')
for n, c in INLINE.items():
    if n in g and c in g: CAPS[n] = g[c]
name_of = {id(g[n]): n for n in CAPS if n in g}
ev, puts = collections.Counter(), collections.Counter()
def wrap(fname):
    orig = g[fname]
    def put(cache, key, value, _o=orig):
        n = name_of.get(id(cache), 'other:' + str(id(cache)))
        lim = g['_CAP_BY_ID'].get(id(cache), 8192)
        puts[n] += 1
        if len(cache) >= lim: ev[n] += len(cache) - lim + 1
        return _o(cache, key, value)
    g[fname] = put
wrap('_cache_put'); wrap('_bh_put')
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
sizes0 = {n: len(g[n]) for n in CAPS if n in g and hasattr(g[n], '__len__')}
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
rows = []
for rank in ranks:
    w = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    p0, e0 = dict(puts), dict(ev)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        fl = int(ctx.summary_dict()['flops_used']); ops = len(ctx._op_log)
    sizes = {n: len(g[n]) for n in CAPS if n in g and hasattr(g[n], '__len__')}
    rows.append(dict(rank=rank, flops=fl, ops=ops, pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16], sizes=sizes,
                     puts={k: puts[k] - p0.get(k, 0) for k in puts if puts[k] - p0.get(k, 0)}, evictions={k: ev[k] - e0.get(k, 0) for k in ev if ev[k] - e0.get(k, 0)}))
    print(json.dumps(dict(rank=rank, flops=fl, ops=ops, evictions=rows[-1]['evictions'])), flush=True)
with out.open('x') as f: json.dump(dict(candidate=str(cand), sha256=hashlib.sha256(cand.read_bytes()).hexdigest(), caps=CAPS, sizes_after_setup=sizes0, rows=rows), f, indent=1)
print('PROBE_DONE', flush=True)
