"""TAG opus_engineer quick: one in-process prediction per rank of a candidate, compared with stored RRPI rows/predictions.
usage: quick.py <candidate.py> <comma ranks> [<out.json>] [--check] [--views] [--hzkeys]
Prints per network: bill and its change against RRPI's bill, final-layer MSE and its change, the largest absolute difference of the
16 x 1024 prediction against RRPI's stored prediction, charged-op count, and the candidate's own counters (strip / row-cut / cache).
--views counts every index call on a flopscope array (views and gathers: each is a request at the boundary) by the candidate
function that makes it.
--hzkeys records every key (M, q, N, row strip, column strip) the strip kernels ask leaf views for, in order, and whether its
views had to be built (row['hz_keys']: [M, q, N, hzr, hzc, built]).
--check wraps the strip kernels and compares every strip product with float64 numpy outside the budget (slow; relative error
printed as max over calls, with the count of calls that ran under the B-tree cache).
RRPI reference rows: H/opus_engineer_rrpi_screen_a (35,100,36,45) and H/opus_engineer_rrpi_check_c (52..88)."""
from pathlib import Path
import json
import sys
import time
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy
P2 = Path(__file__).resolve().parents[2]
H = P2/'codex_resume_20260926_v2'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
check = '--check' in sys.argv
candidate = Path(args[0]).resolve(); ranks = [int(r) for r in args[1].split(',')]
out = Path(args[2]) if len(args) > 2 else None
assert out is None or not out.exists()
ref = {}
for tag in ('opus_engineer_rrpi_screen_a', 'opus_engineer_rrpi_check_c'):
    for line in open(H/tag/'rows.jsonl'):
        r = json.loads(line); ref[r['rank']] = (r, H/tag/f"pred_{r['rank']}.npy")
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_quick'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
binds = []      # --binds: every lazily bound tree view of the arena (one reshape billed at its element count each)
seen = set()
if '--binds' in sys.argv:
    _view = mod._ipt_view

    def view_probe(arena, start, shape):
        key = (int(start), tuple(int(x) for x in shape))
        count = 1
        for x in key[1]:
            count *= x
        binds.append((count, key in seen, key))
        seen.add(key)
        return _view(arena, start, shape)
    mod._ipt_view = view_probe
used = []      # --keys: every tree key (kind, depth, rows, cols) a prediction asks for, bound or not (for prebind lists)
if '--keys' in sys.argv:
    _side = mod._ipt_side

    def side_probe(kind, L, rows, cols):
        if getattr(mod, '_R7_A', None) is None or mod._IPT_A is not mod._R7_A:   # not the depth-7 read's private region and caches
            used.append((int(kind), int(L), int(rows), int(cols)))
        return _side(kind, L, rows, cols)
    mod._ipt_side = side_probe
    if hasattr(mod, '_ks_acc'):   # the k-split's second leaf buffer, recorded as kind 3 at depth 6
        _acc = mod._ks_acc

        def acc_probe(M, N):
            used.append((3, 6, int(M), int(N)))
            return _acc(M, N)
        mod._ks_acc = acc_probe
hz_keys = []    # --hzkeys: every call of the strip kernels' leaf-view cache, in order
if '--hzkeys' in sys.argv:
    _hzv = mod._hz_views

    def hz_probe(M, q, N, hzr, hzc):
        hz_keys.append([int(M), int(q), int(N), bool(hzr), bool(hzc), (M, q, N, hzr, hzc) not in mod._HZ_VIEWS])
        return _hzv(M, q, N, hzr, hzc)
    mod._hz_views = hz_probe
views = {}      # --views: index calls per function of the candidate
if '--views' in sys.argv:
    from flopscope import _ndarray as _nd
    _getitem = _nd.FlopscopeArray.__getitem__

    def _counted_getitem(self, key):
        name = sys._getframe(1).f_code.co_name
        views[name] = views.get(name, 0) + 1
        return _getitem(self, key)
    _nd.FlopscopeArray.__getitem__ = _counted_getitem
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
checks = []
if check:
    def wrap(name, shape_of):
        original = getattr(mod, name)

        def probe(a, b, *rest):
            A = np.asarray(a, dtype=np.float64); B = np.asarray(b, dtype=np.float64)
            under = mod._UBC_CONTEXT[0] is not None
            result = original(a, b, *rest)
            want = A @ B
            checks.append((name, list(A.shape) + [int(B.shape[1])], under, float(np.linalg.norm(np.asarray(result, dtype=np.float64) - want)/max(np.linalg.norm(want), 1e-300))))
            return result
        setattr(mod, name, probe)
    for name in ('_hzg_product', '_hz_ks_product'):
        if hasattr(mod, name):
            wrap(name, None)
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
rows = []
for rank in ranks:
    weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    checks.clear(); binds.clear(); used.clear(); views.clear(); hz_keys.clear()
    t0 = time.process_time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        bill = int(ctx.summary_dict()['flops_used']); ops = len(ctx._op_log)
    cpu = time.process_time() - t0
    truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    mse = float(np.mean((y[15]-truth)**2))
    if out is not None:   # the 16 x 1024 prediction, float32 as the estimator returns it, for byte comparisons between quick runs
        np.save(out.with_name(out.stem + f'_pred_{rank}.npy'), np.asarray(y, dtype=np.float32))
    import hashlib
    row = dict(rank=rank, flops=bill, mse=mse, ops=ops, cpu_s=round(cpu, 2), pred_sha=hashlib.sha256(np.asarray(y, dtype=np.float32).tobytes()).hexdigest()[:16])
    if rank in ref:
        r0, pf = ref[rank]
        p0 = np.load(pf).astype(np.float64)
        row.update(bill_pct=100*(bill/r0['flops']-1), mse_pct=100*(mse/r0['mse']-1), max_abs_dpred=float(np.max(np.abs(y-p0))),
                   max_abs_dpred_final=float(np.max(np.abs(y[15]-p0[15]))), rms_err_final=float(np.sqrt(r0['mse'])))
    for name in ('_HZ_STATS', '_HZG_STATS', '_RC_STATS', '_DCR_STATS', '_BVP_STATS', '_YNT_STATS', '_UBC_STATS', '_OTT_STATS', '_D7_STATS', '_P6_STATS', '_F4_STATS', '_K4L_STATS', '_F4B_STATS', '_F4C_STATS', '_ECUT_STATS', '_L3_STATS', '_L3O_STATS', '_SX_STATS', '_SS_STATS', '_JR_STATS', '_NB_STATS', '_NP_STATS', '_RQD_STATS', '_RQS_STATS', '_HZ7_STATS', '_HZD_STATS', '_PSDM_STATS', '_DCV_STATS'):
        if hasattr(mod, name):
            v = getattr(mod, name)
            row[name] = {k: v[k] for k in v if k not in ('hit_slots',)} if name != '_UBC_STATS' else {k: v.get(k) for k in ('eligible', 'hits', 'misses', 'hit_slots')}
    if '--views' in sys.argv:
        row['views'] = dict(total=sum(views.values()), by_function=sorted(views.items(), key=lambda kv: -kv[1])[:14])
    if '--hzkeys' in sys.argv:
        row['hz_keys'] = [list(k) for k in hz_keys]
    if '--keys' in sys.argv:
        import collections as _c
        row['key_uses'] = [list(k) + [v] for k, v in sorted(_c.Counter(used).items())]
    if '--binds' in sys.argv:
        row['binds'] = dict(n=len(binds), flops=sum(b[0] for b in binds), pct_of_bill=100*sum(b[0] for b in binds)/bill,
                            rebinds=sum(1 for b in binds if b[1]), rebind_flops=sum(b[0] for b in binds if b[1]), keys_seen=len(seen))
        row['bind_list'] = [[b[2][0], list(b[2][1]), bool(b[1])] for b in binds]
    if check:
        row['checks'] = dict(calls=len(checks), under_cache=sum(1 for c in checks if c[2]), max_rel=max((c[3] for c in checks), default=0.0),
                             max_rel_under_cache=max((c[3] for c in checks if c[2]), default=0.0),
                             worst=sorted(checks, key=lambda c: -c[3])[:3])
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('bind_list', 'key_uses')}), flush=True)
if out is not None:
    out.write_text(json.dumps(dict(candidate=str(candidate), rows=rows)))
