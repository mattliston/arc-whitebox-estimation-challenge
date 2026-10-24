"""TAG opus_engineer a61 (2026-10-11, lead GO on bet A step 1): energy census of the young bank's pairs, in process, at every young-square
read (_cp_read_young_sq) of the given networks (the fired S47e file by default).

A young pair (+ and - column of one hub's birth, weights +a / -a) adds T = a (x+ y+' - x- y-') to the read's plane, x = the read's left
column (response rows of pp^2, the pair law folded in, exactly as the read uses it), y = its live-row column. Its energy E = ||T||_F
(exact, from column norms and the two inner products). Per layer: the share of pairs (and, for reference, of single columns, e = |lam|
||x|| ||y||) that holds 99% / 99.9% of the sum of E, the top-10% share, the Gini of E, and the ACTUAL plane error of dropping the
bottom 10 / 25 / 50% of pairs (||sum of their T||_F over the young plane and over the whole plane the read returns). Per cohort: the pairs
that are bottom-10 / 25% at the cohort's first read, and the share of the cohort's energy they hold at each later read (whether a ranking
at the first read stays right). Nothing is changed in the prediction: the hook only reads the read's operands.
usage: census_v1.py <candidate.py> <comma ranks> <out.json>"""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, ranks, out = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_census'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)

DROPS = (0.10, 0.25, 0.50)
rows_out = []
current = dict(rank=None, layers=[], first=None)


def gini(v):
    v = np.sort(np.asarray(v, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def share_needed(v, q):
    s = np.sort(np.asarray(v, dtype=np.float64))[::-1]
    c = np.cumsum(s)
    return float((np.searchsorted(c, q * c[-1]) + 1) / len(s)) if len(s) else 0.0


parent = mod._cp_read_young_sq


def hooked(left, young_right, lam, live_indices, state):
    result = parent(left, young_right, lam, live_indices, state)
    layer = int(mod._PHYS_LAYER[0]) + 1
    old = int(state['columns'])
    L = np.asarray(left, dtype=np.float64)[:, old:]
    R = np.asarray(young_right, dtype=np.float64)
    lam_y = np.asarray(lam, dtype=np.float64)[old:]
    plane_total = np.asarray(result, dtype=np.float64)
    ny = L.shape[1]
    assert R.shape[1] == ny == lam_y.shape[0], (L.shape, R.shape, lam_y.shape)
    X = L * lam_y[None, :]                                  # x_c = lam_c (left column)
    nx, nr = np.linalg.norm(X, axis=0), np.linalg.norm(R, axis=0)
    e_col = nx * nr
    pairs, owner = [], []                                   # (plus column, minus column) relative to the young block; birth layer
    for b, (start, width) in enumerate(state['cohorts']):
        if start < old:
            continue
        h = width // 2
        for i in range(h):
            pairs.append((start - old + i, start - old + h + i))
            owner.append(b)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    owner = np.asarray(owner, dtype=np.int64)
    p, m = pairs[:, 0], pairs[:, 1]
    # ||x+ y+' + x- y-'||^2 with x- = lam- (left-) already signed: |x+|^2|y+|^2 + |x-|^2|y-|^2 + 2 (x+.x-)(y+.y-)
    xx = np.einsum('rp,rp->p', X[:, p], X[:, m])
    yy = np.einsum('rp,rp->p', R[:, p], R[:, m])
    E = np.sqrt(np.maximum(nx[p] ** 2 * nr[p] ** 2 + nx[m] ** 2 * nr[m] ** 2 + 2 * xx * yy, 0.0))
    young_plane = X @ R.T
    ny_norm, nt_norm = float(np.linalg.norm(young_plane)), float(np.linalg.norm(plane_total))
    order = np.argsort(E)                                   # ascending: the drop candidates first
    drop = {}
    for f in DROPS:
        k = int(round(f * len(E)))
        sel = order[:k]
        cols = np.concatenate([p[sel], m[sel]])
        T = X[:, cols] @ R[:, cols].T
        nT = float(np.linalg.norm(T))
        drop['%d' % round(100 * f)] = dict(pairs=k, energy_share=float(E[sel].sum() / E.sum()), err_young=nT / ny_norm, err_total=nT / nt_norm)
    # persistence: ranking at a cohort's first read vs its energy now
    first = current['first']
    persist = {}
    for b in np.unique(owner):
        idx = np.nonzero(owner == b)[0]
        Eb = E[idx]
        if b not in first:
            ob = np.argsort(Eb)
            first[int(b)] = dict(layer=layer, bottom={str(round(100 * f)): ob[:int(round(f * len(Eb)))].tolist() for f in DROPS[:2]})
        else:
            persist[str(int(b))] = {f: float(Eb[np.asarray(sel, dtype=np.int64)].sum() / max(Eb.sum(), 1e-300))
                                    for f, sel in first[int(b)]['bottom'].items()}
    cohort_share = {str(int(b)): float(E[owner == b].sum() / E.sum()) for b in np.unique(owner)}
    current['layers'].append(dict(layer=layer, old_columns=old, young_columns=int(ny), pairs=int(len(E)),
                                  pair_need99=share_needed(E, 0.99), pair_need999=share_needed(E, 0.999),
                                  pair_top10=float(np.sort(E)[::-1][:max(1, len(E) // 10)].sum() / E.sum()), pair_gini=gini(E),
                                  col_need99=share_needed(e_col, 0.99), col_need999=share_needed(e_col, 0.999), col_gini=gini(e_col),
                                  young_over_total=ny_norm / nt_norm, drop=drop, cohort_share=cohort_share, persist=persist))
    return result


mod._cp_read_young_sq = hooked
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
for rank in ranks:
    current.update(rank=rank, layers=[], first={})
    weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    rows_out.append(dict(rank=rank, layers=list(current['layers'])))
    for r in current['layers']:
        d = r['drop']
        print(f"rank {rank} L{r['layer']:2d} pairs {r['pairs']:5d} need99 {r['pair_need99']:.3f} need999 {r['pair_need999']:.3f} "
              f"top10 {r['pair_top10']:.3f} gini {r['pair_gini']:.3f} | cols need999 {r['col_need999']:.3f} | young/total {r['young_over_total']:.3f} | "
              + ' '.join(f"drop{k}: E {v['energy_share']:.4f} err_y {v['err_young']:.4f} err_t {v['err_total']:.4f}" for k, v in d.items()), flush=True)
out.write_text(json.dumps(rows_out))
print('written', out)
