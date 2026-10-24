#!/usr/bin/env python3
"""TAG analyst A17 (lead, 2026-10-02 07:12Z): can the per-network gradient of a dial be predicted from quantities the native card has
for free on any network? If yes a dial of the form a + b x feature is worth more than the global one.
DATA (stored files only, no card run): the trainer's exact per-network gradients
  T  = O/trainer/cert/m_gradtable.jsonl (rows with the attached star basis; S37 parameters on the S37 physics),
  M  = O/trainer/run_M/results/g*.pt (run M, updates 0..; the dials moved by less than 1% over them: one point),
  R2 = O/trainer/run_M/results/a_round2_g_r*.pt (priority requests at candidate 2).
  A network that occurs at two points is kept at the later one only. Gradient unit: d ln(final-layer MSE) / d dial (the stored
  d(scale x MSE) divided by scale x MSE): the pooled error is a sum of per-network errors, and a feature that only tracks the SIZE of a
  network's error would otherwise look like a predictor of its gradient.
TARGETS: per family of the stored physics leaves, the sum over all layers (the global dial) and over layers 8-15; plus k4a, k4b, k4c.
FEATURES (none needs a card run; * = the card has it in its own state, here taken from a 4,096-input Monte Carlo or the organizers'
means, which agree with the card's to 0.1-0.4%):
  *lv{l}   log of the mean per-unit variance of layer l (trace(C)/n), l = 3, 7, 11, 13, 15        (a15_variance_census)
  *lr{l}   log(|mean|^2 / trace(C)) of layer l
  *decay   lv15 - lv7
  *cv{l}   sd / mean of the per-unit means of layer l, l = 7, 11, 15
  *dead{l} share of units whose mean is below 1% of the layer's average
   s1_*    largest singular value of W: mean over layers 8-15, mean over all layers, layers 13, 14, 15; s2_deep the second one
   rowsd_deep, colsd_deep   sd over rows (columns) of the squared row (column) norm of W, mean over layers 8-15
   fro_deep, fro_all        mean log of the squared Frobenius norm of W over layers 8-15 / all layers
  Not in any stored file (would need a forward run): per-layer radial, alive fraction, mean gate, K4L's beta and dk.
METHOD: gradients are z-scored inside each point; model = one intercept per point + b x feature (or two features); leave-one-network-
out by the hat matrix (exact for least squares); R2 out of sample = 1 - PRESS(model) / PRESS(intercepts only). The best feature and
the best pair are chosen on that R2, so the null is the same selection on gradients permuted among the networks of each point
(NPERM permutations): a dial counts only if its R2 beats the 95th percentile of the null.
Gain of the conditional dial against the global step, first order: sum of predicted gradient x gradient over sum of the point mean
(leave-one-out) x gradient; and at equal curvature: sum (2 pred x g - pred^2) over sum (2 mean x g - mean^2).
numpy / scipy, one thread. usage: a17_conditional_dials.py [NPERM=500] -> a17_conditional_dials.md, a17_conditional_dials.json, a17_features.npz"""
import os, sys, json, glob, re, time, itertools
for k in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; TR = O / 'trainer'
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
t0 = time.time()

# ------------------------------------------------------------------ gradients
import torch
torch.set_num_threads(1)
rows = []
for line in open(TR / 'cert/m_gradtable.jsonl'):
    r = json.loads(line)
    if r.get('star_attach'): rows.append(dict(rank=int(r['rank']), point='T', mse=r['mse'], pl={k: np.array(v, float) for k, v in r['pl'].items()}, dial={k: float(v) for k, v in r['dial_grads'].items()}, unit=1e8))
def load_pt(f, point):
    z = torch.load(f, map_location='cpu', weights_only=True)
    return dict(rank=int(z['rank']), point=point, mse=float(z['mse']), pl={k: v.numpy().astype(float) for k, v in z['pl_grads'].items()}, dial={k: float(v) for k, v in z['dial_grads'].items()}, unit=float(z['scale']))
gfiles = sorted(glob.glob(str(TR / 'run_M/results/g*.pt'))); rows += [load_pt(f, 'M') for f in gfiles]
rows += [load_pt(f, 'R2') for f in sorted(glob.glob(str(TR / 'run_M/results/a_round2_g_r*.pt')))]
order = dict(T=0, M=1, R2=2); last = {}
for r in rows:
    if r['rank'] not in last or order[r['point']] >= order[last[r['rank']]['point']]: last[r['rank']] = r
rows = sorted(last.values(), key=lambda r: (order[r['point']], r['rank'])); N = len(rows); ranks = [r['rank'] for r in rows]; point = np.array([order[r['point']] for r in rows])
fams = [f for f in rows[0]['pl'] if all(f in r['pl'] for r in rows)]
targets = {}
for f in fams:
    rel = np.array([r['pl'][f] / (r['unit'] * r['mse']) for r in rows])              # d ln MSE / d dial, per layer
    targets[f'{f}, all layers'] = rel.sum(1); targets[f'{f}, layers 8-15'] = rel[:, 8:].sum(1)
for k in ('k4a', 'k4b', 'k4c'): targets[f'{k} (band dial)'] = np.array([r['dial'][k] / (r['unit'] * r['mse']) for r in rows])
targets = {k: v for k, v in targets.items() if np.std(v) > 0}
print(f'{N} networks: ' + ', '.join(f"{p} {int((point == order[p]).sum())}" for p in order) + f'; {len(fams)} families, {len(targets)} targets; run M files {len(gfiles)}', flush=True)

# ------------------------------------------------------------------ features (cached)
cache = HERE / 'a17_features.npz'; FE = {}
if cache.exists():
    z = np.load(cache, allow_pickle=True); FE = {int(r): dict(zip(z['names'], row)) for r, row in zip(z['ranks'], z['values'])}
todo = [r for r in ranks if r not in FE]
if todo:
    from scipy.sparse.linalg import svds
    STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    cz = np.load(HERE / 'a15_variance_census.npz'); LAY = [int(x) for x in cz['layers']]; cidx = {int(i): j for j, (k, i) in enumerate(zip(cz['kind'], cz['id'])) if k == 'public'}
    x0 = np.random.Generator(np.random.PCG64(20261002)).standard_normal((4096, 1024)).astype(np.float32)        # the census's common inputs
    for r in todo:
        w = np.load(STORE / Path(man[r]['path']).parent / 'weights.npy'); fe = {}
        if r in cidx: v, q = cz['v'][cidx[r]], cz['q'][cidx[r]]
        else:
            y = x0; v = []; q = []
            for l in range(16):
                y = np.maximum(y @ w[l], 0)
                if l in LAY: var = y.var(0, dtype=np.float64); m = y.mean(0, dtype=np.float64); v.append(var.mean()); q.append((m * m).mean() - var.mean() / 4096)
        for c, l in enumerate(LAY): fe[f'lv{l}'] = np.log(v[c]); fe[f'lr{l}'] = np.log(q[c] / v[c])
        fe['decay'] = fe['lv15'] - fe['lv7']
        mt = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{r:04d}.npz')['mc_post'].astype(np.float64)
        for l in (7, 11, 15): fe[f'cv{l}'] = mt[l].std() / mt[l].mean(); fe[f'dead{l}'] = float((mt[l] < 0.01 * mt[l].mean()).mean())
        s = np.array([np.sort(svds(w[l].astype(np.float64), k=3, tol=1e-7, return_singular_vectors=False))[::-1] for l in range(16)])
        fe.update(s1_deep=s[8:, 0].mean(), s1_all=s[:, 0].mean(), s1_L13=s[13, 0], s1_L14=s[14, 0], s1_L15=s[15, 0], s2_deep=s[8:, 1].mean())
        w2 = w.astype(np.float64) ** 2; rn = w2.sum(2); cn = w2.sum(1)
        fe.update(rowsd_deep=rn[8:].std(1).mean(), colsd_deep=cn[8:].std(1).mean(), fro_deep=np.log(rn[8:].sum(1)).mean(), fro_all=np.log(rn.sum(1)).mean())
        FE[r] = fe; print(f'  features of network {r} ({len(FE)} of {len(ranks)}), {time.time() - t0:.0f} s', flush=True)
    names = list(next(iter(FE.values())).keys()); np.savez(cache, ranks=np.array(sorted(FE)), names=np.array(names), values=np.array([[FE[r][k] for k in names] for r in sorted(FE)]))
fnames = list(FE[ranks[0]].keys()); X = np.array([[FE[r][k] for k in fnames] for r in ranks]); X = (X - X.mean(0)) / X.std(0); F = len(fnames)

# ------------------------------------------------------------------ leave-one-out by the hat matrix
D0 = np.stack([(point == p).astype(float) for p in sorted(set(point))], 1)
def loo_operator(Z):
    """matrix M with (M y)_i = leave-one-out residual of y_i under least squares on the columns of Z."""
    H = Z @ np.linalg.pinv(Z); return (np.eye(N) - H) / (1 - np.diag(H))[:, None]
designs = [('', D0)] + [(fnames[i], np.column_stack([D0, X[:, i]])) for i in range(F)] + [(fnames[i] + ' + ' + fnames[j], np.column_stack([D0, X[:, i], X[:, j]])) for i, j in itertools.combinations(range(F), 2)]
OPS = [loo_operator(Z) for _, Z in designs]; single = np.arange(1, F + 1); pair = np.arange(F + 1, len(designs))
def zscore(y):
    out = np.empty_like(y)
    for p in sorted(set(point)): m = point == p; out[m] = (y[m] - y[m].mean(0)) / y[m].std(0)
    return out
tn = list(targets); Y = zscore(np.array([targets[k] for k in tn]).T)                       # (N, targets)
MSE = np.array([r['mse'] for r in rows]) * 1e8
_man = json.load(open(P2 / 'fable_refit_v1/truth688/manifest.json')); OLD = np.array([float(r in set(_man['train160'])) for r in ranks])      # the learned arrays were trained on these
rng = np.random.default_rng(1702)
def permuted(Y):
    """the networks of each point are shuffled, the same shuffle for every dial (keeps the dials' correlations with one another)."""
    P = np.empty_like(Y)
    for p in sorted(set(point)):
        m = np.where(point == p)[0]; P[m] = Y[rng.permutation(m)]
    return P
def r2_all(Y):
    press = np.array([((M @ Y) ** 2).sum(0) for M in OPS]); return 1 - press[1:] / press[0]            # (designs - 1, targets)
R2 = r2_all(Y); best1 = R2[:F].argmax(0); best2 = R2[F:].argmax(0)
null1 = np.empty((NPERM, len(tn))); null2 = np.empty((NPERM, len(tn)))
for b in range(NPERM):
    Rp = r2_all(permuted(Y)); null1[b] = Rp[:F].max(0); null2[b] = Rp[F:].max(0)
print(f'analysis done, {time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------------ report
res = []
for c, name in enumerate(tn):
    y = targets[name]; yz = Y[:, c]
    # consistency of the raw relative gradient: t of the mean inside each point, pooled as the mean of the z-like ratios
    tpts = {p: float(y[point == order[p]].mean() / (y[point == order[p]].std(ddof=1) / np.sqrt((point == order[p]).sum()))) for p in order if (point == order[p]).sum() >= 4}
    same = int(max((y > 0).sum(), (y < 0).sum()))
    r1, r2 = float(R2[best1[c], c]), float(R2[F + best2[c], c]); q1, q2 = float(np.percentile(null1[:, c], 95)), float(np.percentile(null2[:, c], 95))
    p1, p2 = float((null1[:, c] >= r1).mean()), float((null2[:, c] >= r2).mean())
    # gains with the leave-one-out predictions of the best single feature, in the gradient's own unit
    base = np.empty(N); pred = np.empty(N)
    for p in sorted(set(point)):
        m = point == p; mu, sd = y[m].mean(), y[m].std()
        base[m] = mu + sd * (yz[m] - (OPS[0] @ yz)[m]); pred[m] = mu + sd * (yz[m] - (OPS[1 + best1[c]] @ yz)[m])
    g1 = float((pred * y).sum() / (base * y).sum()) if (base * y).sum() > 0 else float('nan')
    gc = float((2 * pred * y - pred ** 2).sum() / (2 * base * y - base ** 2).sum()) if (2 * base * y - base ** 2).sum() > 0 else float('nan')
    cpt = {p: float(np.corrcoef(X[point == order[p], best1[c]], y[point == order[p]])[0, 1]) for p in order if (point == order[p]).sum() >= 4}
    mM = point == order['M']; scale_M = (float(y[mM].mean()), float(y[mM].std(ddof=1)))
    yraw = zscore((y * MSE)[:, None])[:, 0]; pr = np.array([((M @ yraw) ** 2).sum() for M in (OPS[0], OPS[1 + best1[c]])]); r_raw = float(1 - pr[1] / pr[0])
    Zo = np.column_stack([D0, OLD]); Mo0 = loo_operator(Zo); Mo1 = loo_operator(np.column_stack([Zo, X[:, best1[c]]])); r_old = float(1 - ((Mo1 @ yz) ** 2).sum() / ((Mo0 @ yz) ** 2).sum())
    res.append(dict(dial=name, t=tpts, same_sign=same, corr_by_point=cpt, mean_sd_at_M=scale_M, r2_raw_gradient=r_raw, r2_with_old_rank_term=r_old, best_feature=fnames[best1[c]], r2_single=r1, null95_single=q1, p_single=p1, best_pair=designs[1 + F + best2[c]][0], r2_pair=r2, null95_pair=q2, p_pair=p2,
                    passes=bool(r1 > q1), gain_first_order=g1, gain_equal_curvature=gc))
npass = sum(r['passes'] for r in res)
# family-wise null: how many dials pass in a permutation (same threshold rule applied to each permutation against the other permutations' 95th percentile)
thr = np.percentile(null1, 95, axis=0); fw = (null1 > thr).sum(1)
# family-wise p of the best dial: in each permutation, the smallest per-dial p (rank of its R2 within that dial's null)
rank_p = 1 - (np.argsort(np.argsort(null1, axis=0), axis=0) + 0.5) / NPERM; minp_null = rank_p.min(1); minp_obs = min(r['p_single'] for r in res)
fw_p_best = float((minp_null <= max(minp_obs, 0.5 / NPERM)).mean()); fw_p_count = float((fw >= npass).mean())
out = dict(TAG='analyst A17 conditional dials', networks=N, points={p: int((point == order[p]).sum()) for p in order}, ranks=ranks, features=fnames, nperm=NPERM, results=res,
           dials_passing=npass, dials=len(res), familywise_null_mean=float(fw.mean()), familywise_null_95=float(np.percentile(fw, 95)), familywise_p_of_the_count=fw_p_count, familywise_p_of_the_best_dial=fw_p_best,
           old_training_ranks=int(OLD.sum()), run_M_gradient_files=len(gfiles))
json.dump(out, open(HERE / 'a17_conditional_dials.json', 'w'), indent=1)
L = []; W = L.append
W(f"# A17. Can a dial's per-network gradient be predicted from what the card knows for free? ({N} networks, {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC)\n")
W(f"Networks: {out['points']['T']} at the S37 parameters (the trainer's gradient table), {out['points']['M']} at run M's point ({len(gfiles)} gradient files), {out['points']['R2']} at candidate 2. Gradient = d ln(final-layer MSE) / d dial, z-scored inside each point. "
  f"{F} features, {len(designs) - 1 - F} pairs, {NPERM} permutations. {npass} of {len(res)} dials beat the 95th percentile of their null with one feature; permuted data give {fw.mean():.1f} on average (95th percentile {np.percentile(fw, 95):.0f}; "
  f"a count of {npass} or more in {fw_p_count * 100:.0f}% of permutations). The best dial's p of {minp_obs:.3f} is matched by some dial in {fw_p_best * 100:.0f}% of permutations. {int(OLD.sum())} of the {N} networks are old training ranks.\n")
if (HERE / 'a17_reading.md').exists(): W((HERE / 'a17_reading.md').read_text())
W('| dial | t of the mean gradient at T / M / R2 | networks with the majority sign | best feature | its correlation with the gradient at T / M / R2 | R2 out of sample | null 95th | p | best pair | R2 out of sample | null 95th | p | counts? | gain against the global step, first order | at equal curvature | mean and sd across networks of the gradient at the point of run M (d ln MSE per unit of dial) | R2 with the raw gradient d MSE | R2 with an old-training-rank term in both models |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(res, key=lambda r: r['p_single']):
    tt = ' / '.join(f"{r['t'].get(p, float('nan')):+.1f}" for p in ('T', 'M', 'R2'))
    cc = ' / '.join(f"{r['corr_by_point'].get(p, float('nan')):+.2f}" for p in ('T', 'M', 'R2'))
    W(f"| {r['dial']} | {tt} | {r['same_sign']} of {N} | {r['best_feature']} | {cc} | {r['r2_single']:+.3f} | {r['null95_single']:+.3f} | {r['p_single']:.3f} | {r['best_pair']} | {r['r2_pair']:+.3f} | {r['null95_pair']:+.3f} | {r['p_pair']:.3f} | {'YES' if r['passes'] else 'no'} | "
      f"{r['gain_first_order']:.2f} | {r['gain_equal_curvature']:.2f} | {r['mean_sd_at_M'][0]:+.3f}, {r['mean_sd_at_M'][1]:.3f} | {r['r2_raw_gradient']:+.3f} | {r['r2_with_old_rank_term']:+.3f} |")
# ---- the common factors: principal components of the per-network gradients of the global dials (fewer, independent tests)
gl = [c for c, k in enumerate(tn) if k.endswith(', all layers')]; U, S, Vt = np.linalg.svd(Y[:, gl], full_matrices=False); K = 4
PC = U[:, :K] * S[:K]; share = (S ** 2 / (S ** 2).sum())[:K]
Rpc = r2_all(PC); npc = np.empty((NPERM, K))
for b in range(NPERM): npc[b] = r2_all(permuted(PC))[:F].max(0)
W('\n## The common factors of the dial gradients\n')
W(f"The {len(gl)} global-dial gradients move together across networks: principal components of their z-scores, tested like single dials (one feature, same null).\n")
W('| component | share of the variance | dials with the largest loadings | best feature | correlation at T / M / R2 | R2 out of sample | null 95th | p |\n|---|---|---|---|---|---|---|---|')
pcs = []
for k in range(K):
    j = int(Rpc[:F, k].argmax()); r = float(Rpc[j, k]); pv = float((npc[:, k] >= r).mean()); top = np.argsort(-np.abs(Vt[k]))[:5]
    cc = ' / '.join(f"{np.corrcoef(X[point == q, j], PC[point == q, k])[0, 1]:+.2f}" for q in sorted(set(point)))
    W(f"| {k + 1} | {share[k] * 100:.0f}% | " + ', '.join(f"{tn[gl[t]].split(',')[0]} {Vt[k, t]:+.2f}" for t in top) + f" | {fnames[j]} | {cc} | {r:+.3f} | {np.percentile(npc[:, k], 95):+.3f} | {pv:.3f} |")
    pcs.append(dict(component=k + 1, share=float(share[k]), best_feature=fnames[j], r2=r, null95=float(np.percentile(npc[:, k], 95)), p=pv, loadings={tn[gl[t]].split(',')[0]: float(Vt[k, t]) for t in top}))
out['components'] = pcs; json.dump(out, open(HERE / 'a17_conditional_dials.json', 'w'), indent=1)
(HERE / 'a17_conditional_dials.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L[-(K + 4):]))
