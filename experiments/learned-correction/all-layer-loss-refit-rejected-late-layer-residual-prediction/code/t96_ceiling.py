#!/usr/bin/env python3
"""TAG opus-physicist T96: PER-LAYER PREDICTABILITY CEILING at layers 12-14 (lead 10-11 08:10Z, P2 = the gate for an all-layer-loss refit).
Card: S47 physics = the trainer's frozen Nd point on the v11 cut simulator (t96_worker_dumpl.py dumps, certified: the replica's correction = the
card's bit for bit). Target per unit at layer l: r = (truth_l - m_l) / sigma_l, m_l = the card's mean at layer l (its correction included).
A flexible regressor (sklearn HistGradientBoosting, iterations and shrink by 5-fold cross-validation over the FIT networks 7100136-151) predicts r
from S0 = the head's 128 first-layer pre-activations at l (everything the correction sees, linearly mixed), S1 = S0 + 4 cheap candidates
(the unit's variance share in the top 1 / 4 / 16 eigenmodes of C_pre, b_pre / sigma), S2 = S1 + the 30 raw head features; judged once on the
held-out networks 7100152-167: % of the per-layer residual power removed (m_l + shrink sigma r_hat vs the truth), raw and with the truth's
Monte Carlo noise taken out (noise per unit = (half A - half B)^2 / 4 of the plan's truth).
usage: t96_ceiling.py [run dir]"""
import json, math, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 't96_dump1214'
FIT, VAL = list(range(7100136, 7100152)), list(range(7100152, 7100168))
NET = {}
for s in FIT + VAL:
    fd, fr = RUN / 'dumps' / ('L_%d.npz' % s), RUN / 'results' / ('t96_%d_L.json' % s)
    if not (fd.exists() and fr.exists()): continue
    D, R = np.load(fd), json.loads(fr.read_text()); T = np.load(R['truth_metadata']['path'])
    pred = np.array(R['prediction'], dtype=np.float64)
    NET[s] = {}
    for l in (12, 13, 14):
        sig = D['sigma_%d' % l].reshape(-1); extra = np.stack([D['mode1_%d' % l], D['mode4_%d' % l], D['mode16_%d' % l], D['bpre_%d' % l]], 1)
        y = T['mc_post'][l].astype(np.float64); noise = (T['mc_post_A'][l].astype(np.float64) - T['mc_post_B'][l].astype(np.float64)) ** 2 / 4
        NET[s][l] = dict(S0=D['pre_%d' % l], S1=np.concatenate([D['pre_%d' % l], extra], 1), S2=np.concatenate([D['pre_%d' % l], extra, D['x_%d' % l]], 1),
                         sig=sig, m=pred[l], y=y, noise=noise, r=(y - pred[l]) / sig)
fit, val = [s for s in FIT if s in NET], [s for s in VAL if s in NET]
print('T96 %s: %d fit + %d held-out networks (S47 / Nd point, v11 simulator)' % (RUN.name, len(fit), len(val)), flush=True)
STAGES, SHRINK = (25, 50, 100, 200, 400), (0.25, 0.5, 0.75, 1.0)
gbm = lambda it: HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
for l in (12, 13, 14):
    mse = lambda nets: np.mean([((NET[s][l]['m'] - NET[s][l]['y']) ** 2).mean() for s in nets])
    nz = lambda nets: np.mean([NET[s][l]['noise'].mean() for s in nets])
    print('layer %d: card per-layer MSE fit %.4e / held-out %.4e; truth noise share held-out %.1f%%' % (l, mse(fit), mse(val), 100 * nz(val) / mse(val)), flush=True)
    for key in ('S0', 'S1', 'S2'):
        folds = [f for f in (fit[i::5] for i in range(5)) if f]; cv = {}
        for f in folds:
            tr = [s for s in fit if s not in set(f)]
            mdl = gbm(max(STAGES)).fit(np.concatenate([NET[s][l][key] for s in tr]), np.concatenate([NET[s][l]['r'] for s in tr]))
            stages = list(mdl.staged_predict(np.concatenate([NET[s][l][key] for s in f])))
            cut = np.cumsum([0] + [len(NET[s][l]['r']) for s in f])
            for it in STAGES:
                p = stages[it - 1]
                for sh in SHRINK:
                    num = sum(((NET[s][l]['m'] + sh * NET[s][l]['sig'] * p[cut[i]:cut[i + 1]] - NET[s][l]['y']) ** 2).mean() for i, s in enumerate(f))
                    den = sum(((NET[s][l]['m'] - NET[s][l]['y']) ** 2).mean() for s in f)
                    cv.setdefault((it, sh), []).append(num / den)
        (it, sh), best = min(cv.items(), key=lambda kv: np.mean(kv[1]))
        mdl = gbm(it).fit(np.concatenate([NET[s][l][key] for s in fit]), np.concatenate([NET[s][l]['r'] for s in fit]))
        a = np.array([((NET[s][l]['m'] + sh * NET[s][l]['sig'] * mdl.predict(NET[s][l][key]) - NET[s][l]['y']) ** 2).mean() for s in val])
        b = np.array([((NET[s][l]['m'] - NET[s][l]['y']) ** 2).mean() for s in val]); n_ = np.array([NET[s][l]['noise'].mean() for s in val])
        lr = np.log(a / b)
        print('  %s: cv %+.2f%% (it %d, shrink %.2f) | HELD-OUT residual power removed %.2f%% raw, %.2f%% of the noise-free power; per network mean log %+.2f%% (se %.2f%%), lower on %d of %d' % (
            key, 100 * (np.mean(best) - 1), it, sh, 100 * (1 - a.sum() / b.sum()), 100 * (b.sum() - a.sum()) / (b.sum() - n_.sum()), 100 * lr.mean(),
            100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(a)), flush=True)
