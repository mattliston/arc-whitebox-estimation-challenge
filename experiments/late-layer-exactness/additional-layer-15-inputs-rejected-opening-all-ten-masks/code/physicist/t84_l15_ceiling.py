#!/usr/bin/env python3
"""TAG opus-physicist T84: HOW MUCH OF THE FINAL-LAYER RESIDUAL IS PREDICTABLE AT LAYER 15 AT ALL (a ceiling for any new layer-15 head input or head refit).
Target per unit: r_i = (truth_i - final_i) / sigma_i, final = the card's final prediction at jmIv (the residual its own layer-15 correction leaves).
A flexible regressor (sklearn HistGradientBoosting, iterations chosen by 5-fold cross-validation over the training NETWORKS) predicts r from
  S0 = the head's 128 first-layer pre-activations at layer 15 (everything the current correction sees, linearly mixed),
  S1 = S0 + the ten masked inputs 13-15, 22-28 (head-normalised),
  S2 = the ten masked inputs alone;
the corrected prediction final + shrink x sigma x r_hat (shrink chosen on the training networks) is judged once on the held-out networks.
usage: t84_l15_ceiling.py <label> <training csv> <held-out csv>"""
import sys, math
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = Path(__file__).resolve().parent
LABEL, TR0, VA0 = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
have = lambda r: (HERE / 'l15j_dump' / f'{LABEL}_{r}.npz').exists()
TRAIN, VAL = [r for r in TR0 if have(r)], [r for r in VA0 if have(r)]
FEATS = (13, 14, 15, 22, 23, 24, 25, 26, 27, 28)
NET = {}
for r in TRAIN + VAL:
    D = np.load(HERE / 'l15j_dump' / f'{LABEL}_{r}.npz'); c, s = D['centers'], D['scales']
    z = np.stack([np.clip((D['f%d' % k] - c[k]) / s[k], -8, 8) for k in FEATS], 1)
    sig = D['sigma'].reshape(-1)
    NET[r] = dict(S0=D['pre'], S1=np.concatenate([D['pre'], z], 1), S2=z, sig=sig, fin=D['final'], y=D['truth'], r=(D['truth'] - D['final']) / sig)


def stack(nets, key): return np.concatenate([NET[r][key] for r in nets])


def change(nets, preds, shrink):
    a = np.array([((NET[r]['fin'] + shrink * NET[r]['sig'] * p - NET[r]['y']) ** 2).mean() for r, p in zip(nets, preds)])
    b = np.array([((NET[r]['fin'] - NET[r]['y']) ** 2).mean() for r in nets])
    return a, b


STAGES = (25, 50, 100, 200, 400); SHRINK = (0.25, 0.5, 0.75, 1.0)
model = lambda: HistGradientBoostingRegressor(max_iter=max(STAGES), learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
print('T84 %s: %d training, %d held-out networks; base final-layer MSE train %.4e held-out %.4e' % (
    LABEL, len(TRAIN), len(VAL), np.mean([((NET[r]['fin'] - NET[r]['y']) ** 2).mean() for r in TRAIN]), np.mean([((NET[r]['fin'] - NET[r]['y']) ** 2).mean() for r in VAL])), flush=True)
for key in ('S0', 'S1', 'S2'):
    folds = [f for f in (TRAIN[i::5] for i in range(5)) if f]; cv = {}
    for f in folds:
        fit_on = [r for r in TRAIN if r not in set(f)]
        mdl = model().fit(stack(fit_on, key), stack(fit_on, 'r'))
        stages = list(mdl.staged_predict(stack(f, key)))
        for it in STAGES:
            p = stages[it - 1]; cut = np.cumsum([0] + [len(NET[r]['r']) for r in f]); preds = [p[cut[i]:cut[i + 1]] for i in range(len(f))]
            for sh in SHRINK:
                a, b = change(f, preds, sh); cv.setdefault((it, sh), []).extend(list(a / b))
    (it, sh), best = min(cv.items(), key=lambda kv: np.mean(kv[1]))
    mdl = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0).fit(stack(TRAIN, key), stack(TRAIN, 'r'))
    a, b = change(VAL, [mdl.predict(NET[r][key]) for r in VAL], sh); lr = np.log(a / b)
    print('  %s: cv on training networks %+.3f%% (iterations %d, shrink %.2f) | HELD-OUT pooled %+.3f%%, mean log %+.3f%% (se %.3f%%), lower on %d of %d' % (
        key, 100 * (np.mean(best) - 1), it, sh, 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(a)), flush=True)
