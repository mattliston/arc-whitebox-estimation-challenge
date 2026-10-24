#!/usr/bin/env python3
"""TAG analyst hard public networks v1 (lead 2026-10-12 23:3xZ): why is our estimator worse than the leaders on certain public networks (15, 40, 16, 14)?
Board data only (scan_merged.jsonl: per public network the final-layer MSE and the all-layers MSE of every graded entry, the billed flops per network).
The public networks' WEIGHTS cannot be regenerated: whestbench builds them from SeedSequence(spec seed).spawn(3 n) and names them from the third
substream; no spec seed in 0-300,000 (or the usual round numbers) gives 'natasha-heath' for network 0, so the per-network weight features the lead
asked for are out of reach; this uses what the board shows per network instead.
Residual of an entry on network i = log MSE_i(entry) - mean over the four consensus entries of log MSE_i, minus its mean over the 50 networks.
  (1) persistence: our current #335633 against older entries of ours (#335349 S48e-era, #333607 S42);
  (2) final-layer residual against the residual of layers 0-14 (MSE_0-14 = (16 all - final) / 15): is the excess made only at the last layer?
  (3) correlations (Pearson r and Spearman rho over the 50 networks) of our final-layer residual with: consensus difficulty (final, layers 0-14,
      their ratio = late growth), our layers 0-14 residual, our and the rivals' billed flops per network, each rival's own residual.
usage: hard_nets_v1.py   -> table on stdout and hard_nets_v1.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
rows = {}
for l in open(HERE / 'scan_merged.jsonl'):
    r = json.loads(l)
    if r.get('http') == 200: rows[r['id']] = r
CONS = {'marius_binner': 335629, 'oliver_chen': 335467, 'Luna': 335625, 'J2W': 335438}; OURS = 335633; OLD = {'S48e #335349': 335349, 'S42 #333607': 333607}


def nets(i):
    r = rows[i]; p = sorted(r['pub']); fin = np.array([x[1] for x in p]); al = np.array([x[2] for x in p])
    return fin, (16 * al - fin) / 15, np.array(r['flops'][:50], dtype=float)


F = {t: nets(i) for t, i in CONS.items()}; ours = nets(OURS)
cf = np.mean([np.log(F[t][0]) for t in CONS], 0); ce = np.mean([np.log(F[t][1]) for t in CONS], 0)
res = lambda x, c: (np.log(x) - c) - (np.log(x) - c).mean()
rf, re_ = res(ours[0], cf), res(ours[1], ce)


def rank(x): return np.argsort(np.argsort(x)).astype(float)
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(rank(a), rank(b))[0, 1])


out = dict(our_final_residual_sd=float(rf.std()), worst=[[int(i), round(float(rf[i]) * 100, 1)] for i in np.argsort(-rf)[:8]])
print(f"our #335633 final-layer residual against the consensus: sd {rf.std() * 100:.1f}%; worst: " + ', '.join(f"{i} {v:+.1f}%" for i, v in out['worst']))
pers = {}
for name, i in OLD.items():
    if i in rows:
        o = nets(i); ro = res(o[0], cf); pers[name] = corr(rf, ro)
        print(f"(1) persistence vs {name}: r {pers[name][0]:+.2f}, rho {pers[name][1]:+.2f}; its worst: " + ', '.join(f"{j} {ro[j] * 100:+.0f}%" for j in np.argsort(-ro)[:5]))
out['persistence'] = pers
feats = {'consensus final-layer log MSE (difficulty)': cf - cf.mean(), 'consensus layers 0-14 log MSE': ce - ce.mean(),
         'consensus late growth log(final / layers 0-14)': (cf - ce) - (cf - ce).mean(), 'OUR layers 0-14 residual': re_,
         'our billed flops (log)': np.log(ours[2]) - np.log(ours[2]).mean()}
for t in CONS:
    feats[f'{t} billed flops (log)'] = np.log(F[t][2]) - np.log(F[t][2]).mean()
    others = [u for u in CONS if u != t]; c3 = np.mean([np.log(F[u][0]) for u in others], 0)
    feats[f'{t} final residual vs the other three'] = res(F[t][0], c3)
tab = sorted(((k, *corr(rf, v)) for k, v in feats.items()), key=lambda x: -abs(x[1]))
print('(2)-(3) correlation with our final-layer residual over the 50 public networks (|r| > 0.28 is about p < 0.05):')
for k, r_, rho in tab: print(f"  {k}: r {r_:+.2f}, rho {rho:+.2f}")
w = [int(i) for i in np.argsort(-rf)[:4]]
print('worst networks, values: ' + '; '.join(f"net {i}: our final {rf[i] * 100:+.0f}%, our layers 0-14 {re_[i] * 100:+.0f}%, difficulty {(cf - cf.mean())[i] * 100:+.0f}%, late growth {((cf - ce) - (cf - ce).mean())[i] * 100:+.0f}%" for i in w))
out['correlations'] = [dict(feature=k, r=r_, rho=rho) for k, r_, rho in tab]
out['worst_values'] = {str(i): dict(final=float(rf[i]), early=float(re_[i]), difficulty=float((cf - cf.mean())[i]), late_growth=float(((cf - ce) - (cf - ce).mean())[i])) for i in w}
(HERE / 'hard_nets_v1.json').write_text(json.dumps(out, indent=1))
