#!/usr/bin/env python3
"""TAG fable-analyst (AN-64): PER-NET ADAPTIVE ALLOCATION, step 1 = the oracle bound (lead 00:12 / 01:3x). The board score is the mean over
MLPs of MSE_i x util_i (checked on S23's page), so each net can take its own card variant. From the porter's native VALID40 rows
(fable_porter_v1/paired_valid40_*.json, 'cand:<rank>' rows: mse, util, adjusted = mse x max(util, .1)), per variant set:
  best single = min over variants of mean_i adj_iv;   oracle = mean_i min_v adj_iv;   gain = oracle / best single - 1
plus the nets where the per-net winner differs from the best single, and the sd of the per-net log adjusted ratio (for 2-sets).
Noise floor: two equal-mean variants whose per-net rounding-level differences have sd s give an oracle 'gain' of about -s/sqrt(2 pi)
(s ~ .29% from the porter's native-vs-torch parity: ~-0.12%).
usage: oracle_router.py   -> prints the table, writes oracle_router.json"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PO = HERE.parent / 'fable_porter_v1'
V = {'S21': 'paired_valid40_stack_S21_round1_mem.json', 'S21jn': 'paired_valid40_stack_S21jn_mem.json',
     'S21s1': 'paired_valid40_e2_S21s1lr.json', 'S21s1z': 'paired_valid40_e2_S21s1z.json', 'S22': 'paired_valid40_stack_S22s1_r2_mem.json',
     'S23': 'paired_valid40_stack_S23_mem.json', 'S23nys': 'paired_valid40_S23_lrnys_q1.json'}
D, ranks = {}, None
for k, f in V.items():
    d = json.load(open(PO / f)); rk = d['ranks']; ranks = ranks or rk; assert rk == ranks, k
    rows = [d['results'][f'cand:{r}'] for r in rk]
    D[k] = dict(adj=np.array([x['adjusted'] for x in rows]), mse=np.array([x['mse'] for x in rows]), util=np.array([x['util'] for x in rows]))
    assert np.allclose(D[k]['adj'], D[k]['mse'] * np.maximum(D[k]['util'], .1), rtol=1e-9), k
SETS = [('S21', 'S21jn'), ('S21s1', 'S21s1z'), ('S21s1', 'S22'), ('S23', 'S23nys'), ('S21s1z', 'S22'), ('S21', 'S21jn', 'S21s1', 'S21s1z', 'S22'),
        ('S22', 'S23', 'S23nys')]
out = {}
print('variant set | mean adjusted per variant | best single | oracle gain vs best | nets where the winner differs | sd of per-net log ratio (pairs)')
for S in SETS:
    A = np.stack([D[k]['adj'] for k in S]); means = A.mean(1); b = int(np.argmin(means))
    orc = A.min(0).mean(); gain = orc / means[b] - 1; flips = int((A.argmin(0) != b).sum())
    sd = float(np.std(np.log(A[1] / A[0]))) if len(S) == 2 else None
    out['+'.join(S)] = dict(means={k: float(m) for k, m in zip(S, means)}, best=S[b], oracle=float(orc), gain=float(gain), flips=flips, sd_logratio=sd)
    print(f"{' vs '.join(S):40s} | " + ' '.join(f'{k} {m:.4e}' for k, m in zip(S, means)) + f" | {S[b]} | {100 * gain:+.3f}% | {flips}/40 | "
          + (f'{100 * sd:.2f}%' if sd is not None else '-'))
(HERE / 'oracle_router.json').write_text(json.dumps(out, indent=1))
