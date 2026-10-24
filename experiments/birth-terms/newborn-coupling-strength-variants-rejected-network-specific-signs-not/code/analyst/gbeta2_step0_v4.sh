#!/bin/bash
# TAG analyst gbeta2 STEP 0 v4 (scout-card 10-13 01:2xZ: f_kink in the GATE's definition, alpha = exact mean / Monte Carlo sd, from step0_fkink_v2.json; the
# Gaussian-inverted f_kink of step0_feat_v1.json is printed for information; lambda_2 / lambda_1 from step0_feat_v1.json). v3 text: (lead 10-13 01:2xZ amendment: the VERDICT uses ONLY the physicist's 3 pre-declared features, f_kink mean L12-15, f_kink
# mean L8-11, lambda_2 / lambda_1 at L15: PASSES if any |r| >= 0.3 at n = 24, else DROPPED; per-layer L8-14 values are information only; chance of a pass
# about 1 in 3). v2 text: when scout-card's 16 extension evals land (16 files
# run_k7/results/sc_k72ca_r*_gk+gbeta2.json) AND step0_feat_v1.py has finished (24 networks 7100584-607), post ONE TEAM_LOG line (date -u): gk+gbeta2 vs gk
# per network (log final-MSE ratio, scout-card's k_line_v1.py arithmetic) correlated (Pearson r, Spearman rho) with f_kink L8-11 / L12-15 / per layer
# L8-14 and their mean, the dead fraction, and lambda_2 / lambda_1 of Cpre at EACH layer L8-14, their mean and L15. VERDICT stated explicitly by the lead's
# pre-declared rule: if every |r| < 0.3 on f_kink (L12-15, L8-11) and lambda_2 / lambda_1 (L15; per layer L8-14 and their mean), the gated beta is DROPPED.
# Polls files only. Gives up at 2026-10-13T05:00Z. Log: gbeta2_step0_v4.out.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1; A=$O/analyst; S=$O/scout_card
until { [ $(ls $S/run_k7/results 2>/dev/null | grep -c '^sc_k72ca_r.*_gk+gbeta2\.json$') -ge 16 ] && grep -q '"done": true' $A/step0_feat_v1.out 2>/dev/null && grep -q '"done": true' $A/step0_fkink_v2only.out 2>/dev/null; } || [ "$(date -u +%Y%m%dT%H%M)" \> "20261013T0500" ]; do sleep 60; done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python - <<PYEOF >> $O/TEAM_LOG.md
import json, math, subprocess
from pathlib import Path
import numpy as np
t = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(); S = Path('$S'); A = Path('$A')
res = {}
for one in ('run_k1', 'run_k7'):
    for f in sorted((S / one / 'results').glob('sc_*.json')):
        r = json.loads(f.read_text()); seed, a = r['id'].split('_r')[1].split('_', 1)
        if 7100584 <= int(seed) <= 7100607: res.setdefault(a, {}).setdefault(int(seed), r)
eff = {s: math.log(r['mse'] / res['gk'][s]['mse']) * 100 for s, r in res.get('gk+gbeta2', {}).items() if s in res.get('gk', {})}
try:
    F = {int(k): v for k, v in json.loads((A / 'step0_feat_v1.json').read_text()).items()}; K = {int(k): v for k, v in json.loads((A / 'step0_fkink_v2.json').read_text()).items()}
    for s in list(F):
        if s not in K: del F[s]; continue
        F[s]['kink 12-15 (Gaussian-inverted s)'] = F[s]['kink 12-15']; F[s]['kink 8-11 (Gaussian-inverted s)'] = F[s]['kink 8-11']
        F[s].update({k: v for k, v in K[s].items() if not k.endswith('(Gaussian-inverted s)')})
except Exception: F = {}
seeds = sorted(s for s in eff if s in F)
if len(seeds) < 20:
    print(f"[analyst] {t} GBETA2 STEP 0: only {len(seeds)} networks with both gbeta2's effect and the features (effects {len(eff)}, features {len(F)}); STEP 0 NOT DECIDED, the analyst redoes it by hand.")
else:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float); e = np.array([eff[s] for s in seeds])
    def c(k):
        x = np.array([F[s][k] for s in seeds]); return float(np.corrcoef(x, e)[0, 1]), float(np.corrcoef(rank(x), rank(e))[0, 1])
    rule = ['kink 12-15', 'kink 8-11', 'lambda_2 / lambda_1 L15']
    other = ['kink 12-15 (Gaussian-inverted s)', 'kink 8-11 (Gaussian-inverted s)', 'lambda_2 / lambda_1 L8-14 mean'] + [f'lambda_2 / lambda_1 L{l}' for l in range(8, 15)] + ['kink 8-14 mean', 'dead 12-15'] + [f'kink L{l}' for l in range(8, 15)]
    R = {k: c(k) for k in rule + other}
    best = max(rule, key=lambda k: abs(R[k][0]))
    cells = 'VERDICT FEATURES: ' + '; '.join(f"{k}: r {R[k][0]:+.2f} rho {R[k][1]:+.2f}" for k in rule) + ' | INFORMATION ONLY (band / input layer after a pass): ' + '; '.join(f"{k}: r {R[k][0]:+.2f}" for k in other)
    verdict = (f"PASSES: {best} r {R[best][0]:+.2f} (|r| >= 0.3 on a pre-declared feature), the gated beta goes on to the fresh 24-network confirm" if abs(R[best][0]) >= 0.3 else
               f"FAILS: all three pre-declared features |r| < 0.3 (largest {best} {R[best][0]:+.2f}): the gated beta is DROPPED") + '; chance of a pass on noise alone about 1 in 3 (0.17 per feature at n = 24, three correlated features)'
    print(f"[analyst] {t} GBETA2 STEP 0, n = {len(seeds)} (lead 01:1xZ, pre-declared; amended 01:2xZ: verdict on the 3 pre-declared features only; gk+gbeta2 vs gk per network, log final-MSE ratio: mean {e.mean():+.2f}%, sd {e.std():.2f}%; features: f_kink = share of units with |alpha| < 1.5, alpha = exact mean / Monte Carlo sd (the gate's definition, O/analyst/step0_fkink_v2.json), lambda_2 / lambda_1 of Cpre from 16,384 inputs (O/analyst/step0_feat_v1.json); negative r = helps more where the feature is larger): {cells}. VERDICT: {verdict}.")
PYEOF
echo "$(date -u +%FT%TZ) posted"
