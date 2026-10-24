#!/usr/bin/env python3
"""TAG opus-physicist T79b (numpy): table of a T79 fresh judgement: candidate label against base label on the same generated seeds: pooled final-layer MSE change,
the same with each seed's truth noise (mean over units of mc_post_var / N at layer 15, from the analyst's truth file) taken out of both, mean log ratio, se, seeds lower.
usage: t79b_table.py <tag> <candidate label> <base label>"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
tag, cand, base = sys.argv[1:4]
R = {}
for line in open(HERE / f't79_{tag}.jsonl'):
    d = json.loads(line); R.setdefault(d['seed'], {})[d['label']] = d['mse']
seeds = sorted(s for s in R if cand in R[s] and base in R[s])
noise = {}
for s in seeds:
    t = np.load(O / f'analyst/fresh/truth_seed{s}.npz'); noise[s] = float(np.mean(t['mc_post_var'][15]) / float(t['N']))
a = np.array([R[s][cand] for s in seeds]); b = np.array([R[s][base] for s in seeds]); nz = np.array([noise[s] for s in seeds])
lr = np.log(a / b); lrc = np.log((a - nz) / (b - nz))
print('T79 %s: %s against %s on %d untouched generated seeds (%d..%d): pooled %+.3f%% (truth noise out %+.3f%%); mean log ratio %+.3f%% (se %.3f%%; noise out %+.3f%%, se %.3f%%); lower on %d of %d; '
      'mean MSE %.4e / %.4e, mean truth noise %.2e' % (tag, cand, base, len(seeds), seeds[0], seeds[-1], 100 * (a.sum() / b.sum() - 1), 100 * ((a - nz).sum() / (b - nz).sum() - 1),
      100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), 100 * lrc.mean(), 100 * lrc.std(ddof=1) / math.sqrt(len(lrc)), int((a < b).sum()), len(seeds), a.mean(), b.mean(), nz.mean()))
