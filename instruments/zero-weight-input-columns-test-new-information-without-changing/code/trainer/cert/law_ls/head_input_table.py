"""TAG opus-trainer (2026-10-02): the lead's test of NEW HEAD INPUTS (17:52Z): each new per-unit column enters the learned mean correction's first layer
with 128 weights that are zero; this reads the exact gradients of those weights over the training networks (gradient runs with the law installed and
the head inputs on: results a_c3q4zm_lawg_r*.pt that carry 'hu_grads' and a_c3q4zm_lawg_z_r*.pt) and prints per column:
  beyond 3     how many of its 128 weights have a mean gradient beyond 3 standard errors (0.35 expected with no signal)
  mean t^2     the mean of t squared over the 128 weights (1.0 with no signal, sd 0.125)
  halves       cosine between the mean gradients of the two halves of the networks (0 with no signal)
  gain         first-order change of the pooled final-layer error for a step of 0.05 along minus the mean gradient, with the part of the mean gradient's
               length that noise alone would give taken out (0 if nothing is left)
and the same numbers for six EXISTING input columns of enc0_w as a yardstick (trained weights: no signal expected). The last line is the TEAM_LOG line.
usage (from O/trainer): python -B cert/law_ls/head_input_table.py"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2] / 'run_M/results'
files = sorted(R.glob('a_c3q4zm_lawg_r*.pt')) + sorted(R.glob('a_c3q4zm_lawg_z_r*.pt'))
rows = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
rows = [r for r in rows if 'hu_grads' in r]
N = len(rows)
mse = float(np.mean([r['mse'] for r in rows]))
names = ('diag_resid', 'law_rownorm', 'law_rowsum_rho', 'k31l_rownorm', 'k31l_rowsum_rho', 'forms_share')
text = ('the unit-specific part of the law matrix diagonal over tr C v_i', "row norm of the law's fold term", "the law's fold term summed against the correlation",
        'row norm of the k31l term', 'the k31l term summed against the correlation', 'fitted diagonal over B_ii')


def stats(G):      # G: networks x 128, d MSE / d weight
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = m / np.maximum(se, 1e-300)
    a, b = G[0::2].mean(0), G[1::2].mean(0)
    left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
    return int((np.abs(t) > 3).sum()), float((t * t).mean()), float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())), -100 * 0.05 * left / mse


print('%d training networks with the head inputs; pooled final-layer MSE %.4e' % (N, mse))
print('%-18s %9s %9s %8s %10s' % ('column', 'beyond 3', 'mean t^2', 'halves', 'gain 0.05'))
out = []
for k, name in enumerate(names):
    s = stats(np.stack([r['hu_grads'][k].double().numpy() / r['scale'] for r in rows]))
    out.append(s)
    print('%-18s %9d %9.2f %+8.2f %+9.3f%%' % ((name,) + s))
ALL = np.stack([np.concatenate([r['hu_grads'][k].double().numpy() / r['scale'] for k in range(len(names))]) for r in rows])
m = ALL.mean(0); se = ALL.std(0, ddof=1) / N ** .5
left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
print('all six together: first-order %+.3f%% for a step of 0.05 along minus the mean gradient (noise share taken out)' % (-100 * 0.05 * left / mse))
old = []
for j, label in ((3, 'alpha'), (5, 'gate'), (10, 'pre k3'), (12, 'pre k4'), (20, 'post k3'), (22, 'plane row norm')):
    s = stats(np.stack([r['grads']['p.enc0_w'][:, j].double().numpy() / r['scale'] for r in rows]))
    old.append(s)
    print('existing %-9s %9d %9.2f %+8.2f %+9.3f%%' % ((label,) + s))
print('NEW HEAD INPUTS, GRADIENT OF THEIR ZERO WEIGHTS OVER %d TRAINING NETWORKS (lead 17:52Z; S39 card + law (constant regressor, three forms) + k31l, second-solve '
      'dials; each column enters the first layer of the learned mean correction with 128 weights at zero, so predictions are unchanged and the gradient is exact; '
      'python -B O/trainer/cert/law_ls/head_input_table.py). Per column: weights of 128 whose mean gradient is beyond 3 standard errors (0.35 expected with no signal) / '
      'mean of t squared over the 128 (1.0 with no signal, sd 0.125) / cosine between the mean gradients of the two halves of the networks / first-order change of the '
      'pooled final-layer error for a step of 0.05 along minus the mean gradient with the noise share of its length taken out: %s. All six together: %+.3f%%. YARDSTICK, '
      'six existing input columns of the same layer (trained weights): mean t squared %s, weights beyond 3: %s.' % (
          N, '; '.join('%s: %d / %.2f / %+.2f / %+.3f%%' % ((t_,) + s) for t_, s in zip(text, out)), -100 * 0.05 * left / mse,
          ', '.join('%.2f' % s[1] for s in old), ', '.join('%d' % s[0] for s in old)))
