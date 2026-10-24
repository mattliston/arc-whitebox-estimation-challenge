"""TAG opus-trainer (2026-10-02): the lead's SECOND BATCH of head inputs (18:42Z), the same test as cert/law_ls/head_input_table.py: gradients of the
128 zero first-layer weights of each new column over the training networks, here taken from the gradient runs of the head-column solve (so column 6,
the diagonal residual, is at its SOLVED weights and the others at zero). Per column: weights of 128 beyond 3 standard errors, mean t squared, cosine
between the two halves' mean gradients, first-order change for a step of 0.05 along minus the mean gradient with the noise share taken out; six
existing columns as a yardstick. The last line is the TEAM_LOG line.
usage (from O/trainer): python -B cert/head/batch2_table.py <gradient label, e.g. 0hv3rg>"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2] / 'run_M/results'
label = sys.argv[1]
rows = [torch.load(f, map_location='cpu', weights_only=False) for f in sorted(R.glob('a_%s_r*.pt' % label))]
rows = [r for r in rows if 'hu_grads' in r and r['hu_grads'].shape[0] >= 17]
N = len(rows); mse = float(np.mean([r['mse'] for r in rows]))
cols = ((6, 'the diagonal residual on the response rows (at its solved weights)'), (7, 'carried-column part of the diagonal over tr C v_i'), (8, 'pair cross part of the diagonal over tr C v_i'),
        (9, "diagonal of the young pairs' fourth-chaos plane over v_i^2"), (10, "previous layer's row factor rv pushed through W, over sigma_i"),
        (11, "previous layer's variance tilt d o Pv pushed through W, over sigma_i"),
        (12, 'k3 of the young bank columns over v sigma'), (13, "k3 of the old tier's columns over v sigma"), (14, 'k3 of the star over v sigma'),
        (15, 'k3 of the b-form, 3 b_i / sigma_i'), (16, "the K4L local quartic's per-unit value over v^2"))


def stats(G):
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = m / np.maximum(se, 1e-300)
    a, b = G[0::2].mean(0), G[1::2].mean(0)
    left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
    return int((np.abs(t) > 3).sum()), float((t * t).mean()), float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())), -100 * 0.05 * left / mse


print('%d training networks; pooled final-layer MSE %.4e' % (N, mse))
out = []
for k, text in cols:
    s = stats(np.stack([r['hu_grads'][k].double().numpy() / r['scale'] for r in rows])); out.append(s)
    print('%2d %-72s %4d %7.2f %+6.2f %+8.3f%%' % ((k, text) + s))
old = []
for j, name in ((3, 'alpha'), (5, 'gate'), (10, 'pre k3'), (12, 'pre k4'), (20, 'post k3'), (22, 'plane row norm')):
    s = stats(np.stack([r['grads']['p.enc0_w'][:, j].double().numpy() / r['scale'] for r in rows])); old.append(s)
    print('   existing %-63s %4d %7.2f %+6.2f %+8.3f%%' % ((name,) + s))
print('SECOND AND THIRD BATCH OF HEAD INPUTS, GRADIENT OF THEIR ZERO WEIGHTS OVER %d TRAINING NETWORKS (lead 18:42Z and 19:02Z; gradient runs %s of the head-column solve: card = S39 card, law with '
      'the constant regressor and three forms, k31l ages 1-2, second dials, the diagonal column at its solved weights; python -B O/trainer/cert/head/batch2_table.py %s). '
      'Per column: weights of 128 beyond 3 standard errors (0.35 expected with no signal) / mean of t squared (1.0 with no signal, sd 0.125) / cosine between the two halves / '
      'first-order change for a step of 0.05 along minus the mean gradient, noise share out: %s. YARDSTICK, six existing columns of the same layer: mean t squared %s, weights '
      'beyond 3: %s. Definitions: O/trainer/otr_m.py, comment HEAD INPUTS (columns 7-16). The left-factor row sizes (item c) are dropped (trainer 19:28Z).' % (
          N, label, label, '; '.join('%s: %d / %.2f / %+.2f / %+.3f%%' % ((text,) + s) for (k, text), s in zip(cols, out)),
          ', '.join('%.2f' % s[1] for s in old), ', '.join('%d' % s[0] for s in old)))
