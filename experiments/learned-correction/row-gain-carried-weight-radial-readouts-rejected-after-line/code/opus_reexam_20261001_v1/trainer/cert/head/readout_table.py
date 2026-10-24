"""TAG opus-trainer (2026-10-02): the lead's BATCH 4 (19:51Z / 19:53Z / 20:01Z): zero-weight gradients of three linear readouts over the training networks
(gradient runs with request field 'rg': 1; results carry 'rg_grads'): the row gain of the bank read (2 x 52 weights: 52 inputs, a constant and a slope in
the layer), the carried b (64 per layer, layers 1-14) and the radial (64 per layer). Per readout: weights beyond 3 standard errors, mean t squared, cosine
between the two halves' mean gradients, first-order change of the pooled final-layer error for a step of the given size along minus the mean gradient with
the noise share of its length taken out; for the row gain also the t of each of the four named inputs. The last line is the TEAM_LOG line.
usage (from O/trainer): python -B cert/head/readout_table.py <gradient label>"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2] / 'run_M/results'
label = sys.argv[1]
rows = [torch.load(f, map_location='cpu', weights_only=False) for f in sorted(R.glob('a_%s_r*.pt' % label))]
rows = [r for r in rows if 'rg_grads' in r]
N = len(rows); mse = float(np.mean([r['mse'] for r in rows]))


def stats(G, step):
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = np.where(se > 0, m / np.maximum(se, 1e-300), 0.0)
    live = se > 0
    a, b = G[0::2].mean(0), G[1::2].mean(0)
    left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
    return int((np.abs(t) > 3).sum()), int(live.sum()), float((t[live] ** 2).mean()), float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-300)), -100 * step * left / mse, t


print('%d training networks; pooled final-layer MSE %.4e' % (N, mse))
G = {name: np.stack([r['rg_grads'][name].double().numpy().reshape(-1) / r['scale'] for r in rows]) for name in ('row', 'b', 'rad')}
out = {}
for name, step in (('row', 0.01), ('b', 0.05), ('rad', 0.05)):
    out[name] = stats(G[name], step)
    print('%-4s beyond 3: %3d of %3d; mean t^2 %.2f; halves %+.2f; first order for a step of %.2f: %+.3f%%' % ((name,) + out[name][:4] + (step, out[name][4])))
t = out['row'][5].reshape(2, 52)
names = ('log sigma', 'local quartic / v^2', '(plane . mu) / v', 'diagonal column')
print('row gain, t of the named inputs (constant weight / slope in the layer):', '; '.join('%s %+.1f / %+.1f' % (n_, t[0, k], t[1, k]) for k, n_ in enumerate(names)))
print('row gain, message columns: beyond 3: %d of 96; largest |t| %.1f' % (int((np.abs(t[:, 4:]) > 3).sum()), float(np.abs(t[:, 4:]).max())))
for name in ('b', 'rad'):
    tt = out[name][5].reshape(16, 64)
    print('%-4s per layer, weights beyond 3 of 64:' % name, ' '.join('%d' % int((np.abs(tt[l]) > 3).sum()) for l in range(1, 15)), '| mean t^2:', ' '.join('%.1f' % float((tt[l] ** 2).mean()) for l in range(1, 15)))
print('BATCH 4, ZERO-WEIGHT GRADIENTS OF THREE READOUTS OVER %d TRAINING NETWORKS (lead 19:51Z-20:01Z; gradient runs %s on the S42 arithmetic in torch: S39 card, law with the '
      'table and three forms, first dials, k31l ages 1-2, the diagonal column at the weights of step 2; definitions O/trainer/otr_m.py comment READOUTS; '
      'python -B O/trainer/cert/head/readout_table.py %s). (o4) ROW GAIN OF THE BANK READ (52 inputs, one weight vector and its slope in the layer = 104 weights, layers '
      '2-14): %d of %d weights beyond 3 standard errors, mean t squared %.2f, split-half cosine %+.2f, first-order %+.3f%% of the pooled final-layer error for a step of 0.01 '
      'along minus the mean gradient (noise share out); t of the named inputs, constant / slope in the layer: %s; message columns: %d of 96 weights beyond 3, largest |t| %.1f. '
      '(o2) THE CARRIED b (64 weights per layer, layers 1-14): %d of %d beyond 3, mean t squared %.2f, halves %+.2f, first-order %+.3f%% for a step of 0.05. '
      '(o3) THE RADIAL (64 per layer): %d of %d, %.2f, %+.2f, %+.3f%% for a step of 0.05. With no signal: 0.3%% of the weights beyond 3, mean t squared 1.0.' % (
          N, label, label, out['row'][0], out['row'][1], out['row'][2], out['row'][3], out['row'][4],
          '; '.join('%s %+.1f / %+.1f' % (n_, t[0, k], t[1, k]) for k, n_ in enumerate(names)), int((np.abs(t[:, 4:]) > 3).sum()), float(np.abs(t[:, 4:]).max()),
          out['b'][0], out['b'][1], out['b'][2], out['b'][3], out['b'][4], out['rad'][0], out['rad'][1], out['rad'][2], out['rad'][3], out['rad'][4]))
