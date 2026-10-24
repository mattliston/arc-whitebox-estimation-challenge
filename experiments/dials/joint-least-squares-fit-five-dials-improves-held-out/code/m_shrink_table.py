"""TAG opus-trainer (2026-10-02): price 'scale of a learned group' directions from gradients already stored (no card run; lead 07:13Z).
For a learned part whose output is linear in an array X, the derivative of the loss along X -> X (1 + eps) is <dLoss/dX, X>.
Per network, then mean, standard error, t and sign count over the networks; unit: d(1e8 x final-layer MSE)/d eps, and as a
percentage of the pooled error per unit of eps (a NEGATIVE mean says: scale the group UP; positive: shrink it).
  (a) hidden-state part of the birth coefficients: c = cstar + (1 - |cstar|) tanh(H A): <g_A, A> (exact for the scale of A; the
      scale of the response itself only where tanh is in its linear range)
  (b) learned covariance edge: linear in e.output and in e.cov_factor (both must give the same number); per layer 8-14 the
      physics leaf 'edge' gives the same derivative per layer, times (1 + 0.164)
  (c) cstar per layer and leg j0..j3: g_c[l, j] c[l, j]
  (d) the mean correction's head per layer: the physics leaf 'ds' (already a dial)
usage: python -B m_shrink_table.py <label>:<glob of gradient files> [...]"""
import sys
from pathlib import Path

import numpy as np
import torch

from otr_common import read_source

arrays, _ = read_source(np)
for spec in sys.argv[1:]:
    label, pattern = spec.split(':', 1)
    files = sorted(Path('.').glob(pattern))
    rows = [torch.load(f, map_location='cpu', weights_only=True) for f in files]
    scale = np.array([1e8 / row['scale'] for row in rows])
    mse = np.mean([row['mse'] for row in rows]) * 1e8
    print('\n%s: %d networks, mean final-layer MSE x1e8 %.4f' % (label, len(rows), mse))
    def line(name, x):
        se = x.std(ddof=1) / len(x) ** .5
        t = x.mean() / se if se > 0 else 0.0
        print('  %-44s %+9.4f  se %7.4f  t %+6.2f  same sign %2d/%d   %+7.2f%% per unit%s' % (
            name, x.mean(), se, t, int((np.sign(x) == np.sign(x.mean())).sum()), len(x), 100 * x.mean() / mse, '   <==' if abs(t) > 2.5 else ''))
    g = lambda key: np.stack([row['grads'][key].numpy().astype(np.float64) for row in rows]) * scale.reshape((-1,) + (1,) * arrays[key].ndim)
    line('(a) birth coefficients, hidden-state part (A)', (g('a') * arrays['a']).sum((1, 2)))
    line('(b) covariance edge via e.output', (g('e.output') * arrays['e.output']).reshape(len(rows), -1).sum(1))
    line('(b) covariance edge via e.cov_factor', (g('e.cov_factor') * arrays['e.cov_factor']).reshape(len(rows), -1).sum(1))
    if 'pl_grads' in rows[0]:
        edge = np.stack([row['pl_grads']['edge'].numpy() for row in rows]) * scale[:, None] * 1.164
        for l in range(8, 15):
            line('(b) covariance edge, layer %d' % l, edge[:, l])
    gc = g('c') * arrays['c']
    for j in range(4):
        line('(c) cstar leg j%d, all layers' % j, gc[:, :, j].sum(1))
    for l in range(arrays['c'].shape[0]):
        for j in range(4):
            x = gc[:, l, j]
            se = x.std(ddof=1) / len(x) ** .5
            if se > 0 and abs(x.mean() / se) > 2.5:
                line('(c) cstar leg j%d, layer %d' % (j, l), x)
    line('    head of the mean correction, all layers', (g('p.head') * arrays['p.head']).sum((1, 2)) + (g('p.head_b') * arrays['p.head_b']).sum(1))
    if 'pl_grads' in rows[0] and 'ds' in rows[0]['pl_grads']:
        ds = np.stack([row['pl_grads']['ds'].numpy() for row in rows]) * scale[:, None]
        line('(d) correction scale ds (the dial), layers 1-15', ds.sum(1))
        for l in range(1, 16):
            x = ds[:, l]
            se = x.std(ddof=1) / len(x) ** .5
            if se > 0 and abs(x.mean() / se) > 2.5:
                line('(d) correction scale, layer %d' % l, x)
