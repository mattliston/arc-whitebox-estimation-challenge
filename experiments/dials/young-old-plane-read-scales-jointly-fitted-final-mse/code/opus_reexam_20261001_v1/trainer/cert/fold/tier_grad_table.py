"""TAG opus-trainer (2026-10-02): exact gradients of the two read-only tier scales (rso = old tier's column weights, rsy = young tier's; otr_m.py)
over the training networks whose gradient runs carry them, as % of the pooled final-layer error per +1% of the scale: per layer and summed, mean
and t over the networks; the same for a few other dials as a yardstick.
usage (from O/trainer): python -B cert/fold/tier_grad_table.py <label[,label...]> [family ...]"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path('run_M/results')
files = [f for label in sys.argv[1].split(',') for f in sorted(R.glob('a_%s_r*.pt' % label))]
rows = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
rows = [r for r in rows if 'rso' in r['pl_grads']]
mse = np.mean([r['mse'] for r in rows])
print('%d networks; pooled final-layer MSE %.4e' % (len(rows), mse))
for family in (sys.argv[2:] or ['rso', 'rsy']):
    G = np.stack([r['pl_grads'][family].numpy() / r['scale'] for r in rows])        # d MSE / d shift, networks x layers
    per = 100 * 0.01 * G / mse                                                       # % of the pooled error per +0.01 of the shift, per network and layer
    m, se = per.mean(0), per.std(0, ddof=1) / len(rows) ** .5
    tot = per.sum(1)
    print('%-4s all layers: %+.3f%% per +0.01 (t %+.1f over the networks; same sign on %d of %d)' % (family, tot.mean(), tot.mean() / (tot.std(ddof=1) / len(rows) ** .5),
                                                                                              int((np.sign(tot) == np.sign(tot.mean())).sum()), len(rows)))
    print('     per layer %:', ' '.join('%+.3f' % x for x in m))
    print('     t          :', ' '.join('%+.1f' % (x / y) if y > 0 else '   .' for x, y in zip(m, se)))
