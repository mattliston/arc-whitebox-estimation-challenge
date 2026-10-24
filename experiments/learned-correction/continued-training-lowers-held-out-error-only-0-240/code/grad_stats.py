"""TAG opus-trainer (2026-10-01): is there a consistent descent direction, or are the per-network gradients noise around zero?
Reads the per-network gradient files of updates [first, stop) of a run (each is the gradient of 1e8/batch x final-layer MSE of one
training network; rescaled here to 1e8 x MSE). Treats the parameters as fixed over that window (true to first order while the
learning rate is small). Prints, for all trainable arrays together, for the dials, and per array:
  |mean|        norm of the mean gradient over the N networks
  noise         root of the summed per-coordinate variance across networks (the norm of one network's deviation from the mean)
  split cos     cosine between the mean gradients of the odd and the even networks (0 = noise, 1 = same direction)
  |t|>2, |t|>3  share of coordinates whose mean differs from zero by more than 2 / 3 standard errors (pure noise: 4.6% / 0.27%)
  rate %/upd    the first-order change of the TRAINING loss per update if every update pushed along the true mean gradient the
                way Adam does at batch 12 with the trainer's learning rates: -sqrt(12) * sum_j lr_j (g_j^2 - s_j^2/N) / s_j
                (g_j, s_j = mean and spread of coordinate j; the -s_j^2/N removes the part of g_j^2 that is noise), as a
                percentage of the mean loss, with its standard error under "no mean gradient" (from the Gram matrix of the
                centred gradients). A rate that is not several standard errors below zero means: nothing to descend along.
usage: python -B grad_stats.py <run dir> <first update> <stop update>"""
import sys
from pathlib import Path

import torch

from otr_common import DIAL_LR, TRUNK

RUN, FIRST, STOP = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows, losses = [], []
for update in range(FIRST, STOP):
    files = sorted((RUN / 'results').glob('g%05d_*.pt' % update))
    assert files, update
    for path in files:
        row = torch.load(path, map_location='cpu', weights_only=True)
        factor = 1e8 / row['scale']
        losses.append(1e8 * row['mse'])
        rows.append(({key: value.double().reshape(-1) * factor for key, value in row['grads'].items()},
                     {key: value.double().reshape(-1) * factor for key, value in row['dial_grads'].items()}))
N, loss = len(rows), sum(losses) / len(losses)
keys = list(rows[0][0])
stack = {key: torch.stack([row[0][key] for row in rows]) for key in keys}             # (N, size)
rates = {key: torch.full((stack[key].shape[1],), 5e-5 if key[:2] == 'p.' and key[2:] in TRUNK else 1e-4, dtype=torch.float64)
         for key in keys}
stack['DIALS'] = torch.stack([torch.cat(list(row[1].values())) for row in rows])
rates['DIALS'] = torch.tensor([DIAL_LR[key] for key in rows[0][1]], dtype=torch.float64)
stack['ALL ARRAYS'] = torch.cat([stack[key] for key in keys], dim=1)
rates['ALL ARRAYS'] = torch.cat([rates[key] for key in keys])
print('%d networks from updates %d..%d of %s; gradient of 1e8 x final-layer MSE; mean loss %.4f' % (N, FIRST, STOP - 1, RUN, loss))
print('%-18s %6s %10s %10s %9s %7s %7s   %s' % ('array', 'size', '|mean|', 'noise', 'split cos', '|t|>2', '|t|>3',
                                               'rate %/upd (standard error)'))
for key in ['ALL ARRAYS', 'DIALS'] + keys:
    g, lr = stack[key], rates[key]
    live = g.var(0, unbiased=True) > 0
    g, lr = g[:, live], lr[live]
    mean, var = g.mean(0), g.var(0, unbiased=True)
    sd = var.sqrt()
    a, b = g[0::2].mean(0), g[1::2].mean(0)
    t = mean / (var / N).sqrt()
    weight = lr / sd
    rate = -12 ** 0.5 * float((weight * (mean ** 2 - var / N)).sum())
    centred = (g - mean) * weight.sqrt()
    gram = centred @ centred.T
    error = 12 ** 0.5 * (2 ** 0.5 / N) * float((gram ** 2).sum().sqrt()) / (N - 1)
    print('%-18s %6d %10.3e %10.3e %9.3f %6.1f%% %6.1f%%   %+.4f%% (%.4f%%)' % (
        key, g.shape[1], float(mean.norm()), float(var.sum().sqrt()), float(a @ b / (a.norm() * b.norm())),
        100 * float((t.abs() > 2).double().mean()), 100 * float((t.abs() > 3).double().mean()),
        100 * rate / loss, 100 * error / loss))
names = list(rows[0][1])
mean, sd = stack['DIALS'].mean(0), stack['DIALS'].std(0)
print('dials: mean gradient / standard error = ' + '  '.join(
    '%s %+.1f' % (name, float(mean[i] / (sd[i] / N ** 0.5))) for i, name in enumerate(names)))
