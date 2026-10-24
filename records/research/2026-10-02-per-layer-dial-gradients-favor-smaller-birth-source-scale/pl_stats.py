"""TAG opus-trainer (2026-10-01): do the per-layer physics dials have a SYSTEMATIC gradient across networks?
Reads the 'pl_grads' of the per-network gradient files of updates [first, stop) of a run (workers started with TRAINER_PL=1;
otr_pl.py) and rescales them to the gradient of 1e8 x final-layer MSE per unit of the value. Per scalar: mean over the N
networks, standard error, t = mean / standard error, share of networks with a positive gradient. Prints per family the scalars
with |t| >= <tmin> (default 3) and, per family, how many layers have |t| above 2 and 3 (pure noise: 4.6% and 0.27%), the t of
the family's sum over layers (what a single global dial would see), and an estimate of what the per-layer freedom is worth
beyond the global dial: sum over layers of (mean^2 - se^2) relative to the same for the layer-sum, in units of gradient^2.
The parameters move slowly during the window; to first order the gradients are at one point.
usage: python -B pl_stats.py <run dir> <first update> <stop update> [tmin]"""
import sys
from pathlib import Path

import torch

RUN, FIRST, STOP = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
TMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
rows = []
for update in range(FIRST, STOP):
    for path in sorted((RUN / 'results').glob('g%05d_*.pt' % update)):
        row = torch.load(path, map_location='cpu', weights_only=True)
        if 'pl_grads' in row:
            rows.append({family: value.double() * (1e8 / row['scale']) for family, value in row['pl_grads'].items()})
N = len(rows)
print('%d networks with per-layer gradients, updates %d..%d of %s' % (N, FIRST, STOP - 1, RUN))
print('%-8s %6s %6s %6s %12s   scalars with |t| >= %g: layer mean (t, share positive)' % ('family', 'active', '|t|>2', '|t|>3', 't of the sum', TMIN))
total = strong2 = strong3 = 0
for family in rows[0]:
    g = torch.stack([row[family] for row in rows])              # N x 16
    live = [l for l in range(16) if bool(g[:, l].abs().sum() > 0)]
    mean, se = g.mean(0), g.std(0) / N ** 0.5
    t = mean / se.clamp_min(1e-300)
    whole = g[:, live].sum(1)
    hits = ['L%d %+.4f (t %+.1f, %.0f%% +)' % (l, float(mean[l]), float(t[l]), 100 * float((g[:, l] > 0).double().mean()))
            for l in live if abs(float(t[l])) >= TMIN]
    n2, n3 = sum(abs(float(t[l])) > 2 for l in live), sum(abs(float(t[l])) > 3 for l in live)
    total, strong2, strong3 = total + len(live), strong2 + n2, strong3 + n3
    print('%-8s %6d %6d %6d %+12.1f   %s' % (family, len(live), n2, n3, float(whole.mean() / (whole.std() / N ** 0.5)), '; '.join(hits) or '-'))
print('all families: %d active scalars, %d with |t| > 2 (noise alone: %.1f), %d with |t| > 3 (noise alone: %.1f)' % (
    total, strong2, 0.0455 * total, strong3, 0.0027 * total))
