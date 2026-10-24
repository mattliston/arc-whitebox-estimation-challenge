"""TAG opus-trainer (2026-10-01): headroom probe for the learned mean correction at the FINAL layer (the layer the score is on).
The card's final-layer correction is delta = s x (state . head[15] + head_b[15]) with s = sigma x rms x gain per unit and
state = the 64 outputs of the message network. Given the captures of probe_capture.py this solves, by exact least squares on
training networks, what the best correction of that form would be, and richer ones from the same inputs, and scores them on
networks that were not used for the fit:
  A  readout from the 64 state outputs (the 65 numbers the card already has: is Adam's solution the least-squares one?)
  B  readout from the 128 units of the message network's second hidden layer (re-solves its last linear layer for layer 15)
  C  readout from those 128 plus the 126 raw inputs of the message network
  D  as C plus all pairwise products of the 30 standardised unit features (a crude test for missing non-linear structure)
Each row: change of the pooled final-layer MSE against the card as it is, on the fit networks (in-sample), on held-out training
networks and on VAL-A. VAL-B is not read. "no correction" is the card with the final-layer correction removed.
With a list of fit sizes the table is repeated per size (a learning curve: does a richer readout start to generalise with more
networks?); the held-out training networks are then always the last 100 captured ones.
Only the final-layer correction is probed: corrections at earlier layers act on the final layer through the propagation and are
not covered here.
usage: python -B probe_fit.py <capture dir> <comma-separated training ranks> [<fit size>,<fit size>,...]"""
import sys
from pathlib import Path

import numpy as np

from otr_common import splits

DIR = Path(sys.argv[1])
train = [int(x) for x in sys.argv[2].split(',')]
train = [rank for rank in train if (DIR / ('probe_r%04d.npz' % rank)).exists()]
if len(sys.argv) > 3:
    held, sizes = train[-100:], [int(x) for x in sys.argv[3].split(',')]
    assert max(sizes) <= len(train) - 100
else:
    held, sizes = train[len(train) * 2 // 3:], [len(train) * 2 // 3]
val_a = [rank for rank in splits()['val_a'] if (DIR / ('probe_r%04d.npz' % rank)).exists()]
KINDS = (('A', 'A  least-squares readout from the 64 state outputs'), ('B', 'B  least-squares readout from the 128 hidden units'),
         ('C', 'C  from 128 hidden units + 126 inputs'), ('D', 'D  as C + 465 products of the 30 unit features'))


data = {}


def load(rank):
    if rank not in data:
        with np.load(DIR / ('probe_r%04d.npz' % rank)) as saved:
            data[rank] = {key: saved[key].astype(np.float64) for key in ('H', 'h2', 'state', 'scale', 'delta', 'host', 'truth')}
    return data[rank]


def design(row, kind):
    parts = dict(A=[row['state']], B=[row['h2']], C=[row['h2'], row['H']], D=[row['h2'], row['H']])[kind]
    if kind == 'D':
        x = row['H'][:, :30]
        i, j = np.triu_indices(30)
        parts = parts + [x[:, i] * x[:, j]]
    features = np.concatenate(parts + [np.ones((1024, 1))], axis=1)
    return features * row['scale'][:, None]


def mse(row, delta):
    return float(((np.maximum(row['host'] + delta, 0) - row['truth']) ** 2).mean())


def cell(ranks, kind, weights):
    new = np.array([mse(load(rank), design(load(rank), kind) @ weights) for rank in ranks])   # load() is cached
    old = np.array([now[rank] for rank in ranks])
    change = 100 * (new / old - 1)
    return '%+7.3f%% (se %.3f, better on %d/%d)' % (100 * (new.mean() / old.mean() - 1), change.std(ddof=1) / len(change) ** 0.5,
                                                   (change < 0).sum(), len(change))


now, none = {}, {}
for rank in train + val_a:
    row = load(rank)
    now[rank], none[rank] = mse(row, row['delta']), mse(row, 0.0)
print('pooled final-layer MSE of the card as it is: held-out training (%d) %.5e   VAL-A (%d) %.5e' % (
    len(held), np.mean([now[rank] for rank in held]), len(val_a), np.mean([now[rank] for rank in val_a])))
print('no correction at the final layer: held-out training %+.3f%%   VAL-A %+.3f%%' % tuple(
    100 * (np.mean([none[rank] for rank in ranks]) / np.mean([now[rank] for rank in ranks]) - 1) for ranks in (held, val_a)))
grams = {kind: [0.0, 0.0] for kind, _ in KINDS}
done = 0
for size in sizes:
    for rank in train[done:size]:
        row = load(rank)
        for kind, _ in KINDS:
            X = design(row, kind)
            grams[kind][0] = grams[kind][0] + X.T @ X
            grams[kind][1] = grams[kind][1] + X.T @ (row['truth'] - row['host'])
    done = size
    print('--- fitted on %d training networks: change on the fit networks | on %d held-out training networks | on VAL-A' % (
        size, len(held)))
    for kind, label in KINDS:
        gram, rhs = grams[kind]
        weights = np.linalg.solve(gram + 1e-9 * np.trace(gram) / len(gram) * np.eye(len(gram)), rhs)
        print('%-62s' % ('%s [%d numbers]' % (label, len(weights))) + '   '.join(
            cell(ranks, kind, weights) for ranks in (train[:size], held, val_a)), flush=True)
own = []
for rank in val_a:   # each VAL-A network fitted on itself: what a network-specific readout could reach (in-sample, an upper bound)
    row = load(rank)
    X = design(row, 'A')
    own.append(mse(row, X @ np.linalg.lstsq(X, row['truth'] - row['host'], rcond=None)[0]))
print('A fitted on each VAL-A network itself (in-sample; pure noise alone would give %+.1f%%): %+.3f%%' % (
    -100 * 65 / 1024, 100 * (np.mean(own) / np.mean([now[rank] for rank in val_a]) - 1)))
