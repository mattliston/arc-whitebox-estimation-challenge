"""TAG opus_engineer a62 v2 (file written 2026-10-12 03:56Z) = parity_v1.py, also for torch result files that keep only the final layer's
prediction ('final_prediction', the sizing workers' records): then the prediction-change correlation and norm are given at layer 15 only,
and every layer keeps its MSE change. v1 header: (2026-10-11 21:5xZ): native-torch parity of a CHANGE on one network: the trainer's two torch result files (base and
variant, each with 'prediction' (16, 1024) and 'mse_layers') against two native gate_rdz_v1.py arms (their <out>_pred.npz keyed by seed and
their rows' mse_layers). Prints, per layer, the correlation of the two prediction changes and the native / torch change norm, and the
final-layer MSE change of each.
usage: parity_v1.py <torch base .json> <torch variant .json> <native base gate json> <native variant gate json> <seed>"""
import json
import sys
import numpy as np

tb, tv, nb, nv, seed = sys.argv[1:6]
t0, t1 = json.load(open(tb)), json.load(open(tv))
full = 'prediction' in t0 and 'prediction' in t1
key = 'prediction' if full else 'final_prediction'
p0, p1 = np.asarray(t0[key], dtype=float), np.asarray(t1[key], dtype=float)
if not full:   # final layer only: put it at row 15 of an otherwise-NaN (16, 1024) array
    p0, p1 = [np.vstack([np.full((15, x.size), np.nan), x[None, :]]) for x in (p0, p1)]
q0 = np.load(nb.replace('.json', '_pred.npz'))[seed].astype(float)
q1 = np.load(nv.replace('.json', '_pred.npz'))[seed].astype(float)
r0 = {str(x['seed']): x for x in json.load(open(nb))['rows']}[seed]
r1 = {str(x['seed']): x for x in json.load(open(nv))['rows']}[seed]
dt, dn = p1 - p0, q1 - q0
print('seed %s  final-layer MSE change: torch %+.3f%%  native %+.3f%%' % (
    seed, 100 * (t1['mse'] / t0['mse'] - 1), 100 * (r1['mse_final'] / r0['mse_final'] - 1)))
for l in range(16):
    a, b = dt[l], dn[l]
    na, nb_ = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
    c = float(a @ b) / (na * nb_) if na > 0 and nb_ > 0 else float('nan')
    print('L%-2d corr %+.3f  norm native/torch %6.3f  MSE change torch %+.3f%% native %+.3f%%' % (
        l, c, nb_ / na if na > 0 else float('nan'), 100 * (t1['mse_layers'][l] / t0['mse_layers'][l] - 1),
        100 * (r1['mse_layers'][l] / r0['mse_layers'][l] - 1)))
