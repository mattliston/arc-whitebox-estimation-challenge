"""TAG opus_engineer a14 F4 parity: the native port against the physicist's torch arms, on stored predictions only (no estimator run).
usage: parity_torch.py <native base quick name> <native F4 quick name> <torch tag> <torch base arm> <torch F4 arm> [<ranks>]
  native predictions: O/engineer/logs/quick_<name>_pred_<rank>.npy (16 x 1024, float32, clamped at 0)
  torch predictions:  O/physicist/t3_pred_rank<rank>_<tag>.npz, arms e.g. on_0 and on_0.4 (16 x 1024)
Per network: final-layer MSE of the four runs and the two changes; then, per layer, the size of the prediction change (rms over
the 1,024 units, native and torch) and the cosine between the two changes. A faithful port gives cosines near 1 and equal sizes;
what is left is the rounding-level difference between the native estimator and the torch card (kernels, float32 paths)."""
import json
import sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
O = E.parent
P2 = O.parent
base, cand, tag, tbase, tcand = sys.argv[1:6]
if len(sys.argv) > 6:
    ranks = [int(x) for x in sys.argv[6].split(',')]
else:
    ranks = [r['rank'] for r in json.load(open(E/f'logs/quick_{cand}.json'))['rows']]
print(f'native: {base} -> {cand}; torch ({tag}): {tbase} -> {tcand}')
print(f"{'rank':>5s} {'native base':>12s} {'native F4':>10s} {'change':>8s} | {'torch base':>11s} {'torch F4':>9s} {'change':>8s} | {'cos L15':>8s} {'size ratio':>10s} {'min cos L4-15':>13s}")
tot = np.zeros(4)
layer_cos, layer_ratio = [], []
for rank in ranks:
    truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    nb = np.load(E/f'logs/quick_{base}_pred_{rank}.npy').astype(np.float64)
    nf = np.load(E/f'logs/quick_{cand}_pred_{rank}.npy').astype(np.float64)
    z = np.load(O/f'physicist/t3_pred_rank{rank:04d}_{tag}.npz')
    tb = np.maximum(z[tbase], 0).astype(np.float32).astype(np.float64); tf = np.maximum(z[tcand], 0).astype(np.float32).astype(np.float64)
    mse = [float(np.mean((x[15] - truth[15]) ** 2)) for x in (nb, nf, tb, tf)]
    tot += mse
    dn, dt = nf - nb, tf - tb
    size_n, size_t = np.sqrt((dn ** 2).mean(1)), np.sqrt((dt ** 2).mean(1))
    cos = (dn * dt).sum(1) / np.maximum(np.sqrt((dn ** 2).sum(1) * (dt ** 2).sum(1)), 1e-300)
    layer_cos.append(cos); layer_ratio.append(size_n / np.maximum(size_t, 1e-300))
    print(f'{rank:5d} {mse[0]*1e9:12.4f} {mse[1]*1e9:10.4f} {100*(mse[1]/mse[0]-1):+7.2f}% | {mse[2]*1e9:11.4f} {mse[3]*1e9:9.4f} {100*(mse[3]/mse[2]-1):+7.2f}% |'
          f' {cos[15]:8.4f} {size_n[15]/size_t[15]:10.4f} {cos[4:].min():13.4f}')
print(f"pooled {tot[0]/len(ranks)*1e9:11.4f} {tot[1]/len(ranks)*1e9:10.4f} {100*(tot[1]/tot[0]-1):+7.2f}% | {tot[2]/len(ranks)*1e9:11.4f} {tot[3]/len(ranks)*1e9:9.4f} {100*(tot[3]/tot[2]-1):+7.2f}%")
layer_cos, layer_ratio = np.array(layer_cos), np.array(layer_ratio)
print('per layer, mean over networks:')
print(' layer   ' + ' '.join(f'{l:6d}' for l in range(16)))
print(' cosine  ' + ' '.join(f'{x:6.3f}' for x in layer_cos.mean(0)))
print(' size n/t' + ' '.join(f'{x:6.3f}' for x in layer_ratio.mean(0)))
