"""TAG opus_engineer a53 (2026-10-10, lead 20:25Z approximation audit): the native-torch gap at the final layer, split.
With e = torch - truth and d = native - torch (final layer, 1024 units):
  MSE_native - MSE_torch = mean(d^2) + 2 mean(e d)     (noise part, always >= 0; correlated part, either sign)
For each pair (native session, torch json) on one network prints both parts as % of MSE_torch, and the rms of d per layer.
usage: gap_decomp.py <rank> <native session> <torch result json> [<native session> <torch result json> ...]"""
import json
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
P2 = E.parents[1]
rank = int(sys.argv[1])
truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
for nat_name, tor_path in zip(sys.argv[2::2], sys.argv[3::2]):
    nat = np.load(E/f'logs/quick_{nat_name}_pred_{rank}.npy').astype(np.float64)
    tor = np.asarray(json.load(open(tor_path))['prediction'], dtype=np.float64)
    e, d = tor[15] - truth, nat[15] - tor[15]
    mt, mn = float(np.mean(e * e)), float(np.mean((e + d) ** 2))
    noise, corr = float(np.mean(d * d)), 2 * float(np.mean(e * d))
    rms = ' '.join('%.3g' % (1e6 * float(np.sqrt(np.mean((nat[l] - tor[l]) ** 2)))) for l in range(16))
    print(f'{rank} {nat_name} vs {Path(tor_path).stem}: MSE e-9 native {1e9 * mn:.5f} torch {1e9 * mt:.5f}: gap {100 * (mn / mt - 1):+.3f}% = noise {100 * noise / mt:+.3f}% + correlated {100 * corr / mt:+.3f}%')
    print(f'   rms(native - torch) per layer (1e-6): {rms}')
