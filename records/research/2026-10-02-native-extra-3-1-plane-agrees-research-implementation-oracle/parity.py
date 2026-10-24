"""TAG opus_engineer a34: native change against torch change for the oracle K31 plane, layer by layer.
native: <native arm .npy> - <native base .npy>;  torch: <arm> - S38 of O/physicist/k4l_pred_rank<rank>_ora.npz.
Per layer: rms of each change, their cosine, the projection of the native change on the torch one, and the rms of what is left.
usage: parity.py <rank> <native base .npy> <native arm .npy> <torch arm name, e.g. K31f> [<tag of the torch file, default ora>]"""
import sys
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[2]
P2 = O.parent
rank, base, arm, name = int(sys.argv[1]), np.load(sys.argv[2]).astype(np.float64), np.load(sys.argv[3]).astype(np.float64), sys.argv[4]
z = np.load(O/'physicist'/f'k4l_pred_rank{rank:04d}_{sys.argv[5] if len(sys.argv) > 5 else "ora"}.npz')
truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
dn, dt = arm - base, z[name] - z['S38']
print('layer | native change rms, torch change rms (1e-6) | cosine | projection | left over rms (1e-6) | MSE change native %, torch %')
for l in range(16):
    a, b = dn[l], dt[l]
    if not b.any() and not a.any():
        continue
    proj = float(a @ b / max(b @ b, 1e-300))
    mse = lambda p: float(((p[l] - truth[l]) ** 2).mean())
    print('%5d | %8.2f %8.2f | %.4f | %.3f | %6.2f | %+7.2f %+7.2f' % (l, 1e6 * np.sqrt((a ** 2).mean()), 1e6 * np.sqrt((b ** 2).mean()), float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)), proj,
          1e6 * np.sqrt(((a - proj * b) ** 2).mean()), 100 * (mse(arm) / mse(base) - 1), 100 * (mse(z[name]) / mse(z['S38']) - 1)))
print('final-layer MSE (1e-9): native base %.4f, native arm %.4f; torch base %.4f, torch arm %.4f; native base against torch base: rms difference %.2fe-6' % (
    1e9 * ((base[15] - truth[15]) ** 2).mean(), 1e9 * ((arm[15] - truth[15]) ** 2).mean(), 1e9 * ((z['S38'][15] - truth[15]) ** 2).mean(), 1e9 * ((z[name][15] - truth[15]) ** 2).mean(),
    1e6 * np.sqrt(((base[15] - z['S38'][15]) ** 2).mean())))
