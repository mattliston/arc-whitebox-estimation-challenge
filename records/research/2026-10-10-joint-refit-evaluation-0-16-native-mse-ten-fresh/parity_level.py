"""TAG opus_engineer a48 (2026-10-10): native against torch when the trainer stores only the candidate's torch prediction (no torch base): per layer
the change of the native candidate from a native base (mean of the base sessions given) against the change of the torch candidate from the
same native base, as cosine and size ratio (the torch side then also holds the torch-native difference of the bases), the rms gap between the
native and the torch candidate, and the final-layer MSE (688-truth) of all.
usage: parity_level.py <rank> <native candidate session> <torch result json (field 'prediction', 16 x 1024)> <native base session> [...]"""
import json
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
P2 = E.parents[1]
rank = int(sys.argv[1])
nat = np.load(E/f'logs/quick_{sys.argv[2]}_pred_{rank}.npy').astype(np.float64)
tor = np.asarray(json.load(open(sys.argv[3]))['prediction'], dtype=np.float64)
bases = [np.load(E/f'logs/quick_{b}_pred_{rank}.npy').astype(np.float64) for b in sys.argv[4:]]
base = sum(bases) / len(bases)
truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
assert tor.shape == nat.shape == (16, 1024), (tor.shape, nat.shape)
print(f'network {rank}: native {sys.argv[2]}, torch {Path(sys.argv[3]).name}, native base = mean of {", ".join(sys.argv[4:])}')
print('layer   cosine   native change rms (1e-6)   torch change rms (1e-6)   ratio   native-torch gap rms (1e-6)   base draws gap rms (1e-6)')
for l in range(16):
    a, b = nat[l] - base[l], tor[l] - base[l]
    na, nb = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
    gap = float(np.sqrt(np.mean((nat[l] - tor[l]) ** 2)))
    draws = float(np.sqrt(np.mean((bases[0][l] - bases[-1][l]) ** 2))) if len(bases) > 1 else float('nan')
    print(f'{l:5d}   {cos:+.4f}   {1e6 * na:22.3f}   {1e6 * nb:22.3f}   {na / max(nb, 1e-300):5.3f}   {1e6 * gap:24.3f}   {1e6 * draws:22.3f}')
mse = lambda y: float(np.mean((y[15] - truth) ** 2))
print('final-layer MSE e-9: native candidate %.5f, torch candidate %.5f, native base draws %s' % (1e9 * mse(nat), 1e9 * mse(tor), ', '.join('%.5f' % (1e9 * mse(b)) for b in bases)))
