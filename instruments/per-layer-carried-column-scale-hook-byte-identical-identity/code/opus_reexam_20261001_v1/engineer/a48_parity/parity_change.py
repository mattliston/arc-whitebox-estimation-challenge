"""TAG opus_engineer a48 (2026-10-10): native against torch for a candidate, by the CHANGE each makes to its own base on the same network
(native: two quick sessions' stored predictions; torch: two prediction files of the trainer's forwards), so that the two bases' own
difference (the native rounding draw, about 0.5% of a network's final-layer MSE) drops out: per layer the cosine between the native and
the torch change and the ratio of their sizes, and the final-layer MSE (against the 688-truth) of all four.
A torch file is .npy ((16, 1024) or (1024,) = the final layer) or .npz[key].
usage: parity_change.py <rank> <native candidate session> <native base session> <torch candidate file[:key]> <torch base file[:key]>"""
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
P2 = E.parents[1]
rank = int(sys.argv[1])


def torch_file(spec):
    path, _, key = spec.partition(':')
    x = np.load(path, allow_pickle=False)
    x = x[key] if key else x
    return np.asarray(x, dtype=np.float64)


nat1 = np.load(E/f'logs/quick_{sys.argv[2]}_pred_{rank}.npy').astype(np.float64)
nat0 = np.load(E/f'logs/quick_{sys.argv[3]}_pred_{rank}.npy').astype(np.float64)
tor1, tor0 = torch_file(sys.argv[4]), torch_file(sys.argv[5])
if tor1.ndim == 1:
    nat1, nat0, layers = nat1[15:16], nat0[15:16], [15]
    tor1, tor0 = tor1[None, :], tor0[None, :]
else:
    layers = list(range(16))
truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
print(f'network {rank}: native {sys.argv[2]} - {sys.argv[3]}; torch {sys.argv[4]} - {sys.argv[5]}')
print('layer   cosine   native change rms (1e-6)   torch change rms (1e-6)   ratio   native - torch candidate rms (1e-6)')
for i, l in enumerate(layers):
    a, b = nat1[i] - nat0[i], tor1[i] - tor0[i]
    na, nb = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
    gap = float(np.sqrt(np.mean((nat1[i] - tor1[i]) ** 2)))
    print(f'{l:5d}   {cos:+.4f}   {1e6 * na:22.3f}   {1e6 * nb:22.3f}   {na / max(nb, 1e-300):5.3f}   {1e6 * gap:12.3f}')
mse = lambda y: float(np.mean((y - truth) ** 2))
n1, n0, t1, t0 = mse(nat1[-1]), mse(nat0[-1]), mse(tor1[-1]), mse(tor0[-1])
print(f'final-layer MSE e-9: native {1e9 * n1:.5f} vs base {1e9 * n0:.5f} ({100 * (n1 / n0 - 1):+.3f}%); torch {1e9 * t1:.5f} vs base {1e9 * t0:.5f} ({100 * (t1 / t0 - 1):+.3f}%)')
