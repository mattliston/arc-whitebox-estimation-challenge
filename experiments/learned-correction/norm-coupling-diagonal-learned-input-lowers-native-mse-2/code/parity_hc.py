"""TAG opus_engineer a42: the native head column against the trainer's torch column (hand-over O/trainer/candidates/head_col_s1:
torch_pred_rank<rank>_column_off / _on.npy, 16 x 1,024 means). Compared is the CHANGE the column makes, column on minus column off,
per layer: the correlation (cosine) between the native and the torch change, their rms sizes, the ratio; and the change of the
final-layer MSE on both sides (truth: the organizers' means stored in O/physicist/k4l_pred_rank<rank>_ora52.npz).
The torch card is the trainer's (second dials, constant regressor); a native base with other dials has other levels, so the changes
correlate less than on the same card.
usage: parity_hc.py <native session with the column> <native session without> [<hand-over directory>] [ranks, default 651,378]"""
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
O = E.parent
on, off = sys.argv[1:3]
src = O/'trainer'/'candidates'/(sys.argv[3] if len(sys.argv) > 3 else 'head_col_s1')
ranks = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '651,378').split(',')]
for rank in ranks:
    n1 = np.load(E/f'logs/quick_{on}_pred_{rank}.npy').astype(np.float64); n0 = np.load(E/f'logs/quick_{off}_pred_{rank}.npy').astype(np.float64)
    t1 = np.load(src/f'torch_pred_rank{rank:04d}_column_on.npy').astype(np.float64); t0 = np.load(src/f'torch_pred_rank{rank:04d}_column_off.npy').astype(np.float64)
    truth = np.load(O/f'physicist/k4l_pred_rank{rank:04d}_ora52.npz')['truth'].astype(np.float64)
    dn, dt = n1 - n0, t1 - t0
    print(f'network {rank}: {on} minus {off} (native) against column_on minus column_off (torch)')
    print('layer   correlation   native rms (1e-6)   torch rms (1e-6)   ratio')
    for l in range(1, 16):
        a, b = dn[l], dt[l]
        na, nb = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
        print(f'{l:5d}   {float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)):+11.3f}   {1e6 * na:13.2f}   {1e6 * nb:16.2f}   {na / nb if nb else float("nan"):7.2f}')
    mse = lambda p, l=15: float(np.mean((p[l] - truth[l]) ** 2))
    alls = lambda p: float(np.mean((p[1:] - truth[1:]) ** 2))
    a, b = dn[1:].ravel(), dt[1:].ravel()
    print(f'  all layers together: correlation {float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))):+.3f}, size ratio {float(np.linalg.norm(a) / np.linalg.norm(b)):.2f}')
    print(f'  final-layer MSE: native {1e9 * mse(n0):.4f} -> {1e9 * mse(n1):.4f} e-9 ({100 * (mse(n1) / mse(n0) - 1):+.2f}%), torch {1e9 * mse(t0):.4f} -> {1e9 * mse(t1):.4f} e-9 ({100 * (mse(t1) / mse(t0) - 1):+.2f}%)')
    print(f'  layers 1-15:     native {100 * (alls(n1) / alls(n0) - 1):+.2f}%, torch {100 * (alls(t1) / alls(t0) - 1):+.2f}%;  native base against torch base at the final layer: {100 * (mse(n0) / mse(t0) - 1):+.2f}%')
