#!/usr/bin/env python3
"""TAG opus-physicist T67b: JUDGE the handed-over column-17 weights (t67_U_17.npy, fixed, as the trainer and the engineer received them) on every judging network of
runM_val50 that has a layer-15 dump (T66), and write the PARITY file for the native build (lead 04:15Z item 3): per network the standardised column z17 (kept rows),
the change of the final prediction kappa (delta(U) - delta(0)), S42's prediction and the new one. Model and certificate as T67.
-> t67b_parity_c17.npz (keys z17_<rank>, dpred_<rank>, pred_<rank>, new_<rank>) ; usage: t67b_judge17.py"""
import math, re
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2); t = torch; f64 = t.float64
HERE = Path(__file__).resolve().parent; O = HERE.parent
S = torch.load(O / 'trainer/cert/fold/c3q4zd6_optimum.pt', map_location='cpu', weights_only=True)['arrays']
W0, b0, W2, b2, W4, b4 = (S['p.' + k].to(f64) for k in ('enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b'))
HD, HB, RMS, GN = S['p.head'][15].to(f64), S['p.head_b'][15].to(f64), S['p.rms'][15].to(f64), S['p.gain'][15].to(f64)
silu = lambda z: z / (1 + t.exp(-z))
U = t.tensor(np.load(HERE / 't67_U_17.npy'), dtype=f64).reshape(1, 128)
VAL = [r for r in (int(x) for x in re.findall(r'\d+', (HERE / 't66_val50.txt').read_text())) if (HERE / f'l15dump_rank{r:04d}.npz').exists()]
out = {}; base, new, cert = [], [], []
for r in VAL:
    D = np.load(HERE / f'l15dump_rank{r:04d}.npz'); kept = D['kept'] > 0.5
    x = D['c17']; z = np.zeros_like(x); sd = x[kept].std(ddof=1); z[kept] = np.clip((x[kept] - x[kept].mean()) / sd, -8, 8)
    H = t.tensor(D['H'], dtype=f64); scale = t.tensor(D['sigma'], dtype=f64) * RMS * (1 + 0.25 * t.tanh(GN)); pre = H @ W0.T + b0
    head = lambda P: scale * ((silu(silu(P) @ W2.T + b2) @ W4.T + b4) @ HD + HB)
    d0 = head(pre); d1 = head(pre + t.tensor(z, dtype=f64)[:, None] @ U); cert.append(float(np.abs(d0.numpy() - D['delta']).max() / np.abs(D['delta']).max()))
    kappa = float(np.median((D['m'] - D['host']) / np.where(np.abs(D['delta']) > 0, D['delta'], np.nan)[np.isfinite((D['m'] - D['host']) / np.where(np.abs(D['delta']) > 0, D['delta'], np.nan))]))
    dp = kappa * (d1 - d0).numpy(); p0 = D['pred']; p1 = p0 + dp
    base.append(np.mean((p0 - D['truth']) ** 2)); new.append(np.mean((p1 - D['truth']) ** 2))
    out.update({f'z17_{r}': z, f'dpred_{r}': dp, f'pred_{r}': p0, f'new_{r}': p1})
b, nw = np.array(base), np.array(new)
np.savez(HERE / 't67b_parity_c17.npz', ranks=np.array(VAL), **out)
print('T67b column 17 at layer 15 with the handed-over weights (t67_U_17.npy, fixed), %d judging networks of runM_val50: pooled final-layer MSE %.4e -> %.4e = %+.3f%% '
      '(mean of per-network log ratios %+.3f%%, se %.3f%%, %d of %d lower); delta certificate max rel %.1e; parity file t67b_parity_c17.npz' % (
          len(VAL), b.mean(), nw.mean(), 100 * (nw.mean() / b.mean() - 1), 100 * np.mean(np.log(nw / b)), 100 * np.std(np.log(nw / b), ddof=1) / math.sqrt(len(b)),
          int((nw < b).sum()), len(b), max(cert)))
