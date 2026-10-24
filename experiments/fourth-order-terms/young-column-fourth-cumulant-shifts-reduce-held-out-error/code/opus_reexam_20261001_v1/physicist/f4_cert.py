#!/usr/bin/env python3
"""TAG opus-physicist F4 certificate: the drop-in module f4.py on the current torch card.
 1 f4.install(theta 0.4, age 2, cheap anchor) reproduces the t3_screen.py arm 'cp4=0.4,cp4anchor=2,cp4age=2' (saved prediction, if present)
 2 theta = 0 reproduces the unchanged card exactly
 3 backward pass: theta as a float64 leaf, loss = final-layer MSE, card run with per-layer checkpoints (as the trainer does);
   d loss / d theta must be finite and must match a central finite difference of two forward runs
usage: f4_cert.py <rank> <threads>"""
import sys, json, time
import numpy as np, torch
import pcard as pc
import f4
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); tp = pc.truth(rank); W = pc.weights(rank)
out = dict(TAG='opus-physicist F4 certificate', rank=rank)
final = lambda cd: pc.run(cd)
p0 = final(pc.build(W)); out['base_final_x1e9'] = round(float(pc.mse(p0, tp)[15]) * 1e9, 4)
p1 = final(f4.install(pc.build(W), torch, 0.0)); out['theta0_max_abs_diff'] = float(np.abs(p1 - p0).max())
p2 = final(f4.install(pc.build(W), torch, 0.4)); out['theta0.4_final_x1e9'] = round(float(pc.mse(p2, tp)[15]) * 1e9, 4)
ref = None
for tag in ('cp4c', 'v40a'):
    f = pc.HERE / f't3_pred_rank{rank:04d}_{tag}.npz'
    if f.exists(): ref = np.load(f)['on_0.4']; break
if ref is not None: out['vs_t3_screen_max_abs_diff'] = float(np.abs(p2 - ref).max())
# gradient with respect to theta
torch.set_grad_enabled(True)
th = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
cd = f4.install(pc.build(W), torch, th); t0 = time.time()
pred = cd.run(True); loss = ((pred[15].to(torch.float64) - torch.from_numpy(tp[15])) ** 2).mean(); loss.backward()
out['dloss_dtheta_x1e9'] = float(th.grad) * 1e9; out['backward_wall'] = round(time.time() - t0, 1)
torch.set_grad_enabled(False)
h = 0.02; lp = pc.mse(final(f4.install(pc.build(W), torch, 0.4 + h)), tp)[15]; lm = pc.mse(final(f4.install(pc.build(W), torch, 0.4 - h)), tp)[15]
out['finite_difference_x1e9'] = float((lp - lm) / (2 * h)) * 1e9
print(json.dumps(out, indent=1))
