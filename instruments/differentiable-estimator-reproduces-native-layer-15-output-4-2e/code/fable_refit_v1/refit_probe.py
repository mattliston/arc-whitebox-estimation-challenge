#!/usr/bin/env python3
"""TAG fable-refit (RF-0): probe of the research torch card (model_bank_research = the 331824 physics): fidelity vs the native
prediction on one net, forward timing at N threads, and fwd+bwd timing with the physics coefficients (cstar 15x4, A 64x4) trainable.
usage: refit_probe.py <rank> <threads>"""
import sys, os, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
rank, threads = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(threads)
P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); assert w.shape == (16, 1024, 1024) and w.dtype == np.float32
truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64))
p, a, c, e = model.torch_payload(torch, np, False)
t0 = time.time()
card = model.Card(torch, w, p, a, c, e)
with torch.no_grad():
    pred = card.run(False)
t1 = time.time()
mse = float(((pred[15].to(torch.float64) - truth[15]) ** 2).mean())
nat = P2 / "fable_oldtier_economics_v1" / f"bank_r{rank}" / "pred.npy"
fid = float(np.sqrt(((pred[15].numpy().astype(np.float64) - np.load(nat)[15]) ** 2).mean())) if nat.exists() else None
print(json.dumps(dict(stage="forward", rank=rank, threads=threads, wall=round(t1 - t0, 1), mse=mse, rms_vs_native=fid)), flush=True)
# fwd+bwd with physics trainable
a.requires_grad_(True); c.requires_grad_(True)
t0 = time.time()
card = model.Card(torch, w, p, a, c, e)
pred = card.run(True)
loss = ((pred[15].to(torch.float64) - truth[15]) ** 2).mean()
loss.backward()
t1 = time.time()
print(json.dumps(dict(stage="fwd_bwd", rank=rank, threads=threads, wall=round(t1 - t0, 1), loss=float(loss),
                      grad_norm_c=float(c.grad.norm()), grad_norm_a=float(a.grad.norm()), peak_rss_gb=round(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 2**30, 2))), flush=True)
