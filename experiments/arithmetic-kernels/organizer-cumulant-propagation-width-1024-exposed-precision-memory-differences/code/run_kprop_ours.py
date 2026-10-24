#!/usr/bin/env python3
"""TAG g22-run-kprop (T-128): run the ORGANIZERS' cumulant propagation (mlp_kprop, arXiv 2605.05179) on OUR phase-2 nets and score it
against the organizers' 1e9 truth, with their own FLOP counter. Conventions: our weights W[l] (in,out) with pre = post @ W, so
nn.Linear.weight = W[l].T; 16 linear+ReLU layers, then a 17th identity linear layer (no activation) so the kprop output = relu(z16) = mc_post[15].
  <kprop .venv python> run_kprop_ours.py <rank> <k_max> <factor 0/1> <kind SIMPLE|AUGMENT> [device]"""
import sys, time, json, math
from pathlib import Path
import numpy as np, torch
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kp
from mlp_kprop.flop_utils import ExtendedFlopCounterMode

HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank, k_max, factor, kind = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])), sys.argv[4]
dev = torch.device(sys.argv[5] if len(sys.argv) > 5 else ("mps" if torch.backends.mps.is_available() else "cpu"))
dtype = torch.float32 if dev.type == "mps" else torch.float64
torch.set_grad_enabled(False)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
truth = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
n = 1024
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=17, nonlin="relu")
for l in range(16):
    mlp.Ws[l].weight.data = torch.as_tensor(np.asarray(W[l]).T.copy(), dtype=torch.float32)
mlp.Ws[16].weight.data = torch.eye(n)
mlp = mlp.to(dev, dtype)
# sanity: the MLP forward on samples reproduces the truth to MC noise
x = torch.randn(4096, n, device=dev, dtype=dtype); h = mlp(x)
h = h if isinstance(h, torch.Tensor) else getattr(h, "out", None) if getattr(h, "out", None) is not None else list(h.__dict__.values())[0]
print(f"forward sanity: 4096-sample mean vs 1e9 truth mse {float(((h.mean(0).cpu().double().numpy() - truth[15]) ** 2).mean()):.2e} (MC noise ~{0.065/4096:.1e})", flush=True)
K_in = {1: torch.zeros(n, device=dev, dtype=dtype), 2: torch.eye(n, device=dev, dtype=dtype)}
KIND = getattr(kp, kind)
t0 = time.time()
with ExtendedFlopCounterMode(display=False) as fc:
    K_out = kp.mlp_kprop(mlp, K_in, k_max=k_max, factor=factor, kind=KIND)
flops = fc.get_total_flops()
est = K_out[1].to_tensor().cpu().double().numpy().reshape(-1)
mse = float(((est - truth[15]) ** 2).mean())
print(f"KPROP rank {rank} k_max={k_max} factor={factor} kind={kind} dev={dev.type}: final mse vs 1e9 truth {mse:.3e} | their FLOPs {flops:.3e} = {flops / 2 ** 41:.4f} of 2^41 | {time.time() - t0:.0f}s", flush=True)
np.save(HERE / f"kprop_out_rank{rank:04d}_k{k_max}_f{int(factor)}_{kind}.npy", est)
