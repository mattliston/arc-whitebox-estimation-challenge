#!/usr/bin/env python3
"""TAG g22-kprop-rank (T-129): the organizers' hint tested inside THEIR kprop (K=3 factorized, float64): after every nonlinear step, project
the all-distinct third-cumulant FACTORS (CP form, n x r) onto the top-k eigenvectors of the current covariance (the repeated-index slices,
i.e. the plane/diagonal, stay exact). Reports final MSE vs the 1e9 truth for each k. Zero billed cost.  usage: run_kprop_rank.py <rank> <ks,comma>"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kp
from mlp_kprop.factor_k3 import FactoredTensor
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank, ks = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')]
torch.set_grad_enabled(False); n = 1024; dt = torch.float64
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]; W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
truth = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=17, nonlin="relu")
for l in range(16): mlp.Ws[l].weight.data = torch.as_tensor(np.asarray(W[l]).T.copy(), dtype=torch.float32)
mlp.Ws[16].weight.data = torch.eye(n); mlp = mlp.to(dt)

def kprop_with_projection(k):
    """The mlp_kprop loop (kind=SIMPLE, factor=True, k_max=3) with a factor projection after each nonlinear step."""
    k_max = 3; kind = kp.SIMPLE
    wick = [kp.WICK_COEF_D[nl] for nl in mlp.nonlin_names]; scale = mlp.init_scale
    K = kp.coerce_input({1: torch.zeros(n, dtype=dt), 2: torch.eye(n, dtype=dt)}, k_max=k_max, kind=kind)
    ranks = []
    for l, Wm in enumerate(mlp.Ws):
        Wt = Wm.weight
        if l == len(mlp.nonlins):
            K = kp.linear_kprop(K, Wt, k_max=k_max, set_metric=None, bias=None); break
        K = kp.linear_kprop(K, Wt, k_max=k_max, set_metric=scale[l], bias=None)
        K = kp.nonlin_kprop(K, nonlin_wick_coef=wick[l], k_max=k_max, kind=kind, use_pK=True, factor=True)
        if k < n and isinstance(K.get(3), FactoredTensor):
            C = K[2].to_tensor() if hasattr(K[2], "to_tensor") else K[2].core
            ev, V = torch.linalg.eigh(C); V = V[:, -k:]                       # top-k covariance directions
            F = K[3]._factors; ranks.append(F[0].shape[1])
            K[3]._factors = [V @ (V.T @ A) for A in F]
    return K[1].to_tensor().reshape(-1).numpy(), ranks

for k in ks:
    t0 = time.time(); est, ranks = kprop_with_projection(k)
    mse = float(((est - truth[15]) ** 2).mean())
    print(f"rank {rank} K=3 factorized f64, factors projected to top-{k} covariance directions each layer: final mse {mse:.3e} | factor ranks by layer {ranks[:3]}..{ranks[-1:] if ranks else ''} | {time.time()-t0:.0f}s", flush=True)
