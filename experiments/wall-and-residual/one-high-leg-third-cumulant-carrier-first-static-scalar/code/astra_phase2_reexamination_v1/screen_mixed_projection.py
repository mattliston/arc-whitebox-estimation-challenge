"""Unmetered representation certificate for high/high/unrestricted K3 sectors.

Unlike P^tensor3, retain all tensor terms with at least two projected legs:
sum_cyclic T(P,P,I) - 2 T(P,P,P). Its core needs n*k*k coefficients.
No change to the reference recursion; compare static next-layer diagonal
contractions against its full tensor and the independent stored truth.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kp

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
G19 = HERE.parent / "public_g19_attribution_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=35)
    ap.add_argument("--modes", default="16,32,64,128,256")
    ap.add_argument("--layers", default="3,7,11,14")
    ap.add_argument("--preserve", action="store_true")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    outpath = HERE / f"mixed_projection_r{args.rank}_{args.tag}.json"
    if outpath.exists():
        raise FileExistsError(outpath)
    modes = [int(x) for x in args.modes.split(",")]
    layers = [int(x) for x in args.layers.split(",")]
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
    W = np.load(STORE / Path(man[args.rank]["path"]).parent / "weights.npy")
    assert W.shape == (16, 1024, 1024)
    truth = np.load(G19 / f"mps_truth_rank{args.rank:04d}_P32M.npz")["mc_pre_k3"]
    mlp = MLP(input_dim=1024, hidden_dim=1024, output_dim=1024, num_layers=17, nonlin="relu")
    for l in range(16):
        mlp.Ws[l].weight.data = torch.as_tensor(np.asarray(W[l]).T.copy(), dtype=torch.float32)
    mlp.Ws[16].weight.data = torch.eye(1024)
    mlp = mlp.to(torch.float64)
    wick = [kp.WICK_COEF_D[name] for name in mlp.nonlin_names]
    K = kp.coerce_input({1: torch.zeros(1024, dtype=torch.float64), 2: torch.eye(1024, dtype=torch.float64)}, k_max=3, kind=kp.SIMPLE)
    results = []
    t0 = time.time()
    for l in range(max(layers)+1):
        K = kp.linear_kprop(K, mlp.Ws[l].weight, k_max=3, set_metric=mlp.init_scale[l], bias=None)
        K = kp.nonlin_kprop(K, nonlin_wick_coef=wick[l], k_max=3, kind=kp.SIMPLE, use_pK=True, factor=True)
        if l not in layers:
            continue
        C = K[2].to_tensor()
        ev, V = torch.linalg.eigh(C)
        Wn = mlp.Ws[l+1].weight
        full = [Wn @ A for A in K[3]._factors]
        d_full = torch.sum(full[0]*full[1]*full[2], dim=1)
        tr = torch.tensor(truth[l+1], dtype=torch.float64)
        def plane(F):
            a,b,c = F
            return ((a*b)@c.T + (a*c)@b.T + (b*c)@a.T)/3
        def plane_push_diagonal(P):
            return 3*torch.sum((Wn.square()@P)*Wn, dim=1) - 2*Wn.pow(3)@torch.diagonal(P)
        if args.preserve:
            plane_full = K[3].get_dslice((2,1)).clone()
            plane_full.diagonal().copy_(K[3].get_dslice((3,)))
        def rel(a,b):
            return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
        for k in modes:
            U = V[:,-k:]
            WU = Wn @ U
            high = [WU @ (U.T @ A) for A in K[3]._factors]
            d_high = torch.sum(high[0]*high[1]*high[2], dim=1)
            d_two = sum(torch.sum(full[a]*high[(a+1)%3]*high[(a+2)%3], dim=1) for a in range(3)) - 2*d_high
            r = {"layer": l, "k": k, "full_rel_truth": rel(d_full,tr),
                 "threehigh_rel_full": rel(d_high,d_full), "twohigh_rel_full": rel(d_two,d_full),
                 "twohigh_rel_truth": rel(d_two,tr), "threehigh_rel_truth": rel(d_high,tr),
                 "twohigh_scale_to_full": float(torch.dot(d_two,d_full)/torch.dot(d_two,d_two)),
                 "twohigh_rescaled_rel_full": rel(d_two*torch.dot(d_two,d_full)/torch.dot(d_two,d_two),d_full),
                 "variance_capture": float(ev[-k:].sum()/ev.sum())}
            if args.preserve:
                hp = [U @ (U.T @ A) for A in K[3]._factors]
                plane_high = plane(hp)
                plane_two = sum(plane([K[3]._factors[b] if b==a else hp[b] for b in range(3)]) for a in range(3))-2*plane_high
                union_three = d_high + plane_push_diagonal(plane_full-plane_high)
                union_two = d_two + plane_push_diagonal(plane_full-plane_two)
                r.update(threehigh_planes_rel_full=rel(union_three,d_full),
                         twohigh_planes_rel_full=rel(union_two,d_full),
                         twohigh_planes_rel_truth=rel(union_two,tr))
            results.append(r)
            print(json.dumps(r), flush=True)
    outpath.write_text(json.dumps({"args": vars(args), "wall_s": time.time()-t0, "results": results,
                                  "comparison": "static reference K3, 32M pre-k3 truth, unmetered"}, indent=2)+"\n")

if __name__ == "__main__":
    main()
