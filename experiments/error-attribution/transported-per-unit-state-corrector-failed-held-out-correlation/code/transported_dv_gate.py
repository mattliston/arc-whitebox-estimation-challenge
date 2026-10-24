#!/usr/bin/env python3
"""TAG g29-transported-dv-gate (T-90): the decisive gate for a TRANSPORTED per-unit state corrector on the derived host
(corpus review route 1 / design B).  Per net (8 panel nets, derived host = in-card chain + residual carry, feature dumps
feat_derived_retro-{rank}.npz from RETRO_DUMP_FEAT), targets dv_L = truth pre v (8M; 32M for 35/100) - derived pre v,
likewise dk3, dk4.  Ridge, leave-one-net-out, per layer L in 4..15, three feature sets:
  local      : the card's own per-unit state at L (pre m, v, carried k3, k4, CP3 diagonal, star diagonal, K21 row/col norms,
               K22/K31 row norms, weight-in norm) + their ratios
  +oracle_T  : local + the TRUTH previous-layer dv transported through W^2 (three gate weightings) = the ceiling of ANY
               transported state (teacher-forced)
  +auto_T    : local + the PREDICTED previous-layer dv transported (autonomous rollout from L3, held-out net)
Pass bar (corpus review): held-out dv corr >= 0.65 at L10-15 on all nets for the autonomous form; the oracle form bounds it.
Usage: transported_dv_gate.py"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
RANKS = [35, 100, 150, 200, 250, 300, 400, 500]
LAYERS = list(range(3, 16))


def weights_of(rank):
    man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]
    return np.load((STORE / e["path"]).parent / "weights.npy", mmap_mode="r")


def load_net(rank):
    f = np.load(HERE / f"feat_derived_retro-{rank}.npz")
    tf = G19 / (f"mps_truth_rank{rank:04d}_P32M.npz" if rank in (35, 100) else f"mps_truth_rank{rank:04d}_P8M.npz")
    T = np.load(tf); W = weights_of(rank)
    out = {}
    for L in LAYERS:
        if f"pre_v_{L}" not in f.files:
            continue
        m = f[f"pre_m_{L}"].astype(np.float64); v = f[f"pre_v_{L}"].astype(np.float64)
        k3 = f[f"k3_carried_{L}"].astype(np.float64); k4 = f[f"pre_k4_{L}"].astype(np.float64)
        sd = np.sqrt(np.maximum(v, 1e-30)); a = m / sd
        loc = np.stack([m, v, k3, k4, a, k3 / sd ** 3, k4 / sd ** 4, f[f"cp3_diag_{L}"], f[f"star_diag_{L}"], f[f"K21_rownorm_{L}"],
                        f[f"K21_colnorm_{L}"], f[f"K22_rownorm_{L}"], f[f"K31_rownorm_{L}"], f[f"w_in_norm_{L}"],
                        f[f"cp3_energy_{L}"], f[f"cp3_diag_{L}"] / sd ** 3, f[f"K21_rownorm_{L}"] / sd ** 3], 1)
        out[L] = {"X": loc, "dv": T["mc_pre_v"][L].astype(np.float64) - v, "v": v,
                  "dk3": T["mc_pre_k3"][L].astype(np.float64) - k3, "dk4": T["mc_pre_k4"][L].astype(np.float64) - k4,
                  "gate": 0.5 * (1 + np.vectorize(__import__("math").erf)(a / np.sqrt(2))), "W": np.asarray(W[L], dtype=np.float64)}
    return out


def transported(prev_dv, gate_prev, W):
    """three W^2 transports of a previous-layer per-unit variance error into this layer's pre units"""
    W2 = W * W
    return np.stack([W2.T @ prev_dv, W2.T @ (prev_dv * gate_prev), W2.T @ (prev_dv * gate_prev ** 2)], 1)


def ridge_fit(X, y, lam=1e-3):
    mu = X.mean(0); sd = X.std(0) + 1e-30; Xs = (X - mu) / sd; ym = y.mean()
    w = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(X.shape[1]), Xs.T @ (y - ym))
    return mu, sd, w, ym


def ridge_pred(model, X):
    mu, sd, w, ym = model
    return ((X - mu) / sd) @ w + ym


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300))


def main():
    nets = {r: load_net(r) for r in RANKS if (HERE / f"feat_derived_retro-{r}.npz").exists()}
    print("nets:", list(nets))
    res = {}
    for target in ("dv", "dk3", "dk4"):
        res[target] = {}
        for L in LAYERS:
            if any(L not in nets[r] for r in nets) or any((L - 1) not in nets[r] for r in nets):
                continue
            rows = {"local": [], "oracle_T": [], "auto_T": []}
            for held in nets:
                train = [r for r in nets if r != held]
                # local
                Xtr = np.concatenate([nets[r][L]["X"] for r in train]); ytr = np.concatenate([nets[r][L][target] for r in train])
                mdl = ridge_fit(Xtr, ytr); rows["local"].append(corr(ridge_pred(mdl, nets[held][L]["X"]), nets[held][L][target]))
                if target == "dv":
                    # oracle transported: truth previous-layer dv
                    Xtr2 = np.concatenate([np.concatenate([nets[r][L]["X"], transported(nets[r][L - 1]["dv"], nets[r][L - 1]["gate"], nets[r][L]["W"])], 1) for r in train])
                    mdl2 = ridge_fit(Xtr2, ytr)
                    Xh2 = np.concatenate([nets[held][L]["X"], transported(nets[held][L - 1]["dv"], nets[held][L - 1]["gate"], nets[held][L]["W"])], 1)
                    rows["oracle_T"].append(corr(ridge_pred(mdl2, Xh2), nets[held][L]["dv"]))
                    # autonomous: previous-layer dv PREDICTED by the same model rolled out from L0 = 3 on the held-out net
                    prev = np.zeros(1024)
                    for Lp in range(3, L):
                        if Lp == 3:
                            mdl_p = ridge_fit(np.concatenate([nets[r][Lp]["X"] for r in train]), np.concatenate([nets[r][Lp]["dv"] for r in train]))
                            prev = ridge_pred(mdl_p, nets[held][Lp]["X"])
                        else:
                            Xtr_p = np.concatenate([np.concatenate([nets[r][Lp]["X"], transported(nets[r][Lp - 1]["dv"], nets[r][Lp - 1]["gate"], nets[r][Lp]["W"])], 1) for r in train])
                            mdl_p = ridge_fit(Xtr_p, np.concatenate([nets[r][Lp]["dv"] for r in train]))
                            prev = ridge_pred(mdl_p, np.concatenate([nets[held][Lp]["X"], transported(prev, nets[held][Lp - 1]["gate"], nets[held][Lp]["W"])], 1))
                    Xh3 = np.concatenate([nets[held][L]["X"], transported(prev, nets[held][L - 1]["gate"], nets[held][L]["W"])], 1)
                    rows["auto_T"].append(corr(ridge_pred(mdl2, Xh3), nets[held][L]["dv"]))
            res[target][L] = {k: (round(float(np.mean(v)), 3), round(float(np.min(v)), 3)) for k, v in rows.items() if v}
            print(target, f"L{L}", " ".join(f"{k} mean {a:+.3f} min {b:+.3f}" for k, (a, b) in res[target][L].items()), flush=True)
    (HERE / "TRANSPORTED_DV_GATE.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
