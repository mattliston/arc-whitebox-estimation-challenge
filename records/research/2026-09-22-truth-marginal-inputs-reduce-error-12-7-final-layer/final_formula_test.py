#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-13): accuracy of the card's FINAL-LAYER marginal formula nonlinear_scalar(mu, v, k3, k4) given EXACT
inputs (8M truths mc_pre_m/v/k3/k4 at layer 15) vs the post-activation truth mc_post[15], against the truth's own noise (mc_se).
Also the same at layers 8..14 for the mean (m) and the post variance (E[relu^2]-m^2, from aux if available).  usage: final_formula_test.py <cand> <rank>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
spec = importlib.util.spec_from_file_location("fft_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
tr = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P8M.npz")
truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
out = {}
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    for l in (8, 11, 13, 14, 15):
        mu, v, k3, k4 = (fnp.asarray(tr[k][l].astype(np.float64)) for k in ("mc_pre_m", "mc_pre_v", "mc_pre_k3", "mc_pre_k4"))
        m, aux = mod.nonlinear_scalar(mu, v, k3, k4)
        m = np.asarray(m.tolist(), dtype=np.float64)
        e = m - truth[l]; se = tr["mc_se"][l].astype(np.float64)
        # Gaussian-only (k3=k4=0) for scale
        mg, _ = mod.nonlinear_scalar(mu, v, fnp.zeros(1024), fnp.zeros(1024)); mg = np.asarray(mg.tolist(), dtype=np.float64)
        rec = dict(layer=l, mse_exact_inputs=float((e ** 2).mean()), truth_noise_floor=float((se ** 2).mean()), mse_gaussian_only=float(((mg - truth[l]) ** 2).mean()),
                   skew_rms=float(np.sqrt(np.mean((tr["mc_pre_k3"][l] / tr["mc_pre_v"][l] ** 1.5) ** 2))), kurt_rms=float(np.sqrt(np.mean((tr["mc_pre_k4"][l] / tr["mc_pre_v"][l] ** 2) ** 2))))
        out[l] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"final_formula_test_r{rank}.json", "w"), indent=1)
