#!/usr/bin/env python3
"""TAG g19b-teacher-force-v1: per-ingredient ceiling by teacher-forcing.
Override ONE ingredient (k4 | k3 | v | all3) at every layer's univariate
consumption with the saved MC truth, run the host forward, report the
endpoint (L16) and late-layer post-mean error energy vs baseline.
Usage: run_teacher_force.py <rank> <arm>   arm in {base,k4,k3,v,all3}"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")


def load_host(tag):
    spec = importlib.util.spec_from_file_location(tag, BASE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def main():
    rank = int(sys.argv[1])
    arm = sys.argv[2]
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    mc_v = d["mc_pre_v"]
    mc_k3 = d["mc_pre_k3"]
    mc_k4 = d["mc_pre_k4"]
    mc_post, mc_se = d["mc_post"], d["mc_se"]

    mod = load_host(f"tf_{rank}_{arm}")
    fnp = mod.fnp
    counter = {"i": 0}
    orig = mod._univariate_edgeworth

    def override(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2,
                 sigma3, inverse_context=None):
        li = counter["i"] + 1  # tap i covers array layer index i+1
        counter["i"] += 1
        if li < 16 and arm != "base":
            if arm in ("k4", "all3"):
                k4 = fnp.asarray(mc_k4[li], dtype=fnp.float64)
            if arm in ("k3", "all3"):
                k3 = fnp.asarray(mc_k3[li], dtype=fnp.float64)
            if arm in ("v", "all3"):
                s_true = np.sqrt(np.maximum(mc_v[li], 1e-30))
                m_carried = (np.asarray(alpha, dtype=np.float64).view(np.ndarray)
                             * np.asarray(sigma, dtype=np.float64).view(np.ndarray))
                sigma = fnp.asarray(s_true, dtype=fnp.float64)
                alpha = fnp.asarray(m_carried / s_true, dtype=fnp.float64)
                sigma2 = sigma * sigma
                sigma3 = sigma2 * sigma
                pdf = fnp.exp(-0.5 * alpha * alpha) * float(1 / np.sqrt(2 * np.pi))
                inverse_context = None
        return orig(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2,
                    sigma3, inverse_context)

    mod._univariate_edgeworth = override

    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                      allow_pickle=False)
    mlp = MLP(width=1024, depth=16,
              weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"tf-{rank}-{arm}")
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=600.0, quiet=True):
            host_v, _, _ = mod.predict_with_native_tape(mlp)
    post = np.asarray(host_v, dtype=np.float64)
    e16 = mc_post[15] - post[15]
    noise16 = float(np.sum(mc_se[15] ** 2))
    late = 0.0
    late_noise = 0.0
    for l in range(10, 16):
        late += float(np.sum((mc_post[l] - post[l]) ** 2))
        late_noise += float(np.sum(mc_se[l] ** 2))
    res = {"rank": rank, "arm": arm,
           "L16_err_energy_nc": float(np.sum(e16 ** 2)) - noise16,
           "L16_mse_nc": (float(np.sum(e16 ** 2)) - noise16) / 1024.0,
           "late_err_energy_nc": late - late_noise,
           "finite": bool(np.isfinite(post).all()),
           "taps": counter["i"]}
    print(json.dumps(res))
    out = HERE / f"TEACHER_FORCE_rank{rank:04d}_{arm}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
