#!/usr/bin/env python3
"""TAG g19d-force-k21post-v1: teacher-force the TRUE post-activation K21 into
_reset_stars (the star-compression input) at chosen layers; tap carried
pre-K21 at every _full_relu_reset; score endpoint means vs MC truth.
Arms: base | one:<l> | all. Alignment: _reset_stars call j <-> array layer j
(kt tap receipt). Usage: run_force_k21post.py <rank> <arm>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
sys.path.insert(0, str(HERE))
from run_attribution import load_host


def main():
    rank = int(sys.argv[1])
    arm = sys.argv[2]
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    mc_post, mc_se = d["mc_post"], d["mc_se"]
    Pr = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    post_files = [HERE / f"mps_planes_rank{rank:04d}_post.npz",
                  HERE / f"mps_planes_rank{rank:04d}_post2.npz",
                  HERE / f"mps_planes_rank{rank:04d}_post3.npz"]
    truth_post = {}
    if arm == "tensor":
        tp = np.load(HERE / f"mps_planes_rank{rank:04d}.npz".replace(
            "mps_planes", "tensor_planes"))
        for k in tp.keys():
            if k.startswith("post_"):
                truth_post[int(k.split("_")[1])] = tp[k]
    else:
        for pf in post_files:
            if pf.exists():
                z = np.load(pf)
                for k in z.keys():
                    if k.startswith("K21_"):
                        truth_post[int(k.split("_")[1])] = z[k]

    mod = load_host(f"fk_{rank}_{arm}")
    fnp = mod.fnp
    counter = {"j": 0}
    orig_stars = mod._reset_stars

    def stars_wrap(k21_target, cp_post_k21, *rest, **kw):
        j = counter["j"]
        counter["j"] += 1
        force = (arm in ("all", "tensor") or
                 (arm.startswith("one:") and j == int(arm.split(":")[1])) or
                 (arm.startswith("ge:") and j >= int(arm.split(":")[1])))
        if force and j in truth_post:
            k21_target = fnp.asarray(
                truth_post[j].astype(np.float64), dtype=fnp.float64)
        return orig_stars(k21_target, cp_post_k21, *rest, **kw)

    mod._reset_stars = stars_wrap

    pre_k21_taps = []
    orig_frr = mod._full_relu_reset

    def frr_wrap(pre_mean, pre_covariance, pre_k21, *rest, **kw):
        pre_k21_taps.append(np.asarray(pre_k21, dtype=np.float64)
                            .view(np.ndarray).copy())
        return orig_frr(pre_mean, pre_covariance, pre_k21, *rest, **kw)

    mod._full_relu_reset = frr_wrap

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
              name=f"fk-{rank}-{arm}")
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=900.0, quiet=True):
            host_v, _, _ = mod.predict_with_native_tape(mlp)
    post = np.asarray(host_v, dtype=np.float64)

    e16 = mc_post[15] - post[15]
    noise16 = float(np.sum(mc_se[15] ** 2))
    res = {"rank": rank, "arm": arm, "stars_calls": counter["j"],
           "frr_taps": len(pre_k21_taps),
           "L16_mse_nc": (float(np.sum(e16 ** 2)) - noise16) / 1024.0,
           "finite": bool(np.isfinite(post).all())}
    # carried pre-K21 error vs truth at layers with true pre-planes
    pre_err = {}
    for l in range(2, 16):
        key = f"K21_{l}"
        if key in Pr.files and l - 1 < len(pre_k21_taps):
            T = Pr[key].astype(np.float64)
            KT = pre_k21_taps[l - 1]
            pre_err[str(l)] = round(
                float(np.sqrt(np.mean((KT - T) ** 2) / np.mean(T ** 2))), 4)
    res["pre_k21_rel_err"] = pre_err
    print(json.dumps(res, indent=1))
    (HERE / f"FORCE_K21_rank{rank:04d}_{arm.replace(':','')}.json").write_text(
        json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
