#!/usr/bin/env python3
"""TAG g20-retro-screen: run the G20 retrofit card (estimator_retro.py, a
gated copy of the deployed D11 lineage) on stored panel nets; report billed
flops + noise-corrected final-layer MSE vs truth, F2J ON (deployed path)
and F2J OFF (host only), vs the deployed outputs.  Env: RETRO_RESID_CP=1
(carry the K21 residual star as exact CP terms), RETRO_CERT=1, TAG=<name>.
Usage: (venv) RETRO_RESID_CP=1 TAG=residcp run_retro_screen.py 35 [100]"""
import importlib.util as ilu
import json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
def main():
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_limits
    flops.configure(symmetry_warnings=False)
    manifest = json.loads((STORE / "manifest.json").read_text())
    spec = ilu.spec_from_file_location("retrocard", HERE / "estimator_retro.py")
    mod = ilu.module_from_spec(spec); sys.modules["retrocard"] = mod
    spec.loader.exec_module(mod)
    tag = os.environ.get("TAG", "base")
    out = {"tag": tag, "env": {k: v for k, v in os.environ.items() if k.startswith("RETRO") or k in ("F32_COV", "F32_G4", "CP3_CAP")}}
    for a in sys.argv[1:]:
        rank = int(a)
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        W = np.load(rmp.parent / "weights.npy", mmap_mode="r")
        mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"retro-{rank}")
        mlp.validate()
        t0 = time.time()
        with threadpool_limits(limits=int(os.environ.get("THREADS", "4"))):
            ctx = flops.BudgetContext(flop_budget=2**41,
                                      wall_time_limit_s=1200.0, quiet=True)
            with ctx:
                host, native_raw, native_sigma = mod._predict(mlp, return_native_tape=True)
                host_flops = int(ctx.flops_used)
                corr = mod.transport_correction(native_raw, native_sigma, mlp.weights, mod._TRANSPORT_PAYLOAD)
                on = fnp.asarray(fnp.maximum(host + corr, 0.0), dtype=fnp.float32)
        billed = int(ctx.flops_used)
        wall = time.time() - t0
        pred_on = np.asarray(on, dtype=np.float64)[15]
        pred_off = np.asarray(host, dtype=np.float64)[15]
        tr = np.load(G19 / f"mps_truth_rank{rank:04d}_P8M.npz")
        dep = np.load(G19 / f"deployed_rank{rank:04d}.npy")
        tf = os.environ.get("TRUTH_FILE")
        trs = np.load(G19 / tf.replace("{rank}", f"{rank:04d}")) if tf else tr     # scoring truth (independent draw)
        truth = trs["mc_post"][15].astype(np.float64)
        noise = float(np.sum(trs["mc_se"][15] ** 2))
        def nc(p): return (float(np.sum((truth - p) ** 2)) - noise) / 1024
        mse_on, mse_off, mse_dep = nc(pred_on), nc(pred_off), nc(dep[15])
        truth8 = tr["mc_post"][15].astype(np.float64); noise8 = float(np.sum(tr["mc_se"][15] ** 2))
        nc8 = lambda p: (float(np.sum((truth8 - p) ** 2)) - noise8) / 1024
        out[rank] = {"billed": billed, "host_flops": host_flops,
                     "u_local": round(billed / 2**41, 4), "wall_s": round(wall, 1),
                     "mse_nc_f2j_on": mse_on, "mse_nc_f2j_off": mse_off,
                     "dep_mse_nc": mse_dep,
                     "raw_delta_pct_f2j_on_vs_dep": round(100*(mse_on/mse_dep-1), 2),
                     "raw_delta_pct_f2j_off_vs_dep": round(100*(mse_off/mse_dep-1), 2),
                     "max_abs_diff_on_vs_dep_L16": float(np.max(np.abs(pred_on - dep[15]))),
                     "scoring_truth": tf or "P8M(same-sample)",
                     "mse8M_f2j_on": nc8(pred_on), "mse8M_f2j_off": nc8(pred_off), "dep_mse8M": nc8(dep[15])}
        print(rank, json.dumps(out[rank]), flush=True)
        np.save(HERE / f"retro_out_{tag}_rank{rank:04d}.npy", np.stack([np.asarray(host, dtype=np.float64)[15], pred_on]))
    (HERE / f"RETRO_SCREEN_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
if __name__ == "__main__":
    main()
