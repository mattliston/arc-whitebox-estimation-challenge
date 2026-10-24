#!/usr/bin/env python3
"""TAG p2-virgin-gate-v1: paired virgin gate for a CANDIDATE estimator file
on fresh 1024x16 nets (virgin seeds >= 3016500) whose truth (4M, MPS) and
deployed-card baseline were built by the fresh corpus chain
(fresh_<seed>_weights.npy, fresh_<seed>_truth4M.npz, fresh_<seed>_deployed.npy).
Runs the candidate under a real BudgetContext at the phase-2 budget (2^41),
reports per-net noise-corrected final-layer MSE for candidate vs deployed,
the paired ratio, the billed flops, and the mean/sd across nets.
Usage: <venv python> run_virgin_gate_p2.py <cand.py> <seed> [seed ...]"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_cand(path: Path):
    spec = importlib.util.spec_from_file_location("cand_" + hashlib.sha256(
        path.read_bytes()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    cand = Path(sys.argv[1]).resolve()
    seeds = [int(s) for s in sys.argv[2:]]
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    mod = load_cand(cand)
    sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    rows = []
    for seed in seeds:
        tr = np.load(HERE / f"fresh_{seed}_truth4M.npz")
        dep = np.load(HERE / f"fresh_{seed}_deployed.npy")
        W = np.load(HERE / f"fresh_{seed}_weights.npy", mmap_mode="r")
        mlp = MLP(width=1024, depth=16,
                  weights=[np.asarray(W[l]) for l in range(16)],
                  seed=int(derive_estimator_seed(seed)), name=f"fresh-{seed}")
        mlp.validate()
        t0 = time.time()
        with threadpool_limits(limits=4):
            with flops.BudgetContext(flop_budget=2 ** 41,
                                     wall_time_limit_s=900.0,
                                     quiet=True) as ctx:
                pred = np.asarray(mod.Estimator().predict(mlp, 2 ** 41),
                                  dtype=np.float64)
                billed = int(getattr(ctx, "flops_used", 0) or 0)
        wall = time.time() - t0
        truth = tr["mc_post"][15].astype(np.float64)
        noise = float(np.sum(tr["mc_se"][15].astype(np.float64) ** 2))
        def nc(x):
            return (float(np.sum((truth - x) ** 2)) - noise) / 1024
        m_c, m_d = nc(pred[15] if pred.ndim == 2 else pred), nc(
            dep[15] if dep.ndim == 2 else dep)
        row = {"seed": seed, "cand_mse_nc": m_c, "dep_mse_nc": m_d,
               "ratio": m_c / m_d if m_d > 0 else None,
               "cut_pct": round(100 * (1 - m_c / m_d), 2) if m_d > 0 else None,
               "billed": billed, "u": billed / 2 ** 41 if billed else None,
               "wall_s": round(wall, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    cuts = [r["cut_pct"] for r in rows if r["cut_pct"] is not None]
    out = {"cand": str(cand), "sha256": sha, "rows": rows,
           "mean_cut_pct": round(float(np.mean(cuts)), 2) if cuts else None,
           "sd_cut_pct": round(float(np.std(cuts)), 2) if cuts else None,
           "n": len(cuts)}
    print("VIRGIN_GATE", json.dumps({k: v for k, v in out.items()
                                     if k != "rows"}), flush=True)
    (HERE / f"VIRGIN_GATE_{sha[:8]}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
