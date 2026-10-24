"""ORACLE-KAPPA-VARIANTS-V1: capture k3/k4 state of two variant estimators.

TAG: public_oracle_kappa_probe_v1/run_variants.py
Variants: cap8192 = workspace-v4 estimator with _CP4_CAP 2048 -> 8192 (one-line
copy, estimator_cap8192.py; zero CP4 truncation through layer 6 since max
candidate rank is 5120); k22r = public_k22_carrier_ab_probe_v1/estimator_k22r.py
(rank-4 k22-residual carrier; built from the condrepair_v3 lineage).
Knobs: RANKS=(35,100,201); capture-only baseline runs (no substitution);
BudgetContext(flop_budget=2**43, wall_time_limit_s=600, quiet=True);
threadpool_limits(limits=2); path=_predict.

Output per variant per rank: variant_{name}_rank{R:04d}.npz with k3_in/k4_in
(consumed diagonals, calls 1..15 -> layers 2..16) and the prediction rows.
Expected internal check: cap8192 k4 at layers 2-3 must be bit-close to the
base probe's (first CP4 truncation happens at reset 3 = layer-4 state).
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
ROOT = PKG.parent
sys.path.insert(0, str(PKG))
from run_probe import UnivariateTap, load_mlp, to_np64, WIDTH, DEPTH  # noqa: E402

VARIANTS = {
    "cap8192": PKG / "estimator_cap8192.py",
    "k22r": ROOT / "public_k22_carrier_ab_probe_v1/estimator_k22r.py",
}
RANKS = (35, 100, 201)
FLOP_BUDGET = 2 ** 43
WALL_LIMIT_S = 600


def load_variant(name, path):
    spec = importlib.util.spec_from_file_location(f"oracle_kappa_variant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_run(module, mlp, tag):
    original_uni = module._univariate_edgeworth
    tap = UnivariateTap(module, original_uni, None)
    module._univariate_edgeworth = tap
    try:
        context = module.flops.BudgetContext(
            flop_budget=FLOP_BUDGET,
            wall_time_limit_s=WALL_LIMIT_S,
            quiet=True,
            namespace=f"oracle_kappa_variant_{tag}",
        )
        started = time.perf_counter()
        with context:
            output = module._predict(mlp)
        wall = time.perf_counter() - started
        summary = context.summary_dict(by_namespace=False)
    finally:
        module._univariate_edgeworth = original_uni
    rows = to_np64(output)
    if rows.shape != (DEPTH, WIDTH) or len(tap.raw1_out) != DEPTH - 1:
        raise AssertionError(f"{tag}: unexpected capture shape")
    print(f"[run_variants] {tag}: wall={wall:.1f}s flops_used={summary.get('flops_used')}", flush=True)
    return {
        "rows": rows,
        "k3_in": np.stack(tap.k3_in, axis=0),
        "k4_in": np.stack(tap.k4_in, axis=0),
        "wall_s": wall,
        "flops_used": int(summary.get("flops_used", -1)),
    }


def main():
    names = sys.argv[1:] or list(VARIANTS)
    with threadpool_limits(limits=2):
        for name in names:
            module = load_variant(name, VARIANTS[name])
            if name == "cap8192" and int(module._CP4_CAP) != 8192:
                raise AssertionError("cap8192 variant did not take the new cap")
            for rank in RANKS:
                mlp, mlp_id = load_mlp(rank)
                cap = capture_run(module, mlp, f"{name}_rank{rank}")
                out = PKG / f"variant_{name}_rank{rank:04d}.npz"
                np.savez_compressed(
                    out, rank=rank, mlp_id=mlp_id, variant=name,
                    cp4_cap=int(module._CP4_CAP),
                    rows=cap["rows"], k3_in=cap["k3_in"], k4_in=cap["k4_in"],
                    wall_s=cap["wall_s"], flops_used=cap["flops_used"],
                )
                print(f"[run_variants] wrote {out}", flush=True)
    print("[run_variants] DONE", flush=True)


if __name__ == "__main__":
    main()
