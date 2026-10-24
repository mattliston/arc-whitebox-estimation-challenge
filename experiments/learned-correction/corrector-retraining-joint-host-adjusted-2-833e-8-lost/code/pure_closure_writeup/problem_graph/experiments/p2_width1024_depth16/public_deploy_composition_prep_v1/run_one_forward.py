#!/usr/bin/env python3
"""One smoke forward of estimator_composed.py on store row rank 35.

TAG=composed-smoke  Loading pattern copied from public_combined_pilot_v1/
generate_row.py (manifest -> row manifest -> weights.npy, both sha-verified).
BudgetContext(2**42) so an over-2^41 bill is measured, not aborted.  Appends
the result under the "smoke" key of BUILD_RECEIPT.json; no store writes, no
targets, no submission.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
RECEIPT = HERE / "BUILD_RECEIPT.json"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
RANK = 35
SMOKE_BUDGET = 2**42
DEPLOY_BUDGET = 2**41


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=HERE / "estimator_composed.py",
        help="estimator to smoke (default: the composed card, sha-checked "
        "against BUILD_RECEIPT and recorded there; any other path skips "
        "the receipt)",
    )
    args = parser.parse_args()
    source_path = args.source.resolve()
    is_composed_card = source_path == (HERE / "estimator_composed.py")
    source_sha = sha(source_path)
    if is_composed_card:
        receipt = json.loads(RECEIPT.read_text())
        assert source_sha == receipt["output_sha256"], "composed card drifted"

    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][RANK]
    rmp = STORE / entry["path"]
    assert sha(rmp) == str(entry["sha256"])
    row = json.loads(rmp.read_text())
    wp = rmp.parent / "weights.npy"
    assert sha(wp) == row["files"]["weights.npy"]["sha256"]
    weights = np.load(wp, mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"composed-smoke-rank-{RANK:04d}")
    mlp.validate()

    flops.configure(symmetry_warnings=False)
    spec = importlib.util.spec_from_file_location(
        f"composed_smoke_{os.getpid()}", source_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    estimator = module.Estimator()

    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(
            flop_budget=SMOKE_BUDGET, wall_time_limit_s=600.0, quiet=True
        ) as ctx:
            value = estimator.predict(mlp, SMOKE_BUDGET)
    wall = time.perf_counter() - started
    output = np.asarray(value, dtype=np.float64)
    billed = int(ctx.flops_used)

    smoke = {
        "schema": "composed-smoke-forward-v1",
        "source_sha256": source_sha,
        "store_row_rank": RANK,
        "mlp_id": int(entry["mlp_id"]),
        "budget_context": SMOKE_BUDGET,
        "billed_flops": billed,
        "deployment_budget_2pow41": DEPLOY_BUDGET,
        "margin_to_2pow41": DEPLOY_BUDGET - billed,
        "under_deployment_budget": billed <= DEPLOY_BUDGET,
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "nonnegative": bool((output >= 0.0).all()),
        "output_final_layer_rms": float(
            np.sqrt(np.mean(np.square(output[-1])))
        ),
        "wall_seconds": wall,
        "targets_read": 0,
    }
    if is_composed_card:
        receipt["smoke"] = smoke
        tmp = RECEIPT.with_name(f".{RECEIPT.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(RECEIPT)
    print(json.dumps(smoke, indent=1), flush=True)


if __name__ == "__main__":
    main()
