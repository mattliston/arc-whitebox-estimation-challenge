#!/usr/bin/env python3
"""Generate one target-free combined-pilot row for one arm (host only)."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ARMS = {
    "base": {
        "source": P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py",
        "sha256": "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76",
    },
    "combined": {
        "source": P2 / "public_combined_card_v1/estimator.py",
        "sha256": "14c9288da98fbc80f009f0880eb87d73d2ce60016161aa4e5c022e1c942d1b76",
    },
}
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
OUT_ROOT = HERE / "rows"
PHASE2_BUDGET = 2**41
MAX_RSS = 8 * (1 << 30)


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    rank, arm = args.rank, args.arm
    assert 35 <= rank < 235
    source = ARMS[arm]["source"]
    source_sha = sha(source)
    assert source_sha == ARMS[arm]["sha256"], f"{arm} source drifted"
    prereg_sha = sha(PREREGISTRATION)
    out_dir = OUT_ROOT / arm / f"row_{rank:04d}"
    if out_dir.exists():
        existing = json.loads((out_dir / "result.json").read_text())
        if existing.get("status") == "complete-target-free-combined" and existing.get("source_sha256") == source_sha:
            print(json.dumps({"arm": arm, "rank": rank, "status": "already-complete"}))
            return
        raise AssertionError(f"{arm} rank {rank}: incompatible partial output")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    assert sha(rmp) == str(entry["sha256"])
    row = json.loads(rmp.read_text())
    wp = rmp.parent / "weights.npy"
    assert sha(wp) == row["files"]["weights.npy"]["sha256"]
    weights = np.load(wp, mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"combined-{arm}-rank-{rank:04d}")
    mlp.validate()
    spec = importlib.util.spec_from_file_location(f"combined_{arm}_{os.getpid()}", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=PHASE2_BUDGET, wall_time_limit_s=300.0, quiet=True) as ctx:
            value = estimator.predict(mlp, PHASE2_BUDGET)
    wall = time.perf_counter() - started
    host = np.ascontiguousarray(np.asarray(value), dtype=np.dtype("<f4"))
    assert host.shape == (16, 1024) and np.isfinite(host).all() and (host >= 0).all()
    flops_used = int(ctx.flops_used)
    assert flops_used <= PHASE2_BUDGET
    tmp = out_dir.with_name(f".{out_dir.name}.{os.getpid()}.tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    np.save(tmp / "host.npy", host, allow_pickle=False)
    receipt = {
        "schema": "combined-pilot-row-v1",
        "status": "complete-target-free-combined",
        "arm": arm,
        "role_rank": rank,
        "mlp_id": int(entry["mlp_id"]),
        "mlp_seed": int(row["mlp_seed"]),
        "flops_used": flops_used,
        "wall_seconds": wall,
        "source_sha256": source_sha,
        "preregistration_sha256": prereg_sha,
        "targets_read": 0,
    }
    (tmp / "result.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir)
    print(json.dumps({"arm": arm, "rank": rank, "flops_used": flops_used,
                      "wall_seconds": round(wall, 2)}), flush=True)


if __name__ == "__main__":
    main()
