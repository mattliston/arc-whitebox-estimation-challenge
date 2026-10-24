#!/usr/bin/env python3
"""Generate one target-free lambda3-X native row (host/raw20/sigma)."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_lambda3_shell_probe_v1/instrumented_estimator_x.py"
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

def array_sha256(a):
    a = np.ascontiguousarray(a)
    d = hashlib.sha256()
    d.update(a.dtype.str.encode("ascii"))
    d.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    d.update(memoryview(a).cast("B"))
    return d.hexdigest()

def resident_bytes():
    v = int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)
    return v if sys.platform == "darwin" else v * 1024

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    rank = args.rank
    assert 35 <= rank < 235
    source_sha = sha(SOURCE)
    prereg_sha = sha(PREREGISTRATION)
    out_dir = OUT_ROOT / f"row_{rank:04d}"
    if out_dir.exists():
        existing = json.loads((out_dir / "result.json").read_text())
        if existing.get("status") == "complete-target-free-x" and existing.get("source_sha256") == source_sha:
            print(json.dumps({"rank": rank, "status": "already-complete"}))
            return
        raise AssertionError(f"rank {rank}: incompatible partial output")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    assert sha(rmp) == str(entry["sha256"])
    row = json.loads(rmp.read_text())
    wp = rmp.parent / "weights.npy"
    assert sha(wp) == row["files"]["weights.npy"]["sha256"]
    weights = np.load(wp, mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"lambda3-x-rank-{rank:04d}")
    mlp.validate()
    spec = importlib.util.spec_from_file_location(f"x_native_{os.getpid()}", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=PHASE2_BUDGET, wall_time_limit_s=120.0, quiet=True) as ctx:
            host_v, raw_v, sigma_v = module.predict_with_native_tape(mlp)
    wall = time.perf_counter() - started
    host = np.ascontiguousarray(np.asarray(host_v), dtype=np.dtype("<f4"))
    raw20 = np.ascontiguousarray(np.asarray(raw_v), dtype=np.dtype("<f4"))
    sigma = np.ascontiguousarray(np.asarray(sigma_v), dtype=np.dtype("<f4"))
    assert host.shape == (16, 1024) and np.isfinite(host).all() and (host >= 0).all()
    assert raw20.shape == (16, 1024, 20) and np.isfinite(raw20).all()
    assert sigma.shape == (16, 1024) and np.isfinite(sigma).all() and (sigma > 0).all()
    flops_used = int(ctx.flops_used)
    assert flops_used <= PHASE2_BUDGET
    rss = resident_bytes()
    assert rss < MAX_RSS
    tmp = out_dir.with_name(f".{out_dir.name}.{os.getpid()}.tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    np.save(tmp / "host.npy", host, allow_pickle=False)
    np.save(tmp / "raw20.npy", raw20, allow_pickle=False)
    np.save(tmp / "sigma.npy", sigma, allow_pickle=False)
    receipt = {
        "schema": "lambda3-x-native-row-v1",
        "status": "complete-target-free-x",
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": int(entry["mlp_id"]),
        "mlp_seed": int(row["mlp_seed"]),
        "files": {name: {"sha256": sha(tmp / name), "array_sha256": array_sha256(arr)}
                  for name, arr in (("host.npy", host), ("raw20.npy", raw20), ("sigma.npy", sigma))},
        "flops_used": flops_used,
        "wall_seconds": wall,
        "maximum_process_rss_bytes": rss,
        "source_sha256": source_sha,
        "preregistration_sha256": prereg_sha,
        "targets_read": 0,
        "observer_selection_opened": 0,
    }
    (tmp / "result.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir)
    print(json.dumps({"rank": rank, "flops_used": flops_used, "wall_seconds": round(wall, 2)}), flush=True)

if __name__ == "__main__":
    main()
