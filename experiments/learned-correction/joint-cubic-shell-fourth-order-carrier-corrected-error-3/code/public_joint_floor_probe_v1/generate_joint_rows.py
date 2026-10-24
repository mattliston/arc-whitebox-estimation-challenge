#!/usr/bin/env python3
"""Generate joint shell+carrier native rows (host/raw20/sigma), 2 workers."""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "estimator_joint.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = HERE / "rows"

def one(rank):
    out_dir = OUT / f"row_{rank:04d}"
    if (out_dir / "result.json").exists():
        return {"rank": rank, "status": "done"}
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"joint-{rank}")
    mlp.validate()
    spec = importlib.util.spec_from_file_location(f"joint_{os.getpid()}", SOURCE)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**42, wall_time_limit_s=600.0, quiet=True) as ctx:
            host_v, raw_v, sigma_v = m.predict_with_native_tape(mlp)
    host = np.ascontiguousarray(np.asarray(host_v), dtype="<f4")
    raw20 = np.ascontiguousarray(np.asarray(raw_v), dtype="<f4")
    sigma = np.ascontiguousarray(np.asarray(sigma_v), dtype="<f4")
    assert host.shape == (16, 1024) and np.isfinite(host).all()
    assert np.isfinite(raw20).all() and np.isfinite(sigma).all() and (sigma > 0).all()
    tmp = out_dir.with_name(f".{out_dir.name}.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "host.npy", host); np.save(tmp / "raw20.npy", raw20); np.save(tmp / "sigma.npy", sigma)
    (tmp / "result.json").write_text(json.dumps({
        "rank": rank, "mlp_id": int(entry["mlp_id"]), "flops_used": int(ctx.flops_used),
        "status": "complete", "targets_read": 0}) + "\n")
    tmp.replace(out_dir)
    return {"rank": rank, "flops": int(ctx.flops_used)}

if __name__ == "__main__":
    rank = int(sys.argv[1])
    print(json.dumps(one(rank)), flush=True)
