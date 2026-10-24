#!/usr/bin/env python3
"""TAG k31-raw-probe-v1: host raw with R31+k4diag threaded on top of the K22
supply, 3 rows, vs the k22full host baselines (k22full_raw.log)."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = [35, 100, 160]
K22FULL_BASE = {35: 2.8339e-8, 100: 3.0574e-8, 160: 2.7794e-8}

sys.path.insert(0, str(HERE))
from run_state_audit_k31 import load, instrument


def main():
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)
    rel = []
    for rank in ROWS:
        mod = load(f"k31raw_{rank}", BASE)
        instrument(mod, True)
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"k31r-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**44, wall_time_limit_s=900.0, quiet=True):
                host_v, _, _ = mod.predict_with_native_tape(mlp)
        pred = np.asarray(host_v, dtype=np.float64)
        assert pred.shape == (16, 1024), pred.shape
        mse = float(np.mean((targets[rank][15] - pred[15]) ** 2))
        b = K22FULL_BASE[rank]
        rel.append(mse / b - 1.0)
        print(json.dumps({"rank": rank, "k22full": round(b * 1e8, 4),
                          "k31plus": round(mse * 1e8, 4),
                          "rel": round(mse / b - 1.0, 4)}), flush=True)
    print(json.dumps({"mean_rel_change": round(float(np.mean(rel)), 4),
                      "wins": int(sum(r < 0 for r in rel)), "n": len(rel)}))


if __name__ == "__main__":
    main()
