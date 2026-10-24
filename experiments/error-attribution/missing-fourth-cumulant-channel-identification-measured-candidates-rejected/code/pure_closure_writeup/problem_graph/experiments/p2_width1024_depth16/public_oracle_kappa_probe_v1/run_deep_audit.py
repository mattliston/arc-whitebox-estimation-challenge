"""ORACLE-DEEP-AUDIT-V1: supplied-host consumed k3/k4 captures, layers 2-16.

TAG: public_oracle_kappa_probe_v1/run_deep_audit.py
Host: workspace-v4 estimator with the K22-residual supply from
public_k211_supply_v1/run_state_audit.py -- its instrument() is imported and
used VERBATIM (tap on _univariate_edgeworth; K22-residual computed at each
_reset_stars from the post-reset k22 target minus CP4 provider minus
pairing-alpha model; injected at the next _assemble_quartic_provider through
the exact W^2 push).  Prediction path: mod.Estimator().predict(mlp, 2**43)
exactly as the supply script runs it.
Knobs: RANKS=(35,201); BudgetContext(2**43, wall 600s, quiet);
threadpool_limits(2).  Alignment: consumed[L-2] pairs oracle index L-1.
Output: deep_audit_rank{R:04d}.npz with k3_in/k4_in [15,1024] (calls 1..15 =
layers 2..16) and the prediction rows.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
K211 = PKG.parent / "public_k211_supply_v1"
sys.path.insert(0, str(K211))
from run_state_audit import instrument, load, BASE, STORE  # noqa: E402

import flopscope as flops  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from whestbench.domain import MLP  # noqa: E402
from whestbench.seeds import derive_estimator_seed  # noqa: E402

RANKS = (35, 201)


def main():
    manifest = json.loads((STORE / "manifest.json").read_text())
    for rank in RANKS:
        mod = load(f"deep_audit_{rank}", BASE)
        state = instrument(mod)
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"deep-aud-{rank}")
        mlp.validate()
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                rows = mod.Estimator().predict(mlp, 2**43)
        wall = time.perf_counter() - started
        n_calls = len(state["k3"])
        if n_calls < 15:
            raise AssertionError(
                f"rank {rank}: expected >=15 univariate calls (layers 2..16), got {n_calls}"
            )
        if n_calls > 15:
            print(f"[deep_audit] rank={rank}: {n_calls} univariate calls; using the first 15 "
                  f"(the recurrence runs before any later consumer)", flush=True)
        k3_in = np.stack(state["k3"][:15], axis=0)
        k4_in = np.stack(state["k4"][:15], axis=0)
        out = PKG / f"deep_audit_rank{rank:04d}.npz"
        np.savez_compressed(
            out, rank=rank, mlp_id=int(entry["mlp_id"]),
            k3_in=k3_in, k4_in=k4_in, n_univariate_calls=n_calls,
            rows=np.array(np.asarray(rows), dtype=np.float64, copy=True),
            wall_s=wall,
        )
        print(f"[deep_audit] rank={rank} wall={wall:.1f}s wrote {out}", flush=True)
    print("[deep_audit] DONE", flush=True)


if __name__ == "__main__":
    main()
