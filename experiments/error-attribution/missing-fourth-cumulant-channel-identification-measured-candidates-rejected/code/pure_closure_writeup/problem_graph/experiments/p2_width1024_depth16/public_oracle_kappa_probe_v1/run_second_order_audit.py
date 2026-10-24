"""ORACLE-SECOND-ORDER-AUDIT-V1: supplied-host pre_mean/pre_var/pair-cov taps.

TAG: public_oracle_kappa_probe_v1/run_second_order_audit.py
Host: workspace-v4 estimator + the K22-residual supply -- instrument() from
public_k211_supply_v1/run_state_audit_deep.py used VERBATIM (note: that file's
instrument() taps only k3/k4; it contains no second-order taps, so the
pre_mean / pre_variance / pair-covariance capture below is added here via a
_full_relu_reset wrapper -- the exact tap convention the L2-6 probe validated;
_full_relu_reset receives the consumed pre_mean and pre_covariance directly).
Knobs: RANKS=(35,201); BudgetContext(2**43, wall 600s, quiet);
threadpool_limits(2); predict path Estimator().predict (as the supply runs).
Reset call c (1..14) = layer c+1 (layers 2..15).
Output: second_order_rank{R:04d}.npz with pre_mean [14,1024], pre_var [14,1024],
pre_cov_pairs [14,2000] at the pairs_spec.py pair list.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
K211 = PKG.parent / "public_k211_supply_v1"
sys.path.insert(0, str(K211))
sys.path.insert(0, str(PKG))
from run_state_audit_deep import instrument, load, BASE, STORE  # noqa: E402
from pairs_spec import pair_list  # noqa: E402

import flopscope as flops  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from whestbench.domain import MLP  # noqa: E402
from whestbench.seeds import derive_estimator_seed  # noqa: E402

RANKS = (35, 201)


def add_second_order_tap(mod, pairs_i, pairs_j):
    captured = {"pre_mean": [], "pre_var": [], "pre_cov_pairs": []}
    orig_reset = mod._full_relu_reset

    def reset_wrap(pre_mean, pre_covariance, *args, **kwargs):
        cov = np.asarray(pre_covariance, dtype=np.float64).view(np.ndarray)
        captured["pre_mean"].append(np.asarray(pre_mean, dtype=np.float64).view(np.ndarray).copy())
        captured["pre_var"].append(np.diag(cov).copy())
        captured["pre_cov_pairs"].append(cov[pairs_i, pairs_j].copy())
        return orig_reset(pre_mean, pre_covariance, *args, **kwargs)

    mod._full_relu_reset = reset_wrap
    return captured


def main():
    pairs_i, pairs_j = pair_list()
    manifest = json.loads((STORE / "manifest.json").read_text())
    for rank in RANKS:
        mod = load(f"second_order_audit_{rank}", BASE)
        instrument(mod)  # the K22-residual supply + its k3/k4 taps, verbatim
        captured = add_second_order_tap(mod, pairs_i, pairs_j)
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"so-aud-{rank}")
        mlp.validate()
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                mod.Estimator().predict(mlp, 2**43)
        wall = time.perf_counter() - started
        n_resets = len(captured["pre_mean"])
        if n_resets < 14:
            raise AssertionError(f"rank {rank}: expected >=14 reset calls, got {n_resets}")
        if n_resets > 14:
            print(f"[second_order] rank={rank}: {n_resets} reset calls; using first 14", flush=True)
        out = PKG / f"second_order_rank{rank:04d}.npz"
        np.savez_compressed(
            out, rank=rank, mlp_id=int(entry["mlp_id"]),
            pairs_i=pairs_i, pairs_j=pairs_j, n_reset_calls=n_resets,
            pre_mean=np.stack(captured["pre_mean"][:14], axis=0),
            pre_var=np.stack(captured["pre_var"][:14], axis=0),
            pre_cov_pairs=np.stack(captured["pre_cov_pairs"][:14], axis=0),
            wall_s=wall,
        )
        print(f"[second_order] rank={rank} wall={wall:.1f}s wrote {out}", flush=True)
    print("[second_order] DONE", flush=True)


if __name__ == "__main__":
    main()
