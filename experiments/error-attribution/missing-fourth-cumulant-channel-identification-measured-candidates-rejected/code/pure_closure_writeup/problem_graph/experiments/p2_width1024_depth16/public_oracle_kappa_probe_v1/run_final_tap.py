"""ORACLE-FINAL-TAP-V1: supplied-host L16 consumed stats capture.

TAG: public_oracle_kappa_probe_v1/run_final_tap.py
Host: workspace-v4 estimator + K22-residual supply (run_state_audit_deep.py
instrument() verbatim, which also taps the univariate-consumed k3/k4 -- call 15
is the final layer).  Added here: a _q38_coefficients tap capturing the
(pre_mean, pre_variance) arguments; its 15th call in one predict is the final
layer's consumption site.  Output rows are kept for the model's L16 post-mean.
Knobs: RANKS=(35,201); BudgetContext(2**43, wall 600s, quiet); threads 2.
Output: final_tap_rank{R:04d}.npz with model_pre_mean/model_pre_var/
model_k3/model_k4 (all [1024], layer 16) and model_post_mean ([1024]).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
K211 = PKG.parent / "public_k211_supply_v1"
sys.path.insert(0, str(K211))
from run_state_audit_deep import instrument, load, BASE, STORE  # noqa: E402

import flopscope as flops  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from whestbench.domain import MLP  # noqa: E402
from whestbench.seeds import derive_estimator_seed  # noqa: E402

RANKS = tuple(int(a) for a in sys.argv[1:]) or (35, 201)


def main():
    manifest = json.loads((STORE / "manifest.json").read_text())
    for rank in RANKS:
        mod = load(f"final_tap_{rank}", BASE)
        state = instrument(mod)
        q38_calls = []
        orig_q38 = mod._q38_coefficients

        def q38_wrap(mean, variance, hermite_context, _orig=orig_q38, _calls=q38_calls):
            # Tag each call with how many univariate consumptions happened so
            # far: the L16 q38 call is uniquely the one at 14 (q38 for layer 1
            # and layer 2 both precede the first univariate call, and any
            # post-recurrence caller would see 15).
            _calls.append((
                len(state["k3"]),
                np.asarray(mean, dtype=np.float64).view(np.ndarray).copy(),
                np.asarray(variance, dtype=np.float64).view(np.ndarray).copy(),
            ))
            return _orig(mean, variance, hermite_context)

        mod._q38_coefficients = q38_wrap
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"final-tap-{rank}")
        mlp.validate()
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                rows = mod.Estimator().predict(mlp, 2**43)
        wall = time.perf_counter() - started
        final_calls = [c for c in q38_calls if c[0] == 14]
        if len(final_calls) != 1 or len(state["k3"]) < 15:
            raise AssertionError(
                f"rank {rank}: expected exactly one L16 q38 call (got {len(final_calls)}) "
                f"and >=15 univariate calls (got {len(state['k3'])})"
            )
        rows64 = np.array(np.asarray(rows), dtype=np.float64, copy=True)
        out = PKG / f"final_tap_rank{rank:04d}.npz"
        np.savez_compressed(
            out, rank=rank, mlp_id=int(entry["mlp_id"]),
            model_pre_mean=final_calls[0][1], model_pre_var=final_calls[0][2],
            model_k3=state["k3"][14], model_k4=state["k4"][14],
            model_post_mean_corrected=rows64[15], rows=rows64,
            n_q38_calls=len(q38_calls), wall_s=wall,
        )
        print(f"[final_tap] rank={rank} wall={wall:.1f}s q38_calls={len(q38_calls)} "
              f"wrote {out}", flush=True)
    print("[final_tap] DONE", flush=True)


if __name__ == "__main__":
    main()
