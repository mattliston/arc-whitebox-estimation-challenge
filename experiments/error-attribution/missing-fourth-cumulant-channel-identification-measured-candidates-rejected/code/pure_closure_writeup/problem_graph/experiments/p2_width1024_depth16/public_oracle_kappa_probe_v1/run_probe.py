"""ORACLE-KAPPA-PROBE-V1: baseline + MC-oracle-substituted Q24 closure runs.

TAG: public_oracle_kappa_probe_v1/run_probe.py
Knobs: RANKS=(35,100,201)  SUB_LAYERS=2..6 (univariate-Edgeworth calls 1..5)
       BudgetContext(flop_budget=2**42, wall_time_limit_s=600, quiet=True)
       threadpool_limits(limits=2)  prediction path = module._predict (pure Q24
       closure recurrence; the P2R0F2J learned correction head is NOT applied).

Per rank, two runs of the frozen workspace-v4 estimator recurrence:
  baseline  -- tap _univariate_edgeworth to record, per call, the consumed k3
               (diag of pre_k21) and k4 (quartic diagonal) plus the returned
               raw1 (the model's post-activation mean for that layer).
  oracle    -- identical, except on calls 1..5 (layers 2..6) the k3/k4 vectors
               handed to _univariate_edgeworth are replaced by the MC oracle
               values from mc_oracle_rank{R}.npz.  Only the univariate
               consumption is substituted: _edgeworth_weights and every
               bivariate plane still see the model's own state.

Call index i of _univariate_edgeworth inside _predict corresponds to layer i+1
(1-based; layer 1 is the exact Gaussian first reset and consumes no k3/k4).
The estimator source is never modified; taps are runtime attribute patches on
the imported module and are restored after each run.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
WORKSPACE = ROOT / "public_q24_native_transport_energy_final_workspace_v4"
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

RANKS = (35, 100, 201)
WIDTH = 1024
DEPTH = 16
SUB_CALLS = (1, 2, 3, 4, 5)  # univariate calls -> layers 2..6
FLOP_BUDGET = 2 ** 42
WALL_LIMIT_S = 600
THREADS = 2


def load_estimator_module():
    spec = importlib.util.spec_from_file_location(
        "q24_ws4_estimator_oracle_kappa_probe", WORKSPACE / "estimator.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_mlp(rank):
    from whestbench.domain import MLP

    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"row_manifests[{rank}] rank mismatch")
    weights = np.load((STORE / entry["path"]).parent / "weights.npy")
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
        raise AssertionError(f"unexpected weights schema at rank {rank}")
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[np.ascontiguousarray(weights[layer]) for layer in range(DEPTH)],
        seed=0,
        name=f"oracle-kappa-probe-rank-{rank}",
    )
    mlp.validate()
    return mlp, int(entry["mlp_id"])


def to_np64(value):
    return np.array(np.asarray(value), dtype=np.float64, copy=True)


class UnivariateTap:
    """Record (and optionally substitute) the k3/k4 consumed per univariate call."""

    def __init__(self, module, original, substitute_np=None):
        self.module = module
        self.original = original
        self.substitute_np = substitute_np or {}
        self.k3_in = []
        self.k4_in = []
        self.raw1_out = []
        self.substituted_calls = []

    def __call__(self, coefficients, raw, sigma, alpha, pdf, k3, k4,
                 sigma2, sigma3, inverse_context=None):
        call_index = len(self.raw1_out) + 1
        k3_np = to_np64(k3)
        k4_np = to_np64(k4)
        if k3_np.shape != (WIDTH,) or k4_np.shape != (WIDTH,):
            raise AssertionError(
                f"call {call_index}: unexpected k3/k4 shapes {k3_np.shape} {k4_np.shape}"
            )
        self.k3_in.append(k3_np)
        self.k4_in.append(k4_np)
        sub = self.substitute_np.get(call_index)
        if sub is not None:
            fnp = self.module.fnp
            k3 = fnp.asarray(sub[0], dtype=fnp.float64)
            k4 = fnp.asarray(sub[1], dtype=fnp.float64)
            self.substituted_calls.append(call_index)
        out = self.original(
            coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,
            inverse_context,
        )
        self.raw1_out.append(to_np64(out[0]))
        return out


class ResetTap:
    """Record the pre-activation mean/variance entering each _full_relu_reset."""

    def __init__(self, module, original):
        self.module = module
        self.original = original
        self.pre_mean = []
        self.pre_var = []

    def __call__(self, pre_mean, pre_covariance, *args, **kwargs):
        self.pre_mean.append(to_np64(pre_mean))
        self.pre_var.append(np.diag(to_np64(pre_covariance)).copy())
        return self.original(pre_mean, pre_covariance, *args, **kwargs)


def run_once(module, flops, mlp, tag, substitute_np=None):
    original_uni = module._univariate_edgeworth
    original_reset = module._full_relu_reset
    uni_tap = UnivariateTap(module, original_uni, substitute_np)
    reset_tap = ResetTap(module, original_reset)
    module._univariate_edgeworth = uni_tap
    module._full_relu_reset = reset_tap
    try:
        context = flops.BudgetContext(
            flop_budget=FLOP_BUDGET,
            wall_time_limit_s=WALL_LIMIT_S,
            quiet=True,
            namespace=f"oracle_kappa_probe_{tag}",
        )
        started = time.perf_counter()
        with context:
            output = module._predict(mlp)
        wall = time.perf_counter() - started
        summary = context.summary_dict(by_namespace=False)
    finally:
        module._univariate_edgeworth = original_uni
        module._full_relu_reset = original_reset
    rows = to_np64(output)
    if rows.shape != (DEPTH, WIDTH):
        raise AssertionError(f"{tag}: unexpected output shape {rows.shape}")
    if len(uni_tap.raw1_out) != DEPTH - 1:
        raise AssertionError(
            f"{tag}: expected {DEPTH - 1} univariate calls, saw {len(uni_tap.raw1_out)}"
        )
    # Self-consistency: output rows 2..16 are the raw1 returns (float32 cast).
    for call in range(1, DEPTH):
        drift = float(np.max(np.abs(
            rows[call] - uni_tap.raw1_out[call - 1].astype(np.float32)
        )))
        if drift > 1.0e-6:
            raise AssertionError(f"{tag}: row/raw1 mismatch at call {call}: {drift}")
    print(f"[run_probe] {tag}: wall={wall:.1f}s flops_used={summary.get('flops_used')} "
          f"substituted_calls={uni_tap.substituted_calls}", flush=True)
    return {
        "rows": rows,
        "k3_in": np.stack(uni_tap.k3_in, axis=0),
        "k4_in": np.stack(uni_tap.k4_in, axis=0),
        "raw1": np.stack(uni_tap.raw1_out, axis=0),
        "pre_mean": np.stack(reset_tap.pre_mean, axis=0),
        "pre_var": np.stack(reset_tap.pre_var, axis=0),
        "substituted_calls": np.array(uni_tap.substituted_calls, dtype=np.int64),
        "wall_s": wall,
        "flops_used": int(summary.get("flops_used", -1)),
    }


def run_rank(module, flops, rank):
    import os
    suffix = os.environ.get("PROBE_MC_SUFFIX", "")
    mc_path = OUT / f"mc_oracle_rank{rank:04d}{suffix}.npz"
    mc = np.load(mc_path)
    if suffix == "" and int(mc["n_samples"]) < 2_000_000:
        raise AssertionError(f"{mc_path} is not the full-sample oracle")
    mlp, mlp_id = load_mlp(rank)

    baseline = run_once(module, flops, mlp, f"rank{rank}_baseline")

    substitute = {
        call: (
            np.ascontiguousarray(mc["k3"][call], dtype=np.float64),
            np.ascontiguousarray(mc["k4"][call], dtype=np.float64),
        )
        for call in SUB_CALLS  # mc arrays are layer-indexed [layer-1]; call i -> layer i+1 -> index i
    }
    oracle = run_once(module, flops, mlp, f"rank{rank}_oracle", substitute)

    if not np.array_equal(oracle["substituted_calls"], np.array(SUB_CALLS)):
        raise AssertionError(f"rank {rank}: substitution did not hit calls {SUB_CALLS}")
    if baseline["substituted_calls"].size != 0:
        raise AssertionError(f"rank {rank}: baseline run substituted calls")
    # The substituted inputs must differ from the model's own state.
    for call in SUB_CALLS:
        if np.allclose(baseline["k3_in"][call - 1], substitute[call][0]):
            print(f"[run_probe] rank {rank} call {call}: model k3 already equals oracle "
                  f"(unexpected)", flush=True)

    out_path = OUT / f"probe_rank{rank:04d}{suffix}.npz"
    np.savez_compressed(
        out_path,
        rank=rank,
        mlp_id=mlp_id,
        sub_calls=np.array(SUB_CALLS, dtype=np.int64),
        base_rows=baseline["rows"], base_k3=baseline["k3_in"], base_k4=baseline["k4_in"],
        base_raw1=baseline["raw1"], base_pre_mean=baseline["pre_mean"],
        base_pre_var=baseline["pre_var"], base_wall_s=baseline["wall_s"],
        base_flops=baseline["flops_used"],
        sub_rows=oracle["rows"], sub_k3=oracle["k3_in"], sub_k4=oracle["k4_in"],
        sub_raw1=oracle["raw1"], sub_pre_mean=oracle["pre_mean"],
        sub_pre_var=oracle["pre_var"], sub_wall_s=oracle["wall_s"],
        sub_flops=oracle["flops_used"],
    )
    print(f"[run_probe] rank={rank} wrote {out_path}", flush=True)


def main():
    ranks = [int(a) for a in sys.argv[1:]] or list(RANKS)
    print(f"[run_probe] TAG=public_oracle_kappa_probe_v1 ranks={ranks} "
          f"budget=2^42 wall_limit={WALL_LIMIT_S}s threads={THREADS} path=_predict", flush=True)
    module = load_estimator_module()
    flops = module.flops
    with threadpool_limits(limits=THREADS):
        for rank in ranks:
            run_rank(module, flops, rank)
    print("[run_probe] DONE", flush=True)


if __name__ == "__main__":
    main()
