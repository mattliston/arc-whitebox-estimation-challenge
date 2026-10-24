"""Target-free research-only section timing for immutable V4 candidate 888.

This diagnostic is not submission code.  NumPy is used only to recreate the
fixed synthetic weight fixture before the metered estimator call.  The loaded
estimator bytes are immutable.  Thin Python wrappers measure non-overlapping
top-level estimator phases and subtract the active BudgetContext's backend and
FlopScope-overhead deltas, exposing residual time and caller gaps.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import flopscope as flops
import numpy as np


HERE = Path(__file__).resolve().parent
ESTIMATOR = HERE / (
    "estimator_8883b2ccd8fce6f7993b32af615041ff9ec915c179f735a782ac5e7c86199734.py"
)
WIDTH = 1024
DEPTH = 16
SEED = 20260824


def main() -> None:
    estimator_bytes = ESTIMATOR.read_bytes()
    assert hashlib.sha256(estimator_bytes).hexdigest() == (
        "8883b2ccd8fce6f7993b32af615041ff9ec915c179f735a782ac5e7c86199734"
    )
    spec = importlib.util.spec_from_file_location("fb_ec4_888_sections", ESTIMATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    weight_stream = np.random.SeedSequence(SEED).spawn(3)[0]
    rng = np.random.default_rng(weight_stream)
    scale = float(np.sqrt(2.0 / WIDTH))
    weights = [
        (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]
    mlp = SimpleNamespace(width=WIDTH, depth=DEPTH, weights=weights)

    phase_names = (
        "_initialize_h1",
        "_cp_push",
        "_cp4_push",
        "_cp_read",
        "_push_cubic",
        "_cp4_read_planes",
        "_push_quartic",
        "_q38_coefficients",
        "_full_relu_reset",
        "_cp_post_read",
        "_cp4_reset",
        "_reset_stars",
        "_cp_grow",
        "_cp_k3",
        "_push_cubic_diagonal",
        "_push_quartic_diagonal",
        "_cp4_read_diagonal",
        "_univariate_edgeworth",
    )
    events = []
    active_depth = 0
    budget_holder = [None]

    def timed(name, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal active_depth
            if active_depth:
                return function(*args, **kwargs)
            active_depth += 1
            budget = budget_holder[0]
            wall0 = time.perf_counter()
            backend0 = budget.flopscope_backend_time_s
            overhead0 = budget.flopscope_overhead_time_s
            try:
                result = function(*args, **kwargs)
            finally:
                wall1 = time.perf_counter()
                backend1 = budget.flopscope_backend_time_s
                overhead1 = budget.flopscope_overhead_time_s
                active_depth -= 1
            events.append(
                {
                    "phase": name,
                    "start_s": wall0,
                    "stop_s": wall1,
                    "wall_s": wall1 - wall0,
                    "backend_s": backend1 - backend0,
                    "overhead_s": overhead1 - overhead0,
                    "residual_s": (
                        (wall1 - wall0)
                        - (backend1 - backend0)
                        - (overhead1 - overhead0)
                    ),
                }
            )
            return result

        return wrapper

    for phase_name in phase_names:
        setattr(module, phase_name, timed(phase_name, getattr(module, phase_name)))

    context_wall0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as budget:
        budget_holder[0] = budget
        predict_wall0 = time.perf_counter()
        predict_backend0 = budget.flopscope_backend_time_s
        predict_overhead0 = budget.flopscope_overhead_time_s
        output = module._predict(mlp)
        predict_wall1 = time.perf_counter()
        predict_backend1 = budget.flopscope_backend_time_s
        predict_overhead1 = budget.flopscope_overhead_time_s
    context_wall1 = time.perf_counter()
    assert output.shape == (DEPTH, WIDTH)
    assert bool(np.isfinite(np.asarray(output)).all())

    phase_totals = {}
    for event in events:
        totals = phase_totals.setdefault(
            event["phase"],
            {"count": 0, "wall_s": 0.0, "backend_s": 0.0,
             "overhead_s": 0.0, "residual_s": 0.0},
        )
        totals["count"] += 1
        for key in ("wall_s", "backend_s", "overhead_s", "residual_s"):
            totals[key] += event[key]

    intervals = []
    prior_stop = predict_wall0
    prior_phase = "predict_entry"
    for event in events:
        intervals.append(
            {
                "after": prior_phase,
                "before": event["phase"],
                "gap_s": event["start_s"] - prior_stop,
            }
        )
        prior_stop = event["stop_s"]
        prior_phase = event["phase"]
    intervals.append(
        {
            "after": prior_phase,
            "before": "predict_return",
            "gap_s": predict_wall1 - prior_stop,
        }
    )

    predict_wall = predict_wall1 - predict_wall0
    predict_backend = predict_backend1 - predict_backend0
    predict_overhead = predict_overhead1 - predict_overhead0
    predict_residual = predict_wall - predict_backend - predict_overhead
    phase_residual = sum(item["residual_s"] for item in events)
    gap_wall = sum(item["gap_s"] for item in intervals)
    payload = {
        "schema": "fb-ec4-v4-888-target-free-section-timing-v1",
        "candidate_sha256": hashlib.sha256(estimator_bytes).hexdigest(),
        "fixture": {"width": WIDTH, "depth": DEPTH, "seed": SEED},
        "budget_summary": budget.summary_dict(),
        "predict": {
            "wall_s": predict_wall,
            "backend_s": predict_backend,
            "overhead_s": predict_overhead,
            "residual_s": predict_residual,
            "classified_phase_residual_s": phase_residual,
            "caller_gap_wall_s": gap_wall,
            "decomposition_error_s": predict_residual - phase_residual - gap_wall,
        },
        "context_outer_wall_s": context_wall1 - context_wall0,
        "phase_totals": phase_totals,
        "events": events,
        "caller_gaps": intervals,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
