#!/usr/bin/env python3
"""Run the preregistered Q24-versus-V4 full-width seeds 20260825/20260826."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
BASE_DRIVER = HERE / "compare_target_free_v4_q24.py"
PREREGISTER = HERE / "PREREGISTER_ADDITIONAL_FULLWIDTH_SEEDS.json"
BASE_DRIVER_SHA256 = (
    "a2081115d0c1759fad11e18d05a0f53fcad1506088bbfe3a0515dba0e6940566"
)
PREREGISTER_SHA256 = (
    "8396259d723a9006d4d096c5f09a64877cd367fbf70e3e4b4e02d277d448129f"
)
SEEDS = (20260825, 20260826)
WIDTH = 1024
DEPTH = 16
FINAL_MSE_LIMIT = 1e-10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base():
    if _sha256(BASE_DRIVER) != BASE_DRIVER_SHA256:
        raise RuntimeError("base comparator hash mismatch")
    if _sha256(PREREGISTER) != PREREGISTER_SHA256:
        raise RuntimeError("additional-seed preregistration hash mismatch")
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    if preregister["fixture_law"]["seeds_exact"] != list(SEEDS):
        raise RuntimeError("preregistered seed set mismatch")
    spec = importlib.util.spec_from_file_location(
        "frozen_q24_v4_base_comparator", BASE_DRIVER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen base comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._verify_source_bindings()
    return module, preregister


def _fixture(base, seed: int) -> tuple[MLP, dict[str, Any]]:
    streams = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(streams[0])
    scale = float(np.sqrt(2.0 / WIDTH))
    weights = []
    layer_digests = []
    named_digests = []
    for layer in range(DEPTH):
        weight = (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        digest = base._array_digest(weight)
        weights.append(weight)
        layer_digests.append(digest)
        named_digests.append((f"mlp0/layer{layer}", digest))
    aggregate = base._combine_named_digests(named_digests)
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=weights,
        seed=int(streams[2].generate_state(1)[0]),
        name=f"target-free-official-law-seed{seed}",
    )
    mlp.validate()
    return mlp, {
        "seed": seed,
        "width": WIDTH,
        "depth": DEPTH,
        "seed_tree": "numpy.random.SeedSequence(seed).spawn(3)",
        "weight_generation": "float32(N(0,1) float64 * sqrt(2/width))",
        "aggregate_weights_sha256": aggregate,
        "layer_weight_sha256": layer_digests,
        "public_or_synthetic_target_values_read": 0,
        "truth_computed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not sys.dont_write_bytecode:
        raise SystemExit("run with Python -B / PYTHONDONTWRITEBYTECODE=1")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.output}")
    base, preregister = _load_base()
    v4_meter = base._verify_meter_receipt(
        base.V4_RECEIPT, base.V4_RECEIPT_SHA256, base.V4_SOURCE_SHA256
    )
    q24_meter = base._verify_meter_receipt(
        base.Q24_RECEIPT, base.Q24_RECEIPT_SHA256, base.Q24_SOURCE_SHA256
    )
    v4_module = base._load("frozen_v4_additional_seed_panel", base.V4_SOURCE)
    q24_module = base._load("frozen_q24_additional_seed_panel", base.Q24_SOURCE)
    panels = []
    board_hold_reasons = []
    for seed in SEEDS:
        mlp, fixture = _fixture(base, seed)
        result = base._run_pair(
            v4_module,
            q24_module,
            mlp,
            production=True,
            allow_failure=True,
        )
        if not result["v4"]["completed"]:
            board_hold_reasons.append(f"seed {seed}: V4 failed or was nonfinite")
        if not result["q24"]["completed"]:
            board_hold_reasons.append(f"seed {seed}: Q24 failed or was nonfinite")
        if result["disagreement"] is None:
            board_hold_reasons.append(f"seed {seed}: disagreement unavailable")
        else:
            final_mse = float(
                result["disagreement"]["final_prediction_disagreement_mse"]
            )
            if final_mse > FINAL_MSE_LIMIT:
                board_hold_reasons.append(
                    f"seed {seed}: final disagreement MSE {final_mse!r} > 1e-10"
                )
        panels.append({"fixture": fixture, "result": result})
    board_gate_pass = not board_hold_reasons
    receipt = {
        "schema": "fb-ec4-q24-additional-fullwidth-official-law-seeds-v1",
        "status": (
            "TECHNICAL_PASS__Q24_ROBUSTNESS_GATE_PASS__BOARD_AUTHORITY_SEPARATE"
            if board_gate_pass
            else "TECHNICAL_PASS__BOARD_HOLD__PREREGISTERED_ROBUSTNESS_GATE_FAILED"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "v4_source_sha256": base.V4_SOURCE_SHA256,
            "q24_source_sha256": base.Q24_SOURCE_SHA256,
            "base_comparator_sha256": BASE_DRIVER_SHA256,
            "preregister_path": PREREGISTER.name,
            "preregister_bytes": PREREGISTER.stat().st_size,
            "preregister_sha256": PREREGISTER_SHA256,
            "v4_literal_meter_receipt_sha256": base.V4_RECEIPT_SHA256,
            "q24_literal_meter_receipt_sha256": base.Q24_RECEIPT_SHA256,
            "v4_literal_meter_finite": bool(
                v4_meter["measured"]["output_contract"]["all_values_finite"]
            ),
            "q24_literal_meter_finite": bool(
                q24_meter["measured"]["output_contract"]["all_values_finite"]
            ),
        },
        "decision": {
            "final_prediction_disagreement_mse_limit": FINAL_MSE_LIMIT,
            "both_seeds_finite_and_within_limit": board_gate_pass,
            "board_hold": not board_gate_pass,
            "board_hold_reasons": board_hold_reasons,
            "decision_law_matches_preregister": (
                preregister["decision_law"]["threshold_was_fixed_before_both_runs"]
                is True
            ),
        },
        "panels": panels,
        "scope": {
            "dataset_accessed": False,
            "public_target_values_accessed": 0,
            "synthetic_target_values_accessed": 0,
            "truth_computed": False,
            "fits": 0,
            "samples_used_by_estimator": 0,
            "submissions": 0,
            "graph_edits": 0,
            "seeds_executed_exact": list(SEEDS),
            "seed_sweep": False,
            "interpretation": (
                "This fixed two-seed weights-only panel is prediction-disagreement "
                "evidence, not raw-MSE, accuracy, score, or board evidence."
            ),
        },
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, args.output)
    if not args.quiet:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
