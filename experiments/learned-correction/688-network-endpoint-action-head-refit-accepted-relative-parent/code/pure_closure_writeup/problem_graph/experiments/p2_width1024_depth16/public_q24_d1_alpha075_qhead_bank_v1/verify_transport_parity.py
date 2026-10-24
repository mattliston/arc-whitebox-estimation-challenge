#!/usr/bin/env python3
"""Target-free parity and exact incremental meter for the frozen bank.

This verifier opens one already-sealed observer-fit native tape and its public
weights only.  It never opens targets, validation rows, board data, or a full
closure fixture.  The independent authority is the original pair of frozen
PyTorch artifacts; the candidate path is the literal FlopScope recurrence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
CANDIDATE = HERE / "estimator.py"
INCUMBENT = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
J_ARTIFACT = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
Q_ARTIFACT = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
TAPE = P2 / "public_q24_native_transport_energy_pilot_v1/runs/native_rows_v1/row_0035"
WEIGHTS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy"
)
OUTPUT = HERE / "TRANSPORT_PARITY_RESULT.json"
AUTHORITY_ARRAYS = HERE / "TORCH_AUTHORITY_RANK35.npz"
AUTHORITY_MANIFEST = HERE / "TORCH_AUTHORITY_RANK35.json"

EXPECTED = {
    "candidate_sha256": "a6e332c5372839409340b100484d81e423f23239849e0c9de11b72669f9367ea",
    "incumbent_sha256": "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e",
    "j_artifact_sha256": "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5",
    "q_artifact_sha256": "5fb57a510d933940f19ddd5ab11650c3a966c2039c6bd12f73988457c2ce91b8",
    "raw20_sha256": "0768502bbd8beb26b85ffdd6b347f0312e1f59e7d4416c849efac8c2dec88365",
    "sigma_sha256": "c22932b89b08a4ad6be821245a275b1126ef8410cebdc7d75335b62359e55ff5",
    "weights_sha256": "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
}
PHASE2_BUDGET = 2_199_023_255_552


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def meter(label: str, function):
    started = time.perf_counter()
    context = flops.BudgetContext(
        flop_budget=PHASE2_BUDGET,
        wall_time_limit_s=120.0,
        quiet=True,
        namespace=label,
    )
    with threadpool_limits(limits=4), context:
        value = function()
    summary = context.summary_dict(by_namespace=False)
    return value, int(summary["flops_used"]), time.perf_counter() - started


def comparison(observed: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {
        "shape": list(observed.shape),
        "observed_array_sha256": array_sha256(observed),
        "authority_array_sha256": array_sha256(expected),
        "byte_identical": bool(np.array_equal(observed, expected)),
        "maximum_absolute_delta": float(np.max(np.abs(delta))),
        "root_mean_square_delta": float(np.sqrt(np.mean(delta * delta))),
        "maximum_authority_magnitude": float(np.max(np.abs(expected))),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    observed_hashes = {
        "candidate_sha256": sha256(CANDIDATE),
        "incumbent_sha256": sha256(INCUMBENT),
        "j_artifact_sha256": sha256(J_ARTIFACT),
        "q_artifact_sha256": sha256(Q_ARTIFACT),
        "raw20_sha256": sha256(TAPE / "raw20.npy"),
        "sigma_sha256": sha256(TAPE / "sigma.npy"),
        "weights_sha256": sha256(WEIGHTS),
    }
    if observed_hashes != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": observed_hashes})

    raw = np.load(TAPE / "raw20.npy", allow_pickle=False)
    sigma = np.load(TAPE / "sigma.npy", allow_pickle=False)
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    if raw.shape != (16, 1024, 20) or sigma.shape != (16, 1024):
        raise AssertionError("native tape shape changed")
    if weights.shape != (16, 1024, 1024):
        raise AssertionError("weight shape changed")
    if raw.dtype != np.float32 or sigma.dtype != np.float32 or weights.dtype != np.float32:
        raise AssertionError("native input dtype changed")

    authority_record = json.loads(AUTHORITY_MANIFEST.read_text(encoding="utf-8"))
    if authority_record.get("bindings") != {
        key: observed_hashes[key]
        for key in ("j_artifact_sha256", "q_artifact_sha256", "raw20_sha256", "sigma_sha256", "weights_sha256")
    }:
        raise AssertionError("authority bindings changed")
    if sha256(AUTHORITY_ARRAYS) != authority_record.get("arrays_file_sha256"):
        raise AssertionError("authority array file changed")
    with np.load(AUTHORITY_ARRAYS, allow_pickle=False) as authority:
        authority_j = np.ascontiguousarray(authority["j_correction"], dtype=np.float32)
        authority_q = np.ascontiguousarray(authority["q_correction"], dtype=np.float32)
    if array_sha256(authority_j) != authority_record.get("j_array_sha256"):
        raise AssertionError("J authority array changed")
    if array_sha256(authority_q) != authority_record.get("q_array_sha256"):
        raise AssertionError("q authority array changed")
    incumbent = load_module("d1_alpha075_incumbent", INCUMBENT)
    candidate = load_module("d1_alpha075_candidate", CANDIDATE)
    incumbent_value, incumbent_flops, incumbent_wall = meter(
        "d1_incumbent_transport_rank35",
        lambda: incumbent.transport_correction(
            raw, sigma, weights, incumbent._TRANSPORT_PAYLOAD
        ),
    )
    candidate_value, candidate_flops, candidate_wall = meter(
        "d1_alpha075_qhead_transport_rank35",
        lambda: candidate.transport_correction(
            raw,
            sigma,
            weights,
            candidate._TRANSPORT_PAYLOAD,
            candidate._QHEAD_PAYLOAD,
        ),
    )
    candidate_j = np.ascontiguousarray(np.asarray(candidate_value[0]), dtype=np.float32)
    candidate_q = np.ascontiguousarray(np.asarray(candidate_value[1]), dtype=np.float32)
    incumbent_array = np.ascontiguousarray(np.asarray(incumbent_value), dtype=np.float32)
    j_check = comparison(candidate_j, authority_j)
    q_check = comparison(candidate_q, authority_q)
    # PyTorch and FlopScope use independent matmul/nonlinearity front ends.
    # Their float32 recurrences need not be byte-identical, but numerical drift
    # must stay far below the scientific MSE scale.
    if float(j_check["maximum_absolute_delta"]) > 2.0e-7:
        raise AssertionError({"J parity failed": j_check})
    if float(q_check["maximum_absolute_delta"]) > 2.0e-7:
        raise AssertionError({"q-head parity failed": q_check})
    if not np.isfinite(candidate_j).all() or not np.isfinite(candidate_q).all():
        raise FloatingPointError("candidate transport is nonfinite")
    incremental_flops = candidate_flops - incumbent_flops
    if incremental_flops <= 0:
        raise AssertionError("candidate incremental bill must be positive")

    result = {
        "schema": "q24-d1-alpha075-qhead-target-free-transport-parity-v1",
        "status": "PASS_TARGET_FREE_TRANSPORT_PARITY",
        "role": "observer_fit",
        "role_rank": 35,
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "board_resource_opened": False,
        "bindings": observed_hashes,
        "authority": "exact frozen PyTorch alpha=.75 J and q-head artifacts",
        "authority_manifest_sha256": sha256(AUTHORITY_MANIFEST),
        "authority_arrays_sha256": sha256(AUTHORITY_ARRAYS),
        "j_parity": j_check,
        "qhead_parity": q_check,
        "incumbent_transport": {
            "flops_used": incumbent_flops,
            "wall_time_seconds": incumbent_wall,
            "array_sha256": array_sha256(incumbent_array),
        },
        "candidate_transport": {
            "flops_used": candidate_flops,
            "wall_time_seconds": candidate_wall,
            "incremental_flops": incremental_flops,
            "j_array_sha256": array_sha256(candidate_j),
            "qhead_array_sha256": array_sha256(candidate_q),
        },
        "maximum_process_rss_bytes": resident_bytes(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
