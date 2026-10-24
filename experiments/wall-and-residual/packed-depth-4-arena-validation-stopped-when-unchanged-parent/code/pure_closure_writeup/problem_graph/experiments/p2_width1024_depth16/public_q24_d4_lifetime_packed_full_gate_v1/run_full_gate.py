#!/usr/bin/env python3
"""Sealed target-free [35,279,35] full-component packed-D4 gate.

Default invocation performs only a standard-library static audit.  Numerical
execution requires a separately created ``FULL_GATE_RELEASE.json`` that pins
this runner and every authority after a fresh independent GO.  When released,
the coordinator launches parent D4 and packed D4 in separate official-runtime
processes, then compares all five public components byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STATIC = P2 / "public_q24_d4_lifetime_packed_reinvestment_v1"
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D15_DIR = P2 / "p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1"

RUNNER = HERE / "run_full_gate.py"
SOURCE = HERE / "estimator.py"
BUILDER = HERE / "build_candidate.py"
RELEASE_SCAFFOLD = HERE / "RELEASE_SCAFFOLD.json"
GATE_FREEZE = HERE / "GATE_FREEZE.json"
RELEASE = HERE / "FULL_GATE_RELEASE.json"
LOCK = HERE / "FULL_GATE_RUN.lock"
RESULT = HERE / "FULL_GATE_RESULT.json"

D4_SOURCE = D4_DIR / "estimator.py"
D4_RESULT = D4_DIR / "D4_CANDIDATE_RESULT.json"
PACKED_KERNEL = STATIC / "packed_d4_kernel.py"
PACKED_RECEIPT = STATIC / "PACKED_D4_STATIC_RECEIPT.json"
D15_RESULT = D15_DIR / "D15_CANDIDATE_RESULT.json"

SOURCE_SHA256 = "982775b3296709156e1d05c24d703a0982db8bab6dce83dc16744fb09361b066"
BUILDER_SHA256 = "41e3a047b9433cd721089cdea872ba92127c29bfaf4e2e7997e999cb1e27895b"
D4_SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
D4_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
PACKED_KERNEL_SHA256 = "bc23fe4f8a46bd194390b53523d63d5ab136fc1feb0240d589fcaf3fad681a37"
PACKED_RECEIPT_SHA256 = "832220c4c9467d8d57569649198e2465e97bc908ebb96103e549e683964c4b51"
D15_RESULT_SHA256 = "7571c8948ef992643642fb39731295b341fea9e7929aa8f9e53cbbe5c4623964"
D15_PRECEDENT_ROLE = "STATIC_LIFETIME_ONLY__FULL_D15_RESULT_IS_FAIL_TERMINAL"

CALL_ORDER = (35, 279, 35)
PUBLIC_COMPONENTS = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
COMPONENT_SCHEMAS = {
    "host": ("<f4", (16, 1024)),
    "native_raw20": ("<f4", (16, 1024, 20)),
    "native_sigma": ("<f4", (16, 1024)),
    "native_j_correction": ("<f4", (16, 1024)),
    "final_public_prediction": ("<f4", (16, 1024)),
}
FIXTURES = {
    35: (
        P2
        / "learned_residual_observer_design/v4_feature_store_v3"
        / "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy",
        "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    ),
    279: (
        P2
        / "learned_residual_observer_design/v4_feature_store_v3"
        / "sealed_target_blind_feature_store_v3/rows/observer_fit/0279/weights.npy",
        "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298",
    ),
}

EXPECTED_D4_FLOPS = (
    1_319_871_710_583,
    1_356_595_288_584,
    1_317_948_959_232,
)
EXPECTED_PACKED_FLOPS = (
    1_319_838_156_151,
    1_356_595_288_584,
    1_317_948_959_232,
)
EXPECTED_D4_TOTAL_CALLS = (15_416, 15_341, 15_331)
EXPECTED_PACKED_TOTAL_CALLS = (15_414, 15_341, 15_331)
COLD_RESHAPE_FLOP_SAVING = 33_554_432
COLD_RESHAPE_CALL_SAVING = 4
COLD_EMPTY_CALL_INCREMENT = 2

PHASE2_BUDGET = 2**41
MAX_RSS_BYTES = 8 * 2**30
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4
THREADS = 2
WORKER_TIMEOUT_SECONDS = 450
OFFICIAL_INTERPRETER = (P2 / "data_gate/.venv/bin/python").absolute()
OFFICIAL_ENVIRONMENT = {
    "interpreter": str(OFFICIAL_INTERPRETER),
    "python": "3.11.5",
    "flopscope": "0.12.0",
    "whestbench": "0.16.0",
    "numpy": "2.4.6",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def array_content_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with path.open("x") as handle:
        json.dump(safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_array_once(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "content_sha256": array_content_sha256(array),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
    }


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def authority_pins() -> dict[Path, str]:
    return {
        SOURCE: SOURCE_SHA256,
        BUILDER: BUILDER_SHA256,
        D4_SOURCE: D4_SOURCE_SHA256,
        D4_RESULT: D4_RESULT_SHA256,
        PACKED_KERNEL: PACKED_KERNEL_SHA256,
        PACKED_RECEIPT: PACKED_RECEIPT_SHA256,
        D15_RESULT: D15_RESULT_SHA256,
    }


def verify_authority_pins() -> None:
    for path, expected in authority_pins().items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"authority changed: {path}: {observed}")
    d4 = json.loads(D4_RESULT.read_text())
    if d4.get("status") != "PASS_TARGET_FREE_CANDIDATE_GATE":
        raise RuntimeError("bound D4 result is not passing")
    if tuple(d4.get("call_order", ())) != CALL_ORDER:
        raise RuntimeError("bound D4 call order changed")
    if tuple(int(trial["candidate_flops"]) for trial in d4["trials"]) != EXPECTED_D4_FLOPS:
        raise RuntimeError("bound D4 literal bills changed")
    d15 = json.loads(D15_RESULT.read_text())
    if d15.get("status") != "FAIL_TERMINAL":
        raise RuntimeError("D15 caveat changed: full result must remain FAIL_TERMINAL")
    if D15_PRECEDENT_ROLE != "STATIC_LIFETIME_ONLY__FULL_D15_RESULT_IS_FAIL_TERMINAL":
        raise RuntimeError("D15 role was broadened")


def expected_inventory(phase: str, ordinal: int) -> dict[str, Any]:
    d4 = json.loads(D4_RESULT.read_text())
    meter = d4["trials"][ordinal]["literal_meter"]
    calls = {
        name: int(item["calls"])
        for name, item in sorted(meter["operations"].items())
    }
    costs = {
        name: int(item["flop_cost"])
        for name, item in sorted(meter["operations"].items())
    }
    flops = int(meter["flops_used"])
    if phase == "candidate" and ordinal == 0:
        calls["reshape"] -= COLD_RESHAPE_CALL_SAVING
        calls["empty"] += COLD_EMPTY_CALL_INCREMENT
        costs["reshape"] -= COLD_RESHAPE_FLOP_SAVING
        flops -= COLD_RESHAPE_FLOP_SAVING
    expected_flops = EXPECTED_D4_FLOPS if phase == "parent" else EXPECTED_PACKED_FLOPS
    expected_total = (
        EXPECTED_D4_TOTAL_CALLS if phase == "parent" else EXPECTED_PACKED_TOTAL_CALLS
    )
    if flops != expected_flops[ordinal] or sum(calls.values()) != expected_total[ordinal]:
        raise RuntimeError("frozen expected-inventory derivation is inconsistent")
    return {
        "flops": flops,
        "total_calls": sum(calls.values()),
        "calls": calls,
        "costs": costs,
    }


def observed_environment() -> dict[str, str]:
    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "python": platform.python_version(),
        "flopscope": importlib.metadata.version("flopscope"),
        "whestbench": importlib.metadata.version("whestbench"),
        "numpy": importlib.metadata.version("numpy"),
    }


def expected_release_bindings() -> dict[str, str]:
    paths = {
        "runner_sha256": RUNNER,
        "candidate_source_sha256": SOURCE,
        "builder_sha256": BUILDER,
        "d4_source_sha256": D4_SOURCE,
        "d4_result_sha256": D4_RESULT,
        "packed_kernel_sha256": PACKED_KERNEL,
        "packed_static_receipt_sha256": PACKED_RECEIPT,
        "d15_fail_terminal_result_sha256": D15_RESULT,
        "release_scaffold_sha256": RELEASE_SCAFFOLD,
        "gate_freeze_sha256": GATE_FREEZE,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def static_receipt() -> dict[str, Any]:
    verify_authority_pins()
    source_text = SOURCE.read_text()
    d15 = json.loads(D15_RESULT.read_text())
    checks = {
        "candidate_arena_is_511393792": (
            "_CP3_WINOGRAD_ARENA_ELEMENTS = 511393792" in source_text
        ),
        "candidate_has_two_independent_read_empty_calls": (
            source_text.count("fnp.empty((width, width), dtype=fnp.float32)") == 2
        ),
        "candidate_routes_read_outputs_explicitly": (
            "first, second = read_outputs" in source_text
            and "cp3_context[8]" in source_text
        ),
        "candidate_does_not_import_layout_artifact": (
            "packed_d4_kernel" not in source_text
        ),
        "exact_packed_bills": tuple(
            expected_inventory("candidate", index)["flops"] for index in range(3)
        )
        == EXPECTED_PACKED_FLOPS,
        "exact_packed_total_calls": tuple(
            expected_inventory("candidate", index)["total_calls"]
            for index in range(3)
        )
        == EXPECTED_PACKED_TOTAL_CALLS,
        "d15_full_result_is_fail_terminal": d15.get("status") == "FAIL_TERMINAL",
        "d15_role_is_static_only": (
            D15_PRECEDENT_ROLE
            == "STATIC_LIFETIME_ONLY__FULL_D15_RESULT_IS_FAIL_TERMINAL"
        ),
        "official_interpreter_exists": OFFICIAL_INTERPRETER.is_file(),
        "release_scaffold_exists": RELEASE_SCAFFOLD.is_file(),
        "gate_freeze_exists": GATE_FREEZE.is_file(),
        "release_absent": not RELEASE.exists(),
        "result_absent": not RESULT.exists(),
        "run_lock_absent": not LOCK.exists(),
    }
    return {
        "schema": "packed-d4-full-five-component-gate-static-v1",
        "status": (
            "FROZEN_AWAITING_FRESH_INDEPENDENT_GO_HOLD"
            if all(checks.values())
            else "STATIC_HOLD"
        ),
        "checks": checks,
        "call_order": list(CALL_ORDER),
        "public_components": list(PUBLIC_COMPONENTS),
        "expected_d4_flops": list(EXPECTED_D4_FLOPS),
        "expected_packed_flops": list(EXPECTED_PACKED_FLOPS),
        "expected_d4_total_calls": list(EXPECTED_D4_TOTAL_CALLS),
        "expected_packed_total_calls": list(EXPECTED_PACKED_TOTAL_CALLS),
        "resource_gates": {
            "flops_strictly_below": PHASE2_BUDGET,
            "rss_strictly_below_bytes": MAX_RSS_BYTES,
            "residual_strictly_below_seconds": MAX_RESIDUAL_SECONDS,
            "call_wall_strictly_below_seconds": MAX_WALL_SECONDS,
        },
        "d15_precedent_role": D15_PRECEDENT_ROLE,
        "bindings": (
            expected_release_bindings()
            if RELEASE_SCAFFOLD.is_file() and GATE_FREEZE.is_file()
            else {}
        ),
        "scope": {
            "numerical_work_performed": False,
            "estimator_imports": 0,
            "estimator_calls": 0,
            "fixtures_opened": [],
            "targets_read": 0,
            "scores_read": 0,
            "network": False,
            "submission": False,
        },
    }


def require_release() -> dict[str, Any]:
    verify_authority_pins()
    if not RELEASE.is_file():
        raise RuntimeError("FULL_GATE_RELEASE.json absent: execution is not authorized")
    release = json.loads(RELEASE.read_text())
    if release.get("status") != "RELEASE_PACKED_D4_FULL_GATE_ONCE":
        raise RuntimeError("release status does not authorize this full gate")
    if release.get("independent_final_go") is not True:
        raise RuntimeError("release lacks fresh independent FINAL GO")
    if release.get("call_order") != list(CALL_ORDER):
        raise RuntimeError("release call order changed")
    if release.get("public_components") != list(PUBLIC_COMPONENTS):
        raise RuntimeError("release public-component contract changed")
    if release.get("runtime_environment") != OFFICIAL_ENVIRONMENT:
        raise RuntimeError("release runtime environment changed")
    if release.get("bindings") != expected_release_bindings():
        raise RuntimeError("release bindings do not exactly match frozen gate")
    if release.get("no_retry") is not True:
        raise RuntimeError("release must be no-retry")
    if release.get("targets_allowed") is not False or release.get("submission_allowed") is not False:
        raise RuntimeError("release broadens target/submission authority")
    review = release.get("review")
    if not isinstance(review, dict) or not review.get("reviewer_identity") or not review.get("summary"):
        raise RuntimeError("release lacks independent review identity/summary")
    if observed_environment() != OFFICIAL_ENVIRONMENT:
        raise RuntimeError(f"wrong execution environment: {observed_environment()}")
    if RESULT.exists() or LOCK.exists():
        raise FileExistsError("full gate already attempted or locked")
    return release


def load_module(tag: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def load_fixture(role_rank: int) -> Any:
    import numpy as np
    from whestbench.domain import MLP

    path, expected = FIXTURES[role_rank]
    if sha256_file(path) != expected:
        raise RuntimeError(f"fixture weights changed: {role_rank}")
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise RuntimeError(f"fixture schema changed: {role_rank}")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-weight-fixture",
    )
    mlp.validate()
    return mlp


def public_with_components(module: Any, estimator: Any, mlp: Any) -> tuple[Any, ...]:
    original_predict = module._predict
    original_transport = module.transport_correction
    state: dict[str, Any] = {"tape": None, "correction": None}

    def capture_predict(*args: Any, **kwargs: Any) -> Any:
        answer = original_predict(*args, **kwargs)
        if state["tape"] is not None or not isinstance(answer, tuple) or len(answer) != 3:
            raise RuntimeError("public internal tape call count/shape changed")
        state["tape"] = answer
        return answer

    def capture_transport(*args: Any, **kwargs: Any) -> Any:
        answer = original_transport(*args, **kwargs)
        if state["correction"] is not None:
            raise RuntimeError("public native-J executed more than once")
        state["correction"] = answer
        return answer

    module._predict = capture_predict
    module.transport_correction = capture_transport
    try:
        final = estimator.predict(mlp, module._PHASE2_FLOP_BUDGET)
    finally:
        module._predict = original_predict
        module.transport_correction = original_transport
    if state["tape"] is None or state["correction"] is None:
        raise RuntimeError("public path did not expose required tape/native-J")
    host, native_raw, native_sigma = state["tape"]
    return host, native_raw, native_sigma, state["correction"], final


def validate_worker_ticket(ticket_path: Path, phase: str) -> dict[str, Any]:
    nonce = os.environ.get("PACKED_D4_INTERNAL_WORKER_NONCE")
    if not nonce:
        raise RuntimeError("internal worker nonce absent")
    ticket = json.loads(ticket_path.read_text())
    if ticket.get("phase") != phase or ticket.get("nonce_sha256") != sha256_text(nonce):
        raise RuntimeError("internal worker ticket mismatch")
    if not LOCK.is_file() or ticket.get("lock_sha256") != sha256_file(LOCK):
        raise RuntimeError("internal worker lock mismatch")
    if ticket.get("runner_sha256") != sha256_file(RUNNER):
        raise RuntimeError("internal worker runner mismatch")
    if observed_environment() != OFFICIAL_ENVIRONMENT:
        raise RuntimeError("internal worker runtime mismatch")
    verify_authority_pins()
    return ticket


def worker_phase(phase: str, ticket_path: Path, staging: Path) -> None:
    import flopscope as flops
    import numpy as np
    from threadpoolctl import threadpool_limits

    validate_worker_ticket(ticket_path, phase)
    source = D4_SOURCE if phase == "parent" else SOURCE
    module = load_module(f"packed_d4_full_gate_{phase}", source)
    estimator = module.Estimator()
    fixtures = {rank: load_fixture(rank) for rank in sorted(set(CALL_ORDER))}
    d4 = json.loads(D4_RESULT.read_text())
    trials: list[dict[str, Any]] = []
    with threadpool_limits(limits=THREADS):
        for ordinal, role_rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"packed_d4_{phase}_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                remote_values = public_with_components(
                    module, estimator, fixtures[role_rank]
                )
            outer_wall = time.perf_counter() - started
            values = tuple(np.array(value, copy=True, order="C") for value in remote_values)
            summary = context.summary_dict(by_namespace=False)
            inventory = {
                "flops": int(summary["flops_used"]),
                "total_calls": sum(
                    int(item["calls"]) for item in summary["operations"].values()
                ),
                "calls": {
                    name: int(item["calls"])
                    for name, item in sorted(summary["operations"].items())
                },
                "costs": {
                    name: int(item["flop_cost"])
                    for name, item in sorted(summary["operations"].items())
                },
            }
            expected = expected_inventory(phase, ordinal)
            component_records: dict[str, Any] = {}
            schemas_ok = True
            for name, value in zip(PUBLIC_COMPONENTS, values):
                record = write_array_once(
                    staging / f"{phase}_{ordinal}_{name}.npy", value
                )
                expected_dtype, expected_shape = COMPONENT_SCHEMAS[name]
                record["schema_exact"] = (
                    record["dtype"] == expected_dtype
                    and tuple(record["shape"]) == expected_shape
                )
                schemas_ok = schemas_ok and bool(record["schema_exact"] and record["finite"])
                component_records[name] = record

            final_hash_matches_bound_d4 = True
            if phase == "parent":
                final_hash_matches_bound_d4 = (
                    component_records["final_public_prediction"]["content_sha256"]
                    == d4["trials"][ordinal]["output_array_sha256"]
                )
            rss = resident_bytes()
            gates = {
                "literal_flops_exact": inventory["flops"] == expected["flops"],
                "operation_calls_exact": inventory["calls"] == expected["calls"],
                "operation_costs_exact": inventory["costs"] == expected["costs"],
                "total_calls_exact": inventory["total_calls"] == expected["total_calls"],
                "all_component_schemas_and_finiteness": schemas_ok,
                "parent_final_matches_bound_d4_result": final_hash_matches_bound_d4,
                "flops_strictly_below_2pow41": inventory["flops"] < PHASE2_BUDGET,
                "outer_wall_strictly_below_120": outer_wall < MAX_WALL_SECONDS,
                "meter_wall_strictly_below_120": float(summary["wall_time_s"])
                < MAX_WALL_SECONDS,
                "official_residual_strictly_below_0_4": float(
                    summary["residual_wall_time_s"]
                )
                < MAX_RESIDUAL_SECONDS,
                "rss_strictly_below_8_gib": rss < MAX_RSS_BYTES,
            }
            gates["conjunctive_call_pass"] = all(gates.values())
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "inventory": inventory,
                "expected_inventory": expected,
                "meter_wall_time_seconds": float(summary["wall_time_s"]),
                "meter_backend_time_seconds": float(
                    summary["flopscope_backend_time_s"]
                ),
                "meter_overhead_time_seconds": float(
                    summary["flopscope_overhead_time_s"]
                ),
                "official_residual_wall_seconds": float(
                    summary["residual_wall_time_s"]
                ),
                "outer_wall_seconds": outer_wall,
                "maximum_process_rss_bytes": rss,
                "components": component_records,
                "gates": gates,
            }
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "ordinal": ordinal,
                        "role_rank": role_rank,
                        "flops": inventory["flops"],
                        "total_calls": inventory["total_calls"],
                        "residual": trial["official_residual_wall_seconds"],
                        "rss": rss,
                        "pass": gates["conjunctive_call_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not gates["conjunctive_call_pass"]:
                break

    payload = {
        "schema": "packed-d4-full-gate-worker-v1",
        "phase": phase,
        "call_order": list(CALL_ORDER),
        "trials": trials,
        "all_call_gates_pass": (
            len(trials) == len(CALL_ORDER)
            and all(trial["gates"]["conjunctive_call_pass"] for trial in trials)
        ),
        "estimator_calls": len(trials),
        "targets_read": 0,
        "scores_read": 0,
        "network": False,
        "bindings": {
            "runner_sha256": sha256_file(RUNNER),
            "source_sha256": sha256_file(source),
            "d4_result_sha256": sha256_file(D4_RESULT),
            "lock_sha256": sha256_file(LOCK),
        },
    }
    exclusive_json(staging / f"{phase}_worker_result.json", payload)
    if not payload["all_call_gates_pass"]:
        raise RuntimeError(f"{phase} worker terminal call failure")


def run_worker(phase: str, staging: Path, lock_sha256: str) -> dict[str, Any]:
    nonce = secrets.token_hex(32)
    ticket = {
        "schema": "packed-d4-full-gate-worker-ticket-v1",
        "phase": phase,
        "nonce_sha256": sha256_text(nonce),
        "lock_sha256": lock_sha256,
        "runner_sha256": sha256_file(RUNNER),
    }
    ticket_path = staging / f"{phase}_ticket.json"
    exclusive_json(ticket_path, ticket)
    environment = os.environ.copy()
    environment["PACKED_D4_INTERNAL_WORKER_NONCE"] = nonce
    command = [
        str(OFFICIAL_INTERPRETER),
        str(RUNNER),
        "--worker",
        phase,
        "--ticket",
        str(ticket_path),
        "--staging",
        str(staging),
    ]
    completed = subprocess.run(
        command,
        cwd=str(HERE),
        env=environment,
        text=True,
        capture_output=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} worker failed rc={completed.returncode}; "
            f"stdout={completed.stdout[-2000:]!r}; stderr={completed.stderr[-2000:]!r}"
        )
    result_path = staging / f"{phase}_worker_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"{phase} worker result absent")
    result = json.loads(result_path.read_text())
    if result.get("phase") != phase or result.get("all_call_gates_pass") is not True:
        raise RuntimeError(f"{phase} worker result is not passing")
    return result


def byte_equal_files(left: Path, right: Path) -> bool:
    import numpy as np

    a = np.load(left, allow_pickle=False)
    b = np.load(right, allow_pickle=False)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.ascontiguousarray(a).tobytes(order="C")
        == np.ascontiguousarray(b).tobytes(order="C")
    )


def execute() -> None:
    release = require_release()
    lock_payload = {
        "schema": "packed-d4-full-gate-no-retry-lock-v1",
        "status": "IRREVOCABLY_STARTED",
        "call_order": list(CALL_ORDER),
        "public_components": list(PUBLIC_COMPONENTS),
        "release_sha256": sha256_file(RELEASE),
        "runner_sha256": sha256_file(RUNNER),
        "candidate_source_sha256": sha256_file(SOURCE),
        "started_unix_ns": time.time_ns(),
    }
    exclusive_json(LOCK, lock_payload)
    lock_sha256 = sha256_file(LOCK)
    try:
        with tempfile.TemporaryDirectory(prefix="packed-d4-full-gate-", dir=HERE) as temporary:
            staging = Path(temporary)
            parent = run_worker("parent", staging, lock_sha256)
            candidate = run_worker("candidate", staging, lock_sha256)
            parity: list[dict[str, Any]] = []
            for ordinal, role_rank in enumerate(CALL_ORDER):
                component_parity: dict[str, bool] = {}
                for name in PUBLIC_COMPONENTS:
                    left = staging / f"parent_{ordinal}_{name}.npy"
                    right = staging / f"candidate_{ordinal}_{name}.npy"
                    component_parity[name] = byte_equal_files(left, right)
                parity.append({
                    "ordinal": ordinal,
                    "role_rank": role_rank,
                    "components": component_parity,
                    "all_five_byte_identical": all(component_parity.values()),
                })

            repeat: dict[str, dict[str, bool]] = {}
            for phase, worker in (("parent", parent), ("candidate", candidate)):
                repeat[phase] = {
                    name: (
                        worker["trials"][0]["components"][name]["content_sha256"]
                        == worker["trials"][2]["components"][name]["content_sha256"]
                    )
                    for name in PUBLIC_COMPONENTS
                }
            aggregate = {
                "parent_all_call_gates_pass": parent["all_call_gates_pass"],
                "candidate_all_call_gates_pass": candidate["all_call_gates_pass"],
                "all_fifteen_parent_candidate_components_byte_identical": all(
                    row["all_five_byte_identical"] for row in parity
                ),
                "parent_all_five_rank35_repeat_byte_identical": all(
                    repeat["parent"].values()
                ),
                "candidate_all_five_rank35_repeat_byte_identical": all(
                    repeat["candidate"].values()
                ),
                "maximum_candidate_rss_bytes": max(
                    int(trial["maximum_process_rss_bytes"])
                    for trial in candidate["trials"]
                ),
                "maximum_candidate_official_residual_seconds": max(
                    float(trial["official_residual_wall_seconds"])
                    for trial in candidate["trials"]
                ),
            }
            aggregate["rss_gate"] = aggregate["maximum_candidate_rss_bytes"] < MAX_RSS_BYTES
            aggregate["residual_gate"] = (
                aggregate["maximum_candidate_official_residual_seconds"]
                < MAX_RESIDUAL_SECONDS
            )
            aggregate["conjunctive_pass"] = all(
                value for key, value in aggregate.items() if isinstance(value, bool)
            )
            payload = {
                "schema": "packed-d4-full-five-component-gate-result-v1",
                "status": (
                    "PASS_TARGET_FREE_PACKED_D4_FULL_GATE"
                    if aggregate["conjunctive_pass"]
                    else "FAIL_TERMINAL"
                ),
                "call_order": list(CALL_ORDER),
                "public_components": list(PUBLIC_COMPONENTS),
                "parent_worker": parent,
                "candidate_worker": candidate,
                "parent_candidate_parity": parity,
                "rank35_repeat_parity": repeat,
                "aggregate": aggregate,
                "reference_estimator_calls": parent["estimator_calls"],
                "candidate_estimator_calls": candidate["estimator_calls"],
                "target_columns_opened": [],
                "scores_opened": [],
                "names_opened": [],
                "board_opened": False,
                "training": False,
                "network": False,
                "submission": False,
                "d15_precedent_role": D15_PRECEDENT_ROLE,
                "release": release,
                "bindings": {
                    **expected_release_bindings(),
                    "release_sha256": sha256_file(RELEASE),
                    "lock_sha256": lock_sha256,
                },
            }
            exclusive_json(RESULT, payload)
            print(json.dumps({"status": payload["status"], "aggregate": aggregate}, sort_keys=True))
            if not aggregate["conjunctive_pass"]:
                raise RuntimeError("terminal packed-D4 full-gate failure")
    except Exception as error:
        if not RESULT.exists():
            exclusive_json(
                RESULT,
                {
                    "schema": "packed-d4-full-five-component-gate-result-v1",
                    "status": "FAIL_TERMINAL",
                    "reason_type": type(error).__name__,
                    "reason": str(error),
                    "call_order": list(CALL_ORDER),
                    "public_components": list(PUBLIC_COMPONENTS),
                    "target_columns_opened": [],
                    "scores_opened": [],
                    "board_opened": False,
                    "network": False,
                    "submission": False,
                    "d15_precedent_role": D15_PRECEDENT_ROLE,
                    "bindings": {
                        **expected_release_bindings(),
                        "release_sha256": sha256_file(RELEASE),
                        "lock_sha256": lock_sha256,
                    },
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", choices=("parent", "candidate"))
    parser.add_argument("--ticket", type=Path)
    parser.add_argument("--staging", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        if args.execute or args.ticket is None or args.staging is None:
            raise RuntimeError("invalid internal-worker arguments")
        worker_phase(args.worker, args.ticket, args.staging)
        return
    if args.ticket is not None or args.staging is not None:
        raise RuntimeError("worker-only arguments supplied without --worker")
    if args.execute:
        execute()
        return
    print(json.dumps(static_receipt(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
