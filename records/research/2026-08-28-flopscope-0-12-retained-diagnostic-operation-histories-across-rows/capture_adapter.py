#!/usr/bin/env python3
"""Token-gated target-blind adapter for the lean three-bank runtime V4.

Only standard-library modules and the process-local token validator are loaded
at import time.  The runner imports this module only after it owns the durable
claim and has consumed the capability.  Every row performs exactly one
captured parent pass with the accepted literal source.  Only the predeclared
ranks 0, 343, and 687 additionally perform the independent-reference source
execution and unhooked parent pass needed for dynamic byte-parity sentinels.
Sentinel choice is rank-only and is fixed before any row or weight is opened.

The source already computes one H256 contraction whose receiver order nests
H64 and H16.  V3 fetches H256 once, derives the two narrower physical banks by
byte-preserving sector-prefix copies on the host, and verifies the derivation
against the literal and independent source outputs at every sentinel.

The FlopScope context uses an astronomically high diagnostic ceiling and no
wall deadline.  The deployment FLOP cap, RSS, wall, and residual observations
are computed only after finite action bytes have been copied into the staging
bank; no resource observation can veto or alter accuracy generation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import time

from q24_three_bank_runtime_token_v2 import require_runtime_token


WIDTH = 1024
DEPTH = 16
ROW_COUNT = 688
SENTINEL_RANKS = (0, 343, 687)
MEMMAP_FLUSH_INTERVAL_ROWS = 8
CAPTURE_LAYERS = tuple(range(2, 15))
EXPECTED_H = (16, 64, 256)
DEPLOYMENT_FLOP_CAP = 2_199_023_255_552
# FlopScope's remote protocol serializes budgets as unsigned 64-bit integers.
# This protocol-safe diagnostic ceiling is not a deployment threshold.  The
# frozen fixed-row upper bound below is conservative relative to two measured
# 1.357T parent passes plus both source laws; the ceiling retains >2.3 million
# times that bound and therefore cannot bind this fixed-shape accuracy run.
FIXED_ACCURACY_ROW_FLOP_UPPER_BOUND = 4_000_000_000_000
DIAGNOSTIC_METER_CEILING = (1 << 63) - 1
COMPONENT_SHAPES = {
    "host": (16, 1024),
    "native_raw20": (16, 1024, 20),
    "native_sigma": (16, 1024),
    "native_j_correction": (16, 1024),
    "final_public_prediction": (16, 1024),
}
EXPECTED_ROW_MANIFEST_SCHEMA = "phase2-observer-v4-target-blind-feature-row-v1"
EXPECTED_ROW_MANIFEST_ROOT_KEYS = (
    "decoded_columns",
    "edge_materialization",
    "files",
    "fits",
    "frozen_host_base_sha256",
    "frozen_host_worker_sha256",
    "host_caps",
    "host_latch",
    "hosts",
    "mlp_id",
    "mlp_seed",
    "optimizers_constructed",
    "rlimit_as_installed",
    "role",
    "role_rank",
    "schema",
    "scores",
    "source_ordinal",
    "source_row_in_shard",
    "source_shard_index",
    "status",
    "strict_group_rss_watchdog_required",
    "target_columns_projected",
    "target_values_decoded",
    "total_worker_wall_seconds",
    "weights_raw_sha256",
    "worker_environment",
    "worker_max_rss_bytes",
    "worker_source_sha256",
)
EXPECTED_DECODED_COLUMNS = ("mlp_id", "mlp_name", "mlp_seed", "weights")
FORBIDDEN_MANIFEST_KEY_TOKENS = (
    "target",
    "label",
    "ground_truth",
    "mse",
    "score",
    "board",
)
# These three root fields are provenance counters, not target data.  Their
# names are allowed only at these exact paths and only as built-in integer
# zero.  A bool is deliberately rejected even though bool subclasses int.
ALLOWED_ZERO_AUDIT_PATHS = (
    "manifest.scores",
    "manifest.target_columns_projected",
    "manifest.target_values_decoded",
)


class AdapterIntegrityError(RuntimeError):
    """Raised for authority, target-blindness, or byte-integrity failure."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _operation_calls(summary):
    return sum(int(item["calls"]) for item in summary["operations"].values())


def _resident_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _regular_nonsymlink_path(path, label):
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        information = os.lstat(current)
        if stat.S_ISLNK(information.st_mode):
            raise AdapterIntegrityError(label + " contains a symlink component")
    information = os.lstat(absolute)
    if not stat.S_ISREG(information.st_mode):
        raise AdapterIntegrityError(label + " is not a regular file")
    return absolute


def _reject_forbidden_manifest(value, location="manifest"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = location + "." + str(key)
            if any(
                token in str(key).lower()
                for token in FORBIDDEN_MANIFEST_KEY_TOKENS
            ):
                if child_location not in ALLOWED_ZERO_AUDIT_PATHS:
                    raise AdapterIntegrityError(
                        "forbidden field in row manifest: " + child_location
                    )
                if type(child) is not int or child != 0:
                    raise AdapterIntegrityError(
                        "allowed audit counter must be exact built-in int zero: "
                        + child_location
                    )
            _reject_forbidden_manifest(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_manifest(child, location + "[" + str(index) + "]")


def _validate_manifest_metadata(manifest):
    """Validate the exact target-blind row-manifest metadata contract."""
    _reject_forbidden_manifest(manifest)
    if type(manifest) is not dict or tuple(sorted(manifest)) != (
        EXPECTED_ROW_MANIFEST_ROOT_KEYS
    ):
        raise AdapterIntegrityError("row manifest root keyset mismatch")
    if manifest.get("schema") != EXPECTED_ROW_MANIFEST_SCHEMA:
        raise AdapterIntegrityError("row manifest schema mismatch")
    if tuple(manifest.get("decoded_columns", ())) != EXPECTED_DECODED_COLUMNS:
        raise AdapterIntegrityError("row manifest decoded columns mismatch")
    for counter in ("fits", "optimizers_constructed"):
        if type(manifest.get(counter)) is not int or manifest[counter] != 0:
            raise AdapterIntegrityError(
                "row manifest construction counter must be exact built-in int zero: "
                + counter
            )


def _fetch_array(np, value, shape):
    if hasattr(value, "_fetch_data"):
        raw, remote_shape, remote_dtype = value._fetch_data()
        if tuple(remote_shape) != tuple(shape):
            raise AdapterIntegrityError("remote array shape mismatch")
        array = np.frombuffer(raw, dtype=np.dtype(remote_dtype)).reshape(shape).copy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.shape != tuple(shape) or array.dtype != np.dtype("<f4"):
        raise AdapterIntegrityError(
            "array is not exact little-endian float32 geometry"
        )
    if not bool(np.isfinite(array).all()):
        raise AdapterIntegrityError("nonfinite parent or source array")
    return array


def _bytes_equal(np, left, right):
    return left.shape == right.shape and left.dtype == right.dtype and bool(
        np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )


class _LiteralCapture:
    """Execute the literal source at every hook, optionally with a reference."""

    def __init__(
        self,
        *,
        literal_source,
        reference_source,
        literal_workspace,
        reference_workspace,
        budget,
        runtime_token,
    ):
        self._literal = literal_source
        self._reference = reference_source
        self._literal_workspace = literal_workspace
        self._reference_workspace = reference_workspace
        self._budget = budget
        self._token = runtime_token
        self.capture_count = 0
        self.literal_flops = 0
        self.literal_calls = 0
        self.reference_flops = 0
        self.reference_calls = 0
        self.reference_enabled = reference_source is not None

    def _counters(self):
        require_runtime_token(self._token)
        summary = self._budget.summary_dict(by_namespace=False)
        return int(self._budget.flops_used), _operation_calls(summary)

    def capture_one_layer(
        self,
        fnp,
        layer_index,
        post_covariance,
        post_k21,
        post_k4,
        post_k31,
        post_k22,
        post_variance,
        post_alpha,
        _workspace_sentinel,
        runtime_token,
    ):
        require_runtime_token(runtime_token)
        if runtime_token is not self._token:
            raise AdapterIntegrityError("capture token identity changed")
        before_flops, before_calls = self._counters()
        self._literal.capture_one_layer(
            fnp,
            layer_index,
            post_covariance,
            post_k21,
            post_k4,
            post_k31,
            post_k22,
            post_variance,
            post_alpha,
            self._literal_workspace,
            runtime_token,
        )
        after_flops, after_calls = self._counters()
        self.literal_flops += after_flops - before_flops
        self.literal_calls += after_calls - before_calls

        if self.reference_enabled:
            before_flops, before_calls = self._counters()
            self._reference.capture_reference_one_layer(
                fnp,
                layer_index,
                post_covariance,
                post_k21,
                post_k4,
                post_k31,
                post_k22,
                post_variance,
                post_alpha,
                self._reference_workspace,
                runtime_token,
            )
            after_flops, after_calls = self._counters()
            self.reference_flops += after_flops - before_flops
            self.reference_calls += after_calls - before_calls
        self.capture_count += 1

    def bank_views(self, _workspace_sentinel, runtime_token):
        require_runtime_token(runtime_token)
        if runtime_token is not self._token:
            raise AdapterIntegrityError("bank-view token identity changed")
        return self._literal.bank_views(self._literal_workspace, runtime_token)

    def reference_bank_views(self):
        if not self.reference_enabled:
            raise AdapterIntegrityError("reference views requested off sentinel")
        return self._reference.reference_bank_views(
            self._reference_workspace, self._token
        )


def _derive_narrow_from_h256(np, action256, h):
    """Materialize the exact eight-sector receiver prefix as C-order bytes."""
    if h not in (16, 64):
        raise AdapterIntegrityError("narrow derivation requires H16 or H64")
    receivers = h // 8
    if action256.shape == (13, WIDTH, 256):
        leading = (13, WIDTH)
        sector_view = action256.reshape(13, WIDTH, 8, 32)[:, :, :, :receivers]
    elif action256.shape == (WIDTH, 256):
        leading = (WIDTH,)
        sector_view = action256.reshape(WIDTH, 8, 32)[:, :, :receivers]
    else:
        raise AdapterIntegrityError("H256 derivation source geometry mismatch")
    derived = np.ascontiguousarray(sector_view.reshape(*leading, h))
    if derived.shape != (*leading, h) or derived.dtype != np.dtype("<f4"):
        raise AdapterIntegrityError("derived narrow bank geometry mismatch")
    return derived


def _validate_record_and_load_mlp(
    session, record, expected_rank, runtime_token
):
    require_runtime_token(runtime_token)
    if runtime_token is not session.get("runtime_token"):
        raise AdapterIntegrityError("record-loader token identity changed")
    np = session["np"]
    if type(record.get("rank")) is not int or record["rank"] != expected_rank:
        raise AdapterIntegrityError("adapter record rank mismatch")
    manifest_path = _regular_nonsymlink_path(
        record["row_manifest_path"], "row manifest"
    )
    weights_path = _regular_nonsymlink_path(record["weights_path"], "weights")
    if weights_path.parent != manifest_path.parent:
        raise AdapterIntegrityError("weights must be colocated with row manifest")
    if _sha256_file(manifest_path) != record["row_manifest_sha256"]:
        raise AdapterIntegrityError("row manifest digest mismatch")
    if _sha256_file(weights_path) != record["weights_sha256"]:
        raise AdapterIntegrityError("weights file digest mismatch")
    expected_schema = {"dtype": "<f4", "shape": [DEPTH, WIDTH, WIDTH]}
    if hashlib.sha256(_canonical_json_bytes(expected_schema)).hexdigest() != record[
        "weights_schema_sha256"
    ]:
        raise AdapterIntegrityError("weights schema digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_metadata(manifest)
    if type(manifest.get("mlp_id")) is not int or manifest["mlp_id"] != record[
        "mlp_id"
    ]:
        raise AdapterIntegrityError("row manifest MLP identity mismatch")
    if type(manifest.get("mlp_seed")) is not int:
        raise AdapterIntegrityError("row manifest seed must be exact integer")
    file_binding = manifest.get("files", {}).get(weights_path.name, {})
    if file_binding.get("sha256") != record["weights_sha256"]:
        raise AdapterIntegrityError("row manifest does not bind weights file")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.dtype("<f4"):
        raise AdapterIntegrityError("weights geometry mismatch")
    mlp = session["MLP"](
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=int(manifest["mlp_seed"]),
        name="signed-pair-bank-rank-" + str(expected_rank),
    )
    mlp.validate()
    return mlp


def _cross_rung_prefix_equal(np, action16, action64, action256):
    for sector in range(8):
        if not _bytes_equal(
            np,
            action16[:, :, sector * 2:(sector + 1) * 2],
            action256[:, :, sector * 32:sector * 32 + 2],
        ):
            return False
        if not _bytes_equal(
            np,
            action64[:, :, sector * 8:(sector + 1) * 8],
            action256[:, :, sector * 32:sector * 32 + 8],
        ):
            return False
    return True


def begin_target_blind_three_bank_pass(
    *,
    staging_directory,
    parent_module,
    literal_source_module,
    reference_source_module,
    runtime_token,
    callable_component_sha256,
    expected_ranks,
):
    require_runtime_token(runtime_token)
    if tuple(expected_ranks) != tuple(range(ROW_COUNT)):
        raise AdapterIntegrityError("adapter requires exact ranks0..687")
    required_components = {
        "runtime_token.py",
        "parent_capture.py",
        "signed_pair_literal.py",
        "signed_pair_reference.py",
        "capture_adapter.py",
    }
    if not isinstance(callable_component_sha256, dict) or set(
        callable_component_sha256
    ) != required_components:
        raise AdapterIntegrityError("callable component identity map mismatch")
    if literal_source_module.LITERAL_IMPLEMENTATION_ID != (
        "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
    ):
        raise AdapterIntegrityError("literal implementation identity mismatch")
    if reference_source_module.REFERENCE_IMPLEMENTATION_ID != (
        "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
    ):
        raise AdapterIntegrityError("reference implementation identity mismatch")
    np = importlib.import_module("numpy")
    flops = importlib.import_module("flopscope")
    domain = importlib.import_module("whestbench.domain")
    root = Path(staging_directory).resolve()
    if not root.is_dir() or any(root.iterdir()):
        raise AdapterIntegrityError("staging directory must exist and be empty")
    banks = {}
    for h in EXPECTED_H:
        path = root / ("actions_H" + str(h) + ".f32le")
        banks[str(h)] = np.memmap(
            path,
            mode="w+",
            dtype=np.dtype("<f4"),
            shape=(ROW_COUNT, 13, WIDTH, h),
            order="C",
        )
    return {
        "root": root,
        "np": np,
        "flops": flops,
        "MLP": domain.MLP,
        "parent": parent_module,
        "literal_source": literal_source_module,
        "reference_source": reference_source_module,
        "literal_implementation_id": literal_source_module.LITERAL_IMPLEMENTATION_ID,
        "reference_implementation_id": (
            reference_source_module.REFERENCE_IMPLEMENTATION_ID
        ),
        "runtime_token": runtime_token,
        "component_sha256": dict(callable_component_sha256),
        "banks": banks,
        "A15_sha256": {},
        "A15_paths": {},
        "ranks": [],
        "meters": [],
    }


def generate_one_target_blind_row(*, session, record, expected_rank, runtime_token):
    require_runtime_token(runtime_token)
    if runtime_token is not session.get("runtime_token"):
        raise AdapterIntegrityError("row token identity changed")
    if type(expected_rank) is not int or expected_rank != len(session["ranks"]):
        raise AdapterIntegrityError("row call order mismatch")
    sentinel = expected_rank in SENTINEL_RANKS
    mlp = _validate_record_and_load_mlp(
        session, record, expected_rank, runtime_token
    )
    flops = session["flops"]
    np = session["np"]
    context = flops.BudgetContext(
        flop_budget=DIAGNOSTIC_METER_CEILING,
        quiet=True,
        namespace="signed_pair_three_bank_accuracy_rank_" + str(expected_rank),
    )
    started = time.perf_counter()
    with context:
        base_before_flops = int(context.flops_used)
        base_before_calls = _operation_calls(context.summary_dict(False))
        base_remote = None
        if sentinel:
            base_remote = session["parent"].predict_base_five_components(
                mlp, runtime_token
            )
        base_after_flops = int(context.flops_used)
        base_after_calls = _operation_calls(context.summary_dict(False))

        literal_allocation_before_flops = int(context.flops_used)
        literal_allocation_before_calls = _operation_calls(context.summary_dict(False))
        literal_workspace = session["literal_source"].allocate_three_bank_workspace(
            session["parent"].fnp, runtime_token
        )
        literal_allocation_after_flops = int(context.flops_used)
        literal_allocation_after_calls = _operation_calls(context.summary_dict(False))
        reference_allocation_before_flops = int(context.flops_used)
        reference_allocation_before_calls = _operation_calls(context.summary_dict(False))
        reference_workspace = None
        if sentinel:
            reference_workspace = session[
                "reference_source"
            ].allocate_reference_workspace(session["parent"].fnp, runtime_token)
        reference_allocation_after_flops = int(context.flops_used)
        reference_allocation_after_calls = _operation_calls(context.summary_dict(False))

        proxy = _LiteralCapture(
            literal_source=session["literal_source"],
            reference_source=(session["reference_source"] if sentinel else None),
            literal_workspace=literal_workspace,
            reference_workspace=reference_workspace,
            budget=context,
            runtime_token=runtime_token,
        )
        captured_before_flops = int(context.flops_used)
        captured_before_calls = _operation_calls(context.summary_dict(False))
        captured_remote, literal_views_remote = session[
            "parent"
        ].predict_with_signed_pair_capture(
            mlp,
            proxy,
            None,
            runtime_token,
        )
        captured_after_flops = int(context.flops_used)
        captured_after_calls = _operation_calls(context.summary_dict(False))
        reference_views_remote = proxy.reference_bank_views() if sentinel else None

        base = None
        captured = None
        if sentinel:
            base = {
                name: _fetch_array(np, base_remote[name], shape)
                for name, shape in COMPONENT_SHAPES.items()
            }
            captured = {
                name: _fetch_array(np, captured_remote[name], shape)
                for name, shape in COMPONENT_SHAPES.items()
            }

        literal_h256_remote, literal_zero256_remote = literal_views_remote["H256"]
        literal_action256 = _fetch_array(
            np, literal_h256_remote, (13, WIDTH, 256)
        )
        literal_zero256 = _fetch_array(np, literal_zero256_remote, (WIDTH, 256))
        literal_actions = {
            "16": _derive_narrow_from_h256(np, literal_action256, 16),
            "64": _derive_narrow_from_h256(np, literal_action256, 64),
            "256": literal_action256,
        }
        literal_zeros = {
            "16": _derive_narrow_from_h256(np, literal_zero256, 16),
            "64": _derive_narrow_from_h256(np, literal_zero256, 64),
            "256": literal_zero256,
        }
        actual_literal_actions = {}
        actual_literal_zeros = {}
        reference_actions = {}
        reference_zeros = {}
        if sentinel:
            for h in EXPECTED_H:
                key = "H" + str(h)
                literal_tape, literal_zero = literal_views_remote[key]
                reference_tape, reference_zero = reference_views_remote[key]
                actual_literal_actions[str(h)] = _fetch_array(
                    np, literal_tape, (13, WIDTH, h)
                )
                actual_literal_zeros[str(h)] = _fetch_array(
                    np, literal_zero, (WIDTH, h)
                )
                reference_actions[str(h)] = _fetch_array(
                    np, reference_tape, (13, WIDTH, h)
                )
                reference_zeros[str(h)] = _fetch_array(
                    np, reference_zero, (WIDTH, h)
                )
    wall_seconds = float(time.perf_counter() - started)
    summary = context.summary_dict(by_namespace=False)
    complete_calls = _operation_calls(summary)

    if proxy.capture_count != 13:
        raise AdapterIntegrityError("literal source hook did not execute 13 times")
    if proxy.reference_enabled is not sentinel:
        raise AdapterIntegrityError("reference execution sentinel mismatch")

    component_hashes = None
    parent_comparator = None
    source_comparator = None
    if sentinel:
        component_hashes = {"base": {}, "capture": {}}
        for name in COMPONENT_SHAPES:
            component_hashes["base"][name] = _sha256_array(base[name])
            component_hashes["capture"][name] = _sha256_array(captured[name])
            if not _bytes_equal(np, base[name], captured[name]):
                raise AdapterIntegrityError(
                    "base/capture parent component bytes differ: " + name
                )
        for components, label in ((base, "base"), (captured, "capture")):
            recomputed = np.maximum(
                components["host"] + components["native_j_correction"],
                np.float32(0.0),
            ).astype(np.dtype("<f4"), copy=False)
            if not _bytes_equal(
                np, recomputed, components["final_public_prediction"]
            ):
                raise AdapterIntegrityError(
                    label + " final component recompute failed"
                )
        parent_comparator = {
            "comparator": "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE",
            "component_names": list(COMPONENT_SHAPES),
            "base_component_sha256": component_hashes["base"],
            "capture_component_sha256": component_hashes["capture"],
            "all_five_component_bytes_equal": True,
        }

        source_hashes = {"literal": {}, "reference": {}}
        for h in EXPECTED_H:
            key = str(h)
            if not _bytes_equal(
                np, literal_actions[key], actual_literal_actions[key]
            ):
                raise AdapterIntegrityError(
                    "H256-derived/literal action bytes differ: H" + key
                )
            if not _bytes_equal(
                np, literal_zeros[key], actual_literal_zeros[key]
            ):
                raise AdapterIntegrityError(
                    "H256-derived/literal A15 bytes differ: H" + key
                )
            source_hashes["literal"][key] = {
                "action_sha256": _sha256_array(actual_literal_actions[key]),
                "A15_zero_sha256": _sha256_array(actual_literal_zeros[key]),
            }
            source_hashes["reference"][key] = {
                "action_sha256": _sha256_array(reference_actions[key]),
                "A15_zero_sha256": _sha256_array(reference_zeros[key]),
            }
            if not _bytes_equal(
                np, actual_literal_actions[key], reference_actions[key]
            ):
                raise AdapterIntegrityError(
                    "literal/reference action bytes differ: H" + key
                )
            if not _bytes_equal(
                np, actual_literal_zeros[key], reference_zeros[key]
            ):
                raise AdapterIntegrityError(
                    "literal/reference A15 bytes differ: H" + key
                )
        source_comparator = {
            "comparator": "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE",
            "literal_implementation_id": session["literal_implementation_id"],
            "reference_implementation_id": session["reference_implementation_id"],
            "literal_source_sha256": session["component_sha256"][
                "signed_pair_literal.py"
            ],
            "reference_source_sha256": session["component_sha256"][
                "signed_pair_reference.py"
            ],
            "literal": source_hashes["literal"],
            "reference": source_hashes["reference"],
            "all_action_and_A15_bytes_equal": True,
        }
    if not _cross_rung_prefix_equal(
        np,
        literal_actions["16"],
        literal_actions["64"],
        literal_actions["256"],
    ):
        raise AdapterIntegrityError("cross-rung prefix bytes differ")

    # Accuracy bytes are copied before deployment-only resource classification.
    row_banks = {}
    for h in EXPECTED_H:
        key = str(h)
        action = literal_actions[key]
        zero = literal_zeros[key]
        action_hash = _sha256_array(action)
        zero_hash = _sha256_array(zero)
        layer_hashes = [_sha256_array(action[slot]) for slot in range(13)]
        session["banks"][key][expected_rank] = action
        if expected_rank == 0:
            zero_path = session["root"] / ("A15_zero_H" + key + ".f32le")
            with zero_path.open("xb") as handle:
                handle.write(memoryview(zero).cast("B"))
                handle.flush()
                os.fsync(handle.fileno())
            session["A15_sha256"][key] = zero_hash
            session["A15_paths"][key] = zero_path
        elif zero_hash != session["A15_sha256"][key]:
            raise AdapterIntegrityError("row A15 differs from canonical H buffer")
        row_banks[key] = {
            "shape": [13, WIDTH, h],
            "dtype": "float32",
            "finite": True,
            "action_bytes_sha256": action_hash,
            "layer_action_bytes_sha256": layer_hashes,
            "A15_zero_bytes_sha256": zero_hash,
        }

    exact_meters = {
        "complete_flops": int(summary["flops_used"]),
        "complete_calls": int(complete_calls),
        "base_parent_flops": int(base_after_flops - base_before_flops),
        "base_parent_calls": int(base_after_calls - base_before_calls),
        "capture_parent_flops": int(captured_after_flops - captured_before_flops),
        "capture_parent_calls": int(captured_after_calls - captured_before_calls),
        "literal_source_hook_flops": int(proxy.literal_flops),
        "literal_source_hook_calls": int(proxy.literal_calls),
        "reference_source_hook_flops": int(proxy.reference_flops),
        "reference_source_hook_calls": int(proxy.reference_calls),
        "literal_workspace_flops": int(
            literal_allocation_after_flops - literal_allocation_before_flops
        ),
        "literal_workspace_calls": int(
            literal_allocation_after_calls - literal_allocation_before_calls
        ),
        "reference_workspace_flops": int(
            reference_allocation_after_flops - reference_allocation_before_flops
        ),
        "reference_workspace_calls": int(
            reference_allocation_after_calls - reference_allocation_before_calls
        ),
        "peak_rss_bytes": int(_resident_bytes()),
        "wall_seconds": wall_seconds,
        "residual_seconds": float(summary["residual_wall_time_s"]),
        "source_hook_capture_count": int(proxy.capture_count),
        "reference_source_hook_capture_count": (
            int(proxy.capture_count) if sentinel else 0
        ),
        "meter_scope": "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS",
    }
    if any(
        type(value) is not int or value < 0
        for name, value in exact_meters.items()
        if name not in {"wall_seconds", "residual_seconds", "meter_scope"}
    ):
        raise AdapterIntegrityError("invalid exact integer meter")
    if not all(
        type(value) is float and math.isfinite(value) and value >= 0.0
        for value in (
            exact_meters["wall_seconds"],
            exact_meters["residual_seconds"],
        )
    ):
        raise AdapterIntegrityError("nonfinite wall or residual meter")

    # These observations are deliberately computed after bank preservation and
    # are never used by a branch that raises, retries, skips, or mutates banks.
    resource_observation = {
        "classification": "OBSERVATIONAL_DEPLOYMENT_ONLY",
        "deployment_flop_cap": DEPLOYMENT_FLOP_CAP,
        "complete_flops_within_deployment_cap": (
            exact_meters["complete_flops"] <= DEPLOYMENT_FLOP_CAP
        ),
        "peak_rss_bytes": exact_meters["peak_rss_bytes"],
        "wall_seconds": exact_meters["wall_seconds"],
        "residual_seconds": exact_meters["residual_seconds"],
        "accuracy_veto_allowed": False,
    }
    session["ranks"].append(expected_rank)
    session["meters"].append(exact_meters)
    if (expected_rank + 1) % MEMMAP_FLUSH_INTERVAL_ROWS == 0:
        for tape in session["banks"].values():
            tape.flush()
    return {
        "rank": expected_rank,
        "sentinel_rank": sentinel,
        "comparator_mode": (
            "SENTINEL_DUAL_PARITY" if sentinel else "LEAN_CAPTURE_LITERAL_ONLY"
        ),
        "mlp_id": record["mlp_id"],
        "row_manifest_sha256": record["row_manifest_sha256"],
        "weights_sha256": record["weights_sha256"],
        "weights_schema_sha256": record["weights_schema_sha256"],
        "capture_layers_zero_based": list(CAPTURE_LAYERS),
        "same_layer_preactivation_alpha": True,
        "capture_after_full_reset_before_reset_stars": True,
        "A15_exact_zero_float32": all(
            not bool(np.any(value)) for value in literal_zeros.values()
        ),
        "cross_rung_H16_prefix_byte_identical": True,
        "h256_prefix_derivation": {
            "method": "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256",
            "source_h": 256,
            "derived_h": [16, 64],
            "all_derived_bytes_match_written_row": True,
            "sentinel_derived_matches_literal_and_reference": (
                True if sentinel else None
            ),
        },
        "source_action_comparator": source_comparator,
        "five_component_parent_comparator": parent_comparator,
        "banks": row_banks,
        "exact_meters": exact_meters,
        "resource_observation": resource_observation,
    }


def finish_target_blind_three_bank_pass(*, session, runtime_token):
    require_runtime_token(runtime_token)
    if runtime_token is not session.get("runtime_token"):
        raise AdapterIntegrityError("finish token identity changed")
    if session["ranks"] != list(range(ROW_COUNT)):
        raise AdapterIntegrityError("cannot finish incomplete or reordered bank")
    banks = {}
    for h in EXPECTED_H:
        key = str(h)
        tape = session["banks"][key]
        tape.flush()
        tape_path = Path(tape.filename).resolve()
        with tape_path.open("r+b", buffering=0) as handle:
            os.fsync(handle.fileno())
        zero_path = session["A15_paths"][key]
        banks[key] = {
            "path": tape_path.name,
            "shape": [ROW_COUNT, 13, WIDTH, h],
            "dtype": "float32",
            "finite": True,
            "sha256": _sha256_file(tape_path),
            "A15_zero_path": zero_path.name,
            "A15_zero_sha256": _sha256_file(zero_path),
        }
    meter_names = tuple(
        name for name in session["meters"][0] if name != "meter_scope"
    )
    exact_meters = {
        name + "_by_rank": [row[name] for row in session["meters"]]
        for name in meter_names
    }
    directory_fd = os.open(str(session["root"]), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "row_count": ROW_COUNT,
        "ranks": list(session["ranks"]),
        "sentinel_ranks": list(SENTINEL_RANKS),
        "banks": banks,
        "exact_meters": exact_meters,
        "cross_rung_H16_prefix_byte_identical": True,
        "h256_prefix_derivation_complete_all_rows": True,
        "literal_reference_bytes_equal_sentinels": True,
        "base_capture_five_components_equal_sentinels": True,
        "resource_policy": "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY",
    }
