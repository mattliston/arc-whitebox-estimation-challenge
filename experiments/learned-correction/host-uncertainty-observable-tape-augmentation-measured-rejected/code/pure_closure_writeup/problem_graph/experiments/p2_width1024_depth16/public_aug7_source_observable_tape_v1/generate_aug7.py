#!/Users/matt/anaconda3/bin/python3
"""P2R0F2Q stage-1: aug7 source-observable tape generation with bit parity.

Replicates the sealed v3 cp1792 host tape loop operation-for-operation, proves
bit parity against every sealed row (raw32/host/sigma payload hashes plus all
16 entering-CP state hashes), and exports seven additional target-free
per-unit channels defined in PREREGISTRATION.json. The sealed v3 store is
never written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DESIGN = P2 / "learned_residual_observer_design"
V3_BASE = DESIGN / "v3/estimator.py"
V3_BASE_SHA = "f8d48e5fc05d4de53173302e6fd8334c46908dd659a39baaebf2b534f5bdbea6"
STORE = DESIGN / "v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT_ROOT = HERE / "rows/observer_fit"
PREREGISTRATION = HERE / "PREREGISTRATION.json"

WIDTH = 1024
DEPTH = 16
CAP = 1792
ROWS = 688
AUG_CHANNELS = 7
SIGMA3_FLOOR = 1.0e-45
SERIES_FLOOR = 2.0**-40
BOND_THRESHOLD = 0.9
FLOP_BUDGET = 2**41
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_NEW_STORAGE_BYTES = 2 * (1 << 30)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_payload_sha256(array: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def raw_array_sha256(array: Any) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(array)).cast("B")
    ).hexdigest()


def _np(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = value.view(np.ndarray) if hasattr(value, "view") else np.asarray(value)
    return np.array(array, dtype=dtype, order="C", copy=True)


def _state_hash(layer0: int, cap: int, cp_lambda: Any, cp_p: Any, cp_q: Any, retained: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"entering-cp-v1|layer={layer0}|cap={cap}".encode("ascii"))
    for name, value in (("lambda", cp_lambda), ("p", cp_p), ("q", cp_q), ("retained", retained)):
        array = np.ascontiguousarray(_np(value))
        metadata = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_base() -> Any:
    if sha256_file(V3_BASE) != V3_BASE_SHA:
        raise AssertionError("frozen v3 host base changed")
    spec = importlib.util.spec_from_file_location("aug7_frozen_v3_base", V3_BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v3 host base")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resident_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def enforce_resources(where: str) -> int:
    rss = resident_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"RSS bound exceeded at {where}: {rss}")
    storage = sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())
    if storage > MAX_NEW_STORAGE_BYTES:
        raise OSError(f"new-store storage bound exceeded at {where}: {storage}")
    return rss


def generate_row(base: Any, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    base._CP_CAP = CAP
    width = WIDTH
    analytic_identity = fnp.eye(width, dtype=fnp.float64)
    cp_identity = fnp.eye(width, dtype=fnp.float32)
    mean = fnp.zeros(width, dtype=fnp.float64)
    covariance = analytic_identity
    cp_lambda = fnp.zeros(0, dtype=fnp.float32)
    cp_p = fnp.zeros((width, 0), dtype=fnp.float32)
    cp_q = fnp.zeros((width, 0), dtype=fnp.float32)
    retained = fnp.asarray(1.0, dtype=fnp.float64)
    drop_lambda: Any | None = None
    drop_p: Any | None = None
    drop_q: Any | None = None
    raw32_rows: list[np.ndarray] = []
    host_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    aug_rows: list[np.ndarray] = []
    entering: list[str] = []
    counters = {"cap_events": 0, "layers_with_discard_channels": 0}

    for layer0 in range(DEPTH):
        weights_raw = weights[layer0]
        entering.append(_state_hash(layer0, CAP, cp_lambda, cp_p, cp_q, retained))
        weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
        weights64 = fnp.asarray(weights_raw, dtype=fnp.float64)
        pre_mean = fnp.transpose(weights64) @ mean
        pre_cov = fnp.einsum("ij,ia,jb->ab", covariance, weights64, weights64)
        pre_var = fnp.maximum(fnp.diag(pre_cov), base._VAR_FLOOR)
        sigma = fnp.sqrt(pre_var)
        alpha = pre_mean / sigma
        pdf = fnp.exp(-0.5 * alpha * alpha) / base._SQRT_2PI
        cdf = base.flops.stats.norm.cdf(alpha)
        coefficients = base._hermite_coefficients(alpha, pdf, cdf)
        gaussian_mean = sigma * coefficients[0]
        if int(cp_lambda.shape[0]) == 0:
            pushed_p, pushed_q = cp_p, cp_q
            k3 = fnp.zeros(width, dtype=fnp.float32)
        else:
            pushed_p = fnp.transpose(weights32) @ cp_p
            pushed_q = fnp.transpose(weights32) @ cp_q
            k3 = base._cp_k3(cp_lambda, pushed_p, pushed_q)
        reported = fnp.maximum(
            gaussian_mean
            + (fnp.asarray(k3, dtype=fnp.float64) / 6.0)
            * (-(alpha * pdf) / pre_var),
            0.0,
        )
        correlation = pre_cov / fnp.outer(sigma, sigma)
        exact_var = fnp.maximum(
            pre_var * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
            - gaussian_mean * gaussian_mean,
            base._VAR_FLOOR,
        )
        edges = base._edge_matrices(weights32, width)
        raw = base._raw_features(
            layer0,
            width,
            int(cp_lambda.shape[0]),
            retained,
            alpha,
            sigma,
            cdf,
            pdf,
            coefficients,
            gaussian_mean,
            exact_var,
            pre_var,
            k3,
            reported,
            correlation,
            edges,
        )
        raw64 = _np(raw, dtype=np.float64)
        raw32_rows.append(np.ascontiguousarray(raw64, dtype=np.float32))
        host_rows.append(_np(fnp.asarray(reported, dtype=fnp.float32), dtype=np.float32))
        sigma_rows.append(_np(fnp.asarray(sigma, dtype=fnp.float32), dtype=np.float32))

        # ---- aug7 channels: separate variables only, no host-state mutation ----
        sigma3 = fnp.maximum(sigma * sigma * sigma, SIGMA3_FLOOR)
        skew = fnp.asarray(k3, dtype=fnp.float64) / sigma3
        if drop_lambda is not None and int(drop_lambda.shape[0]) > 0:
            pushed_pd = fnp.transpose(weights32) @ drop_p
            pushed_qd = fnp.transpose(weights32) @ drop_q
            drop_core = pushed_pd * pushed_pd * pushed_qd
            k3d_signed = 3.0 * (drop_core @ drop_lambda)
            k3d_abs = 3.0 * (fnp.abs(drop_core) @ fnp.abs(drop_lambda))
            ch20 = fnp.asarray(k3d_signed, dtype=fnp.float64) / sigma3
            ch21 = fnp.asarray(k3d_abs, dtype=fnp.float64) / sigma3
            counters["layers_with_discard_channels"] += 1
        else:
            ch20 = fnp.zeros(width, dtype=fnp.float64)
            ch21 = fnp.zeros(width, dtype=fnp.float64)
        series_diag = fnp.zeros(width, dtype=fnp.float64)
        for order in range(1, base._HERMITE_ORDER + 1):
            series_diag = series_diag + coefficients[order] * coefficients[order]
        abs_corr = fnp.abs(correlation)
        offdiag = abs_corr - analytic_identity
        ch23 = fnp.max(offdiag, axis=1)
        ch24 = fnp.sum(
            fnp.asarray(offdiag > BOND_THRESHOLD, dtype=fnp.float64), axis=1
        ) / float(width)
        rho2 = correlation * correlation
        ch25 = (rho2 @ skew) / float(width)
        ch26 = (rho2 @ alpha) / float(width)

        # ---- original covariance-series state update, capturing rho^10 ----
        covariance = fnp.zeros((width, width), dtype=fnp.float64)
        correlation_power = correlation
        power10 = None
        for order in range(1, base._HERMITE_ORDER + 1):
            coefficient = coefficients[order]
            covariance = covariance + fnp.outer(coefficient, coefficient) * correlation_power
            if order == base._HERMITE_ORDER:
                power10 = correlation_power
            correlation_power = correlation_power * correlation
        covariance = covariance * fnp.outer(sigma, sigma)
        covariance = covariance + fnp.diag(fnp.maximum(exact_var - fnp.diag(covariance), 0.0))
        mean = gaussian_mean

        assert power10 is not None
        ch22 = (coefficients[base._HERMITE_ORDER] * (power10 @ coefficients[base._HERMITE_ORDER])) / fnp.maximum(
            series_diag, SERIES_FLOOR
        )
        aug = fnp.stack((ch20, ch21, ch22, ch23, ch24, ch25, ch26), axis=1)
        aug_rows.append(np.ascontiguousarray(_np(aug, dtype=np.float64), dtype=np.float32))

        # ---- original cp growth with discard capture ----
        if layer0 + 1 < DEPTH:
            gate32 = fnp.asarray(cdf, dtype=fnp.float32)
            pre_cov32 = fnp.asarray(pre_cov, dtype=fnp.float32)
            sigma32 = fnp.asarray(sigma, dtype=fnp.float32)
            pdf32 = fnp.asarray(pdf, dtype=fnp.float32)
            post_p = fnp.concatenate(
                [gate32[:, None] * pushed_p, gate32[:, None] * pre_cov32], axis=1
            )
            post_q = fnp.concatenate([gate32[:, None] * pushed_q, cp_identity], axis=1)
            post_lambda = fnp.concatenate([cp_lambda, pdf32 / sigma32])
            if int(post_lambda.shape[0]) <= CAP:
                kept_state = (
                    post_lambda,
                    post_p,
                    post_q,
                    fnp.asarray(1.0, dtype=fnp.float64),
                )
                drop_lambda = None
                drop_p = None
                drop_q = None
            else:
                counters["cap_events"] += 1
                p_norm2 = fnp.sum(post_p * post_p, axis=0)
                q_norm2 = fnp.sum(post_q * post_q, axis=0)
                p_dot_q = fnp.sum(post_p * post_q, axis=0)
                energy = (post_lambda * post_lambda) * (
                    3.0 * p_norm2 * p_norm2 * q_norm2
                    + 6.0 * p_norm2 * p_dot_q * p_dot_q
                )
                order_index = fnp.argsort(-energy)
                keep = order_index[:CAP]
                drop = order_index[CAP:]
                energy64 = fnp.asarray(energy, dtype=fnp.float64)
                total = fnp.sum(energy64)
                kept = fnp.sum(energy64[fnp.sort(keep)])
                fraction = fnp.where(
                    total <= base._ENERGY_EPS,
                    fnp.asarray(1.0, dtype=fnp.float64),
                    fnp.clip(kept / total, 0.0, 1.0),
                )
                kept_state = (
                    post_lambda[keep],
                    post_p[:, keep],
                    post_q[:, keep],
                    fraction,
                )
                drop_lambda = post_lambda[drop]
                drop_p = post_p[:, drop]
                drop_q = post_q[:, drop]
            reference = base._cp_grow(
                cp_lambda,
                pushed_p,
                pushed_q,
                gate32,
                pre_cov32,
                sigma32,
                pdf32,
                cp_identity,
            )
            for mine, theirs in zip(kept_state, reference[:4], strict=True):
                if not np.array_equal(_np(mine), _np(theirs)):
                    raise AssertionError(
                        f"discard-capture cp_grow diverged from frozen cp_grow at layer {layer0}"
                    )
            cp_lambda, cp_p, cp_q, retained = kept_state

    return (
        np.stack(raw32_rows),
        np.stack(host_rows),
        np.stack(sigma_rows),
        np.stack(aug_rows),
        entering,
        counters,
    )


def run(start: int, stop: int) -> None:
    flops.configure(symmetry_warnings=False)
    base = load_base()
    prereg_sha = sha256_file(PREREGISTRATION)
    program_sha = sha256_file(Path(__file__))
    manifest_root = json.loads((STORE / "manifest.json").read_text())
    if manifest_root.get("status") != "SEALED_TARGET_BLIND_FEATURE_STORE":
        raise AssertionError("v3 store is not sealed")
    entries = manifest_root["roles"]["observer_fit"]["row_manifests"]
    if len(entries) != ROWS:
        raise AssertionError("v3 row manifest coverage changed")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = 0
    for rank in range(start, stop):
        out_dir = OUT_ROOT / f"{rank:04d}"
        out_manifest = out_dir / "row_manifest.json"
        if out_manifest.is_file():
            existing = json.loads(out_manifest.read_text())
            if (
                existing.get("status") == "SEALED_ROW_AUG7"
                and existing.get("parity_pass") is True
                and existing.get("program_sha256") == program_sha
            ):
                continue
            raise AssertionError(f"rank {rank}: incompatible partial output present")
        entry = entries[rank]
        if int(entry["role_rank"]) != rank:
            raise AssertionError(f"rank {rank}: v3 manifest rank mismatch")
        v3_manifest_path = STORE / entry["path"]
        if v3_manifest_path.name != "row_manifest.json":
            raise AssertionError(f"rank {rank}: unexpected v3 manifest entry path")
        v3_row_dir = v3_manifest_path.parent
        if sha256_file(v3_manifest_path) != entry["sha256"]:
            raise AssertionError(f"rank {rank}: sealed v3 row manifest hash mismatch")
        v3_manifest = json.loads(v3_manifest_path.read_text())
        weights_path = v3_row_dir / "weights.npy"
        weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
        if weights.dtype != np.dtype("<f4") or tuple(weights.shape) != (DEPTH, WIDTH, WIDTH):
            raise AssertionError(f"rank {rank}: weights schema changed")
        if raw_array_sha256(weights) != v3_manifest["weights_raw_sha256"]:
            raise AssertionError(f"rank {rank}: weights differ from sealed attestation")
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as context:
            raw32, host, sigma, aug, entering, counters = generate_row(
                base, np.asarray(weights, dtype=np.float32)
            )
        wall = time.perf_counter() - started
        sealed = v3_manifest["hosts"]["1792"]
        parity = {
            "raw32": array_payload_sha256(raw32) == sealed["raw32_sha256"],
            "host": array_payload_sha256(host) == sealed["host_sha256"],
            "sigma": array_payload_sha256(sigma) == sealed["sigma_sha256"],
            "entering_hash_matches": sum(
                1
                for mine, theirs in zip(
                    entering, sealed["entering_cp_state_hashes"], strict=True
                )
                if mine == theirs
            ),
        }
        parity_pass = (
            parity["raw32"]
            and parity["host"]
            and parity["sigma"]
            and parity["entering_hash_matches"] == DEPTH
        )
        if not parity_pass:
            raise AssertionError(f"rank {rank}: PARITY FAILURE {parity}")
        if aug.shape != (DEPTH, WIDTH, AUG_CHANNELS) or not np.isfinite(aug).all():
            raise AssertionError(f"rank {rank}: aug channel schema violation")
        out_dir.mkdir(parents=True, exist_ok=True)
        aug_path = out_dir / "aug7_locals.npy"
        with aug_path.open("xb") as handle:
            np.save(handle, aug, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        row_manifest = {
            "schema": "public-aug7-source-observable-row-v1",
            "status": "SEALED_ROW_AUG7",
            "role": "observer_fit",
            "role_rank": rank,
            "mlp_id": int(v3_manifest["mlp_id"]),
            "v3_row_manifest_sha256": entry["sha256"],
            "weights_raw_sha256": v3_manifest["weights_raw_sha256"],
            "parity": parity,
            "parity_pass": True,
            "cap_events": counters["cap_events"],
            "layers_with_discard_channels": counters["layers_with_discard_channels"],
            "aug7_sha256": sha256_file(aug_path),
            "aug7_array_sha256": array_payload_sha256(aug),
            "aug7_shape": list(aug.shape),
            "flops_used": int(context.flops_used),
            "wall_seconds": wall,
            "max_rss_bytes": resident_bytes(),
            "targets_read": 0,
            "observer_selection_opened": 0,
            "preregistration_sha256": prereg_sha,
            "program_sha256": program_sha,
        }
        tmp = out_manifest.with_name(f".{out_manifest.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(row_manifest, indent=2, sort_keys=True) + "\n")
        tmp.replace(out_manifest)
        completed += 1
        if rank % 20 == 0 or rank == stop - 1:
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "wall_seconds": round(wall, 3),
                        "flops_used": int(context.flops_used),
                        "cap_events": counters["cap_events"],
                        "rss": enforce_resources(f"rank {rank}"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del raw32, host, sigma, aug
        gc.collect()
    print(json.dumps({"completed_new": completed, "range": [start, stop]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=ROWS)
    args = parser.parse_args()
    torch_free = True  # no torch anywhere in this generator
    assert torch_free
    run(args.start, args.stop)


if __name__ == "__main__":
    main()
