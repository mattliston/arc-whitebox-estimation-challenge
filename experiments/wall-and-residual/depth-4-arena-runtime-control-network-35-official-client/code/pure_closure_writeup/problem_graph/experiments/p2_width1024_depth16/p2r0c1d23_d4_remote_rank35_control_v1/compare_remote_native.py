#!/usr/bin/env python3
"""Freeze exact D4 remote-versus-native rank-35 numerical drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
REMOTE = HERE / "D23_D4_REMOTE_RANK35_OUTPUT.f32"
REMOTE_RESULT = HERE / "D23_D4_REMOTE_RANK35_CONTROL_RESULT.json"
SERVER_TELEMETRY = HERE / "D23_SERVER_TELEMETRY.json"
NATIVE = (
    P2
    / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
    / "candidate_rank35_first.npy"
)
RESULT = HERE / "D23_D4_REMOTE_NATIVE_DRIFT.json"

EXPECTED = {
    REMOTE: "fcf0279326ee6e8ae3311a0c8a777787d5d2721962a4369395e620febe61e9ad",
    REMOTE_RESULT: "c432588f3ed5123c10d2aa36b393f4335fcbfc1a183339c6cd672df91aac488e",
    SERVER_TELEMETRY: "9685ac1a852bfbeb82201732fee8d7e98f6e2bf636feba38e1eb85190ca1f777",
    NATIVE: "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    for path, expected in EXPECTED.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))
    remote_result = json.loads(REMOTE_RESULT.read_text())
    if (
        remote_result.get("status") != "REMOTE_RETURNED_OUTPUT"
        or remote_result.get("exception") is not None
        or remote_result.get("context_exception") is not None
        or remote_result["output"]["raw_output_file_sha256"] != EXPECTED[REMOTE]
    ):
        raise AssertionError("D23 remote result is not the frozen successful control")

    remote = np.fromfile(REMOTE, dtype=np.dtype("<f4")).reshape(16, 1024)
    native = np.load(NATIVE, allow_pickle=False)
    if remote.shape != native.shape or remote.dtype != native.dtype:
        raise AssertionError((remote.shape, native.shape, remote.dtype, native.dtype))
    delta = remote.astype(np.float64) - native.astype(np.float64)
    absolute = np.abs(delta)
    unequal = remote != native
    payload = {
        "schema": "p2r0c1-d4-remote-native-rank35-drift-v1",
        "status": "PASS_QUANTIFIED_GENERIC_REMOTE_DRIFT",
        "shape": list(remote.shape),
        "dtype": remote.dtype.str,
        "byte_identical": bool(np.array_equal(remote, native)),
        "remote_array_sha256": array_sha256(remote),
        "native_array_sha256": array_sha256(native),
        "unequal_element_count": int(np.count_nonzero(unequal)),
        "total_element_count": int(remote.size),
        "unequal_fraction": float(np.count_nonzero(unequal) / remote.size),
        "maximum_absolute_delta": float(np.max(absolute)),
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "mean_absolute_delta": float(np.mean(absolute)),
        "per_layer_maximum_absolute_delta": [
            float(np.max(absolute[layer])) for layer in range(16)
        ],
        "per_layer_rms_delta": [
            float(np.sqrt(np.mean(delta[layer] * delta[layer])))
            for layer in range(16)
        ],
        "remote_finite": bool(np.all(np.isfinite(remote))),
        "native_finite": bool(np.all(np.isfinite(native))),
        "bindings": {
            "comparator_sha256": sha256(Path(__file__)),
            "remote_raw_sha256": sha256(REMOTE),
            "remote_result_sha256": sha256(REMOTE_RESULT),
            "server_telemetry_sha256": sha256(SERVER_TELEMETRY),
            "native_npy_sha256": sha256(NATIVE),
        },
    }
    temporary = RESULT.with_name(f".{RESULT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULT)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
