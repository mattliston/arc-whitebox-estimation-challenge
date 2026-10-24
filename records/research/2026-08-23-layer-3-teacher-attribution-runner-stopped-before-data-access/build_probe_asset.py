#!/usr/bin/env python3
"""Build the frozen signed-Walsh probe; never choose columns from targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LITERAL = "p2-teacher-v2|probe32"
WIDTH = 1024
RANK = 32


def column_schedule() -> tuple[list[int], list[int]]:
    indices: list[int] = []
    signs: list[int] = []
    counter = 0
    while len(indices) < RANK:
        digest = hashlib.sha256(
            f"{LITERAL}|column|{counter}".encode("ascii")
        ).digest()
        counter += 1
        index = int.from_bytes(digest[:2], "little") % WIDTH
        if index in indices:
            continue
        indices.append(index)
        signs.append(1 if (digest[2] & 1) == 0 else -1)
    return indices, signs


def construct() -> np.ndarray:
    indices, signs = column_schedule()
    rows = np.arange(WIDTH, dtype=np.uint16)
    probe = np.empty((WIDTH, RANK), dtype="<f4")
    for column, (index, sign) in enumerate(zip(indices, signs, strict=True)):
        parity = np.fromiter(
            ((int(row) & index).bit_count() & 1 for row in rows),
            dtype=np.int8,
            count=WIDTH,
        )
        probe[:, column] = (sign * (1 - 2 * parity)).astype(np.float32) / 32.0
    return np.ascontiguousarray(probe, dtype="<f4")


def main() -> None:
    target = HERE / "probe32.f32le"
    spec_path = HERE / "PROBE_SPEC.generated.json"
    probe = construct()
    payload = probe.tobytes(order="C")
    target.write_bytes(payload)
    gram_error = float(
        np.max(
            np.abs(
                probe.astype(np.float64).T @ probe.astype(np.float64)
                - np.eye(RANK, dtype=np.float64)
            )
        )
    )
    indices, signs = column_schedule()
    spec = {
        "schema": "phase2-teacher-probe32-generated-v1",
        "literal": LITERAL,
        "shape": [WIDTH, RANK],
        "dtype": "float32-le",
        "order": "C",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "column_indices": indices,
        "column_signs": signs,
        "entry_magnitude": 0.03125,
        "maximum_orthogonality_error_float64": gram_error,
    }
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
