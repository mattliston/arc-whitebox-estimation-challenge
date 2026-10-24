#!/usr/bin/env python3
"""Mechanically build the D1-push/direct-read hybrid without loading data."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D1_SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
INCUMBENT_SOURCE = P2 / "public_noeigh_card_v1/estimator.py"
DESTINATION = HERE / "estimator.py"

EXPECTED_D1_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
EXPECTED_INCUMBENT_SHA256 = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
EXPECTED_CANDIDATE_SHA256 = "bd479793b7c7d1390c5a9bafce07c041dd4824a7280ba48d636f2f8267ecdce8"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def function_block(text: str, name: str, next_name: str) -> str:
    start = text.index(f"def {name}(")
    stop = text.index(f"def {next_name}(", start)
    return text[start:stop]


def build() -> bytes:
    d1_bytes = D1_SOURCE.read_bytes()
    incumbent_bytes = INCUMBENT_SOURCE.read_bytes()
    if digest(d1_bytes) != EXPECTED_D1_SHA256:
        raise RuntimeError("frozen D1 source changed")
    if digest(incumbent_bytes) != EXPECTED_INCUMBENT_SHA256:
        raise RuntimeError("incumbent source changed")
    d1 = d1_bytes.decode("utf-8")
    incumbent = incumbent_bytes.decode("utf-8")

    d1 = replace_once(
        d1,
        function_block(d1, "_cp_read", "_cp_k3"),
        function_block(incumbent, "_cp_read", "_cp_k3"),
        "restore incumbent direct _cp_read",
    )
    read_specs = next(
        line for line in d1.splitlines(keepends=True)
        if line.startswith("_CP3_READ_ARENA_VIEW_SPECS = ")
    )
    d1 = replace_once(d1, read_specs, "", "remove unused read view specs")
    d1 = replace_once(
        d1,
        "            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),\n",
        "",
        "remove unused read view binding",
    )
    d1 = replace_once(
        d1,
        "            cp3_context[3],\n            cp3_context[4],\n            cp3_context[7],\n",
        "            cp3_context[3],\n            cp3_context[4],\n",
        "restore direct _cp_read call",
    )
    return d1.encode("utf-8")


def main() -> None:
    candidate = build()
    observed = digest(candidate)
    if EXPECTED_CANDIDATE_SHA256 != "TO_BIND" and observed != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"candidate hash mismatch: {observed}")
    DESTINATION.write_bytes(candidate)
    print(observed)


if __name__ == "__main__":
    main()
