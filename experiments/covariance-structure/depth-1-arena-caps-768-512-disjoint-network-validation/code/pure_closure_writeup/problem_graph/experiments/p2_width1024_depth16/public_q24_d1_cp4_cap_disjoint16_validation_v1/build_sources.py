#!/usr/bin/env python3
"""Build exact one-literal CP4 cap descendants of banked D1 #328586."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
SOURCE_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
CAPS = (768, 512)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    source = SOURCE.read_bytes()
    if digest(source) != SOURCE_SHA256:
        raise AssertionError("banked D1 source changed")
    old = b"_CP4_CAP = 2048"
    if source.count(old) != 1:
        raise AssertionError("D1 CP4 cap anchor changed")
    results = {}
    for cap in CAPS:
        path = HERE / f"estimator_cap{cap}.py"
        if path.exists():
            raise FileExistsError(path)
        new = f"_CP4_CAP = {cap}".encode("ascii").ljust(len(old), b" ")
        candidate = source.replace(old, new, 1)
        if len(candidate) != len(source) or candidate.replace(new, old, 1) != source:
            raise AssertionError(f"cap{cap} is not an exact one-literal descendant")
        path.write_bytes(candidate)
        results[str(cap)] = digest(candidate)
    print({"source_sha256": digest(source), "destinations": results})


if __name__ == "__main__":
    main()
