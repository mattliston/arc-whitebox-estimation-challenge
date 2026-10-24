#!/usr/bin/env python3
"""Build exact integrated G11B CP4-cap descendants for disjoint validation."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_q24_d1_alpha075_qhead_bank_v1/estimator.py"
SOURCE_SHA256 = "a6e332c5372839409340b100484d81e423f23239849e0c9de11b72669f9367ea"
CAPS = (768, 512)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    source = SOURCE.read_bytes()
    if digest(source) != SOURCE_SHA256:
        raise AssertionError("G11B source changed")
    old = b"_CP4_CAP = 2048"
    if source.count(old) != 1:
        raise AssertionError("G11B CP4 cap anchor changed")
    results = {}
    for cap in CAPS:
        path = HERE / f"estimator_cap{cap}.py"
        if path.exists():
            raise FileExistsError(path)
        new = f"_CP4_CAP = {cap}".encode("ascii").ljust(len(old), b" ")
        if len(new) != len(old):
            raise AssertionError("literal width changed")
        candidate = source.replace(old, new, 1)
        if candidate.replace(new, old, 1) != source:
            raise AssertionError(f"cap{cap} is not an exact one-literal descendant")
        path.write_bytes(candidate)
        results[str(cap)] = digest(candidate)
    print({"source_sha256": digest(source), "destinations": results})


if __name__ == "__main__":
    main()
