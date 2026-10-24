#!/usr/bin/env python3
"""Build exact one-literal CP4 cap arms for the adaptive bracket."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_cp4_cap1024_card_v1/estimator.py"
SOURCE_SHA256 = "7a1f2e3f3a51ae3c44cc4717c56fd70d0f6bef11c4b2fabf27f3937de6900e0e"
CAPS = (640, 512, 384, 256)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    source = SOURCE.read_bytes()
    if digest(source) != SOURCE_SHA256:
        raise AssertionError("cap1024 source changed")
    old = b"_CP4_CAP = 1024"
    if source.count(old) != 1:
        raise AssertionError("CP4 cap anchor changed")
    results = {}
    for cap in CAPS:
        path = HERE / f"estimator_cap{cap}.py"
        if path.exists():
            raise FileExistsError(path)
        new = f"_CP4_CAP = {cap}".encode("ascii").ljust(len(old), b" ")
        if len(new) != len(old):
            raise AssertionError((cap, len(new), len(old)))
        candidate = source.replace(old, new, 1)
        if candidate.replace(new, old, 1) != source:
            raise AssertionError(f"cap{cap} is not an exact one-literal descendant")
        path.write_bytes(candidate)
        results[str(cap)] = digest(candidate)
    print({"source_sha256": digest(source), "destinations": results})


if __name__ == "__main__":
    main()
