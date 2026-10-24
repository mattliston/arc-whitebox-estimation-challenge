#!/usr/bin/env python3
"""Create the CP4-cap768 arm by one exact literal edit of cap1024."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_cp4_cap1024_card_v1/estimator.py"
DESTINATION = HERE / "estimator_cap768.py"
SOURCE_SHA256 = "7a1f2e3f3a51ae3c44cc4717c56fd70d0f6bef11c4b2fabf27f3937de6900e0e"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    source = SOURCE.read_bytes()
    if sha256_bytes(source) != SOURCE_SHA256:
        raise AssertionError("cap1024 source changed")
    old = b"_CP4_CAP = 1024"
    new = b"_CP4_CAP = 768 "
    if len(old) != len(new) or source.count(old) != 1:
        raise AssertionError("CP4 cap anchor changed")
    destination = source.replace(old, new, 1)
    if destination.replace(new, old, 1) != source:
        raise AssertionError("cap768 is not an exact one-literal descendant")
    DESTINATION.write_bytes(destination)
    print(
        {
            "source_sha256": sha256_bytes(source),
            "destination_sha256": sha256_bytes(destination),
            "byte_length_preserved": len(destination) == len(source),
        }
    )


if __name__ == "__main__":
    main()
