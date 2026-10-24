#!/Users/matt/anaconda3/bin/python3
"""Mechanical Darwin RSS-unit wrapper around the frozen V1 experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys


HERE = Path(__file__).resolve().parent
FROZEN = HERE / "run_experiment.py"
AMENDMENT = HERE / "MECHANICAL_AMENDMENT_V2.json"
SEAL = HERE / "MECHANICAL_SEAL_V2.json"
FROZEN_SHA256 = "955b24cfebaf59fae21b4b87fb9e17883b9f1b4ec9719a2a1d49922960111f57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> None:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-before-any-cv-metric":
        raise AssertionError("mechanical seal status changed")
    if sha256(FROZEN) != FROZEN_SHA256:
        raise AssertionError("frozen V1 runner changed")
    if seal.get("v1_program_sha256") != FROZEN_SHA256:
        raise AssertionError("V1 program binding changed")
    if seal.get("amendment_sha256") != sha256(AMENDMENT):
        raise AssertionError("mechanical amendment changed")
    if seal.get("wrapper_sha256") != sha256(Path(__file__)):
        raise AssertionError("mechanical wrapper changed")


def corrected_resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> None:
    verify()
    specification = importlib.util.spec_from_file_location("cp1536_rff_frozen_v1", FROZEN)
    if specification is None or specification.loader is None:
        raise ImportError(FROZEN)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.resident_bytes = corrected_resident_bytes
    module.CACHE = HERE / "cache_v2"
    module.RESULT = HERE / "CV_RESULTS_V2.json"
    module.main()


if __name__ == "__main__":
    main()
