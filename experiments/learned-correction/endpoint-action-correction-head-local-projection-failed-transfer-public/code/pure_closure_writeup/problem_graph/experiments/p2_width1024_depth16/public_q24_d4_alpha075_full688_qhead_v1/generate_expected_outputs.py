#!/usr/bin/env python3
"""Generate independent weights-only expected outputs for the D4 full gate."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUTHORITY = P2 / "public_q24_d1_alpha075_qhead_bank_v1/generate_expected_outputs.py"
SOURCE = HERE / "estimator.py"
PREREGISTRATION = HERE / "FULL_GATE_PREREGISTRATION.json"
PARITY = HERE / "TRANSPORT_PARITY_RESULT.json"
OUTPUT = HERE / "expected_outputs"

EXPECTED = {
    AUTHORITY: "62aba2873cb738c75e4ef0bf1d47bc17cedfcd2c1dab6e3d44c57d88cf608a46",
    SOURCE: "b37047ad5b19f309110c7f6af40320aeec3f3286c27d537b8cb455ff79f89a97",
    PREREGISTRATION: "55c061b34d2e63698c54c34ef3a5de2e101746e6f43949ba768eb8121ec9f82c",
    PARITY: "fe3138f5d99ef420f12b78c5850c87be9ffcf1bddc83a156719898128dae061b",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, wanted in EXPECTED.items():
        observed = sha256(path)
        if observed != wanted:
            raise AssertionError((str(path), observed, wanted))
    spec = importlib.util.spec_from_file_location("d4_full688_expected_authority", AUTHORITY)
    if spec is None or spec.loader is None:
        raise ImportError(AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE = SOURCE
    module.PREREGISTRATION = PREREGISTRATION
    module.PARITY = PARITY
    module.OUTPUT = OUTPUT
    module.SOURCE_SHA256 = EXPECTED[SOURCE]
    module.PREREGISTRATION_SHA256 = EXPECTED[PREREGISTRATION]
    module.PARITY_SHA256 = EXPECTED[PARITY]
    # Bind the wrapper which pins all overridden authorities, not merely the
    # reused recurrence implementation.
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
