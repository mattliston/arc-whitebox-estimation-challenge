#!/usr/bin/env python3
"""Generate independent weights-only expected outputs for the D3 full gate."""

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
    SOURCE: "25f0a0df7a1910e94ff6d4a9947d7381a0e907a90b6bf24a2fef5013c8d689af",
    PREREGISTRATION: "ed034d5c92b72a8ba5b6713200c34b47cf428cba0ed795c19bd309f30a364f15",
    PARITY: "ae83d1a68e6bd742a7a3dc7ec2209f281098e3fee6e4514d0e063b69facb2218",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, wanted in EXPECTED.items():
        observed = sha256(path)
        if observed != wanted:
            raise AssertionError((str(path), observed, wanted))
    spec = importlib.util.spec_from_file_location("d3_full688_expected_authority", AUTHORITY)
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
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
