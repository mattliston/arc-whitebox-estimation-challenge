#!/usr/bin/env python3
"""Generate the independent target-free PyTorch recurrence authority."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUTHORITY = P2 / "public_q24_d1_alpha075_qhead_bank_v1/generate_torch_authority.py"
Q_ARTIFACT = P2 / "public_q24_qhead_full688_refit_v1/FULL688_Q_SUFFIX_HEAD.pt"
AUTHORITY_SHA256 = "ff4e875a2167dfb3a90d892a926bde8a9b19b2d8abc4750600e937d80185dcdd"
Q_ARTIFACT_SHA256 = "d57407e0534e0be8b29a8b3f8bb80f617e83671584ddb32af757a3631e0fc58f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(AUTHORITY) != AUTHORITY_SHA256:
        raise AssertionError("PyTorch authority generator changed")
    if sha256(Q_ARTIFACT) != Q_ARTIFACT_SHA256:
        raise AssertionError("promoted q-head changed")
    spec = importlib.util.spec_from_file_location("d4_full688_torch_authority", AUTHORITY)
    if spec is None or spec.loader is None:
        raise ImportError(AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.Q_ARTIFACT = Q_ARTIFACT
    module.ARRAYS = HERE / "TORCH_AUTHORITY_RANK35.npz"
    module.MANIFEST = HERE / "TORCH_AUTHORITY_RANK35.json"
    module.EXPECTED = dict(module.EXPECTED)
    module.EXPECTED["q_artifact_sha256"] = Q_ARTIFACT_SHA256
    module.main()


if __name__ == "__main__":
    main()
