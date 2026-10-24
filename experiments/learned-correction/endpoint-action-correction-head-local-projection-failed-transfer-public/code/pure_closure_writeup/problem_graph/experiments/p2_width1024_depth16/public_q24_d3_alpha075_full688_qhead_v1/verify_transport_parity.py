#!/usr/bin/env python3
"""Verify literal D3 transport against the frozen independent PyTorch authority."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUTHORITY = P2 / "public_q24_d1_alpha075_qhead_bank_v1/verify_transport_parity.py"
CANDIDATE = HERE / "estimator.py"
INCUMBENT = P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py"
Q_ARTIFACT = P2 / "public_q24_qhead_full688_refit_v1/FULL688_Q_SUFFIX_HEAD.pt"
TORCH_AUTHORITY_DIR = P2 / "public_q24_d4_alpha075_full688_qhead_v1"
AUTHORITY_SHA256 = "a55510f885cc66c90b6509c8d8d19302f4dc266be1821f9b701ad4cf99f5320c"
CANDIDATE_SHA256 = "25f0a0df7a1910e94ff6d4a9947d7381a0e907a90b6bf24a2fef5013c8d689af"
INCUMBENT_SHA256 = "1771a616c86507ce1aad75dced22be57caddb93f5121034c5022d4a11c62c8c4"
Q_ARTIFACT_SHA256 = "d57407e0534e0be8b29a8b3f8bb80f617e83671584ddb32af757a3631e0fc58f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, wanted in {
        AUTHORITY: AUTHORITY_SHA256,
        CANDIDATE: CANDIDATE_SHA256,
        INCUMBENT: INCUMBENT_SHA256,
        Q_ARTIFACT: Q_ARTIFACT_SHA256,
    }.items():
        observed = sha256(path)
        if observed != wanted:
            raise AssertionError((str(path), observed, wanted))
    spec = importlib.util.spec_from_file_location("d3_full688_transport_verifier", AUTHORITY)
    if spec is None or spec.loader is None:
        raise ImportError(AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CANDIDATE = CANDIDATE
    module.INCUMBENT = INCUMBENT
    module.Q_ARTIFACT = Q_ARTIFACT
    module.OUTPUT = HERE / "TRANSPORT_PARITY_RESULT.json"
    module.AUTHORITY_ARRAYS = TORCH_AUTHORITY_DIR / "TORCH_AUTHORITY_RANK35.npz"
    module.AUTHORITY_MANIFEST = TORCH_AUTHORITY_DIR / "TORCH_AUTHORITY_RANK35.json"
    module.EXPECTED = dict(module.EXPECTED)
    module.EXPECTED.update(
        candidate_sha256=CANDIDATE_SHA256,
        incumbent_sha256=INCUMBENT_SHA256,
        q_artifact_sha256=Q_ARTIFACT_SHA256,
    )
    module.main()


if __name__ == "__main__":
    main()
