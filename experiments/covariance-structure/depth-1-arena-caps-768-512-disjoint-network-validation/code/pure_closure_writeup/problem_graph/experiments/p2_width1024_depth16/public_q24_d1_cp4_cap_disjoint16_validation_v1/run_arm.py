#!/usr/bin/env python3
"""Run one isolated target-free D1/cap arm on the frozen 16-panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_RUNNER = P2 / "public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py"
ROSTER_AUTHORITY = P2 / "public_q24_cp4_cap_disjoint16_validation_v1/PREREGISTRATION.json"
BASE_RUNNER_SHA256 = "b4fe8329e81b91db4670b3c5d62d7abb5ead0697d7f60d345164673c1621bdb8"
ROSTER_AUTHORITY_SHA256 = "2cbaedbf506d345234e744ca1e5521faf6823a922659ea177f04781e847a6c51"
PREREGISTRATION_SHA256 = "aa93dc769d3388512a6262d6ff77b10438b9796c183369ec6a946b19e40477fd"
SOURCES = {
    "d1": (
        P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py",
        "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e",
    ),
    "cap768": (
        HERE / "estimator_cap768.py",
        "5e631aea5fcfba83ccae15a97f820c82762a55b70400f61fbf4ef9e225216a11",
    ),
    "cap512": (
        HERE / "estimator_cap512.py",
        "bfae86342955e8a4795a0e4de974d607057b81466509d6f227bc89d5f392f0c7",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(BASE_RUNNER) != BASE_RUNNER_SHA256:
        raise AssertionError("audited isolated runner changed")
    if sha256(ROSTER_AUTHORITY) != ROSTER_AUTHORITY_SHA256:
        raise AssertionError("frozen roster authority changed")
    preregistration = HERE / "PREREGISTRATION.json"
    if sha256(preregistration) != PREREGISTRATION_SHA256:
        raise AssertionError("D1 panel preregistration changed")
    roster = json.loads(ROSTER_AUTHORITY.read_text(encoding="utf-8"))["roster"]
    feature_store = (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit"
    )
    fixtures = {
        int(row["role_rank"]): (
            feature_store / f"{int(row['role_rank']):04d}/weights.npy",
            str(row["weights_sha256"]),
        )
        for row in roster
    }
    spec = importlib.util.spec_from_file_location("d1_cp4_disjoint16_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.PREREGISTRATION = preregistration
    module.PREREGISTRATION_SHA256 = PREREGISTRATION_SHA256
    module.SOURCES = SOURCES
    module.FIXTURES = fixtures
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
