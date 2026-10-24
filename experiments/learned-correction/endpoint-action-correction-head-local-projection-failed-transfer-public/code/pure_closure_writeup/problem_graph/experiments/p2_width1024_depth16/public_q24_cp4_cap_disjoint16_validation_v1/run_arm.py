#!/usr/bin/env python3
"""Run one isolated target-free integrated arm on the frozen 16-panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_RUNNER = P2 / "public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py"
BASE_RUNNER_SHA256 = "b4fe8329e81b91db4670b3c5d62d7abb5ead0697d7f60d345164673c1621bdb8"
PREREGISTRATION_SHA256 = "2cbaedbf506d345234e744ca1e5521faf6823a922659ea177f04781e847a6c51"
SOURCES = {
    "g11b": (
        P2 / "public_q24_d1_alpha075_qhead_bank_v1/estimator.py",
        "a6e332c5372839409340b100484d81e423f23239849e0c9de11b72669f9367ea",
    ),
    "cap768": (
        HERE / "estimator_cap768.py",
        "91dce2514914c8b9b930f709e5c87185ce35197ebcfc4b3d4b1815f22b5065ed",
    ),
    "cap512": (
        HERE / "estimator_cap512.py",
        "b8ce15b72ad68356b5f46cfd56ee118afa3e8d1e911815a5d5370a88868f159c",
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
    preregistration = HERE / "PREREGISTRATION.json"
    if sha256(preregistration) != PREREGISTRATION_SHA256:
        raise AssertionError("panel preregistration changed")
    record = json.loads(preregistration.read_text(encoding="utf-8"))
    fixtures = {}
    feature_store = (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit"
    )
    for row in record["roster"]:
        rank = int(row["role_rank"])
        fixtures[rank] = (
            feature_store / f"{rank:04d}/weights.npy",
            str(row["weights_sha256"]),
        )
    spec = importlib.util.spec_from_file_location("cp4_disjoint16_isolated_base", BASE_RUNNER)
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
