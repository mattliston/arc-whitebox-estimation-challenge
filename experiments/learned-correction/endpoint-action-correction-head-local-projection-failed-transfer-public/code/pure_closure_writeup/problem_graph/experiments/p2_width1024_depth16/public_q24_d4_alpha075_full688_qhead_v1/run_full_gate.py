#!/usr/bin/env python3
"""One-shot target-free [35,279,35] gate for D4 + promoted full688 q-head."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUTHORITY = P2 / "public_q24_d1_alpha075_qhead_bank_v1/run_full_gate.py"
SOURCE = HERE / "estimator.py"
BUILDER = HERE / "build_candidate.py"
GENERATOR = HERE / "generate_expected_outputs.py"
PREREGISTRATION = HERE / "FULL_GATE_PREREGISTRATION.json"
PARITY = HERE / "TRANSPORT_PARITY_RESULT.json"
EXPECTED_DIR = HERE / "expected_outputs"
RESULT = HERE / "FULL_GATE_RESULT.json"
OUTPUT_DIR = HERE / "integrated_outputs"
REFIT_RESULT = P2 / "public_q24_qhead_full688_refit_v1/RESULT.json"
D4_RESULT = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/D4_CANDIDATE_RESULT.json"

EXPECTED = {
    AUTHORITY: "1a8f2db58e46ad047523967e888744eacaa3bfd4f274be44d4fe217696a961d6",
    SOURCE: "b37047ad5b19f309110c7f6af40320aeec3f3286c27d537b8cb455ff79f89a97",
    BUILDER: "0f46d07ef51fc2a92ebff45feb5624c8a007c02379e3b122603d9a507a9bbf55",
    GENERATOR: "7c5bb44fa8c8c89eae4a4170c296fe82c4613eed67aa8f8dcf78e4f56f63d1d4",
    PREREGISTRATION: "55c061b34d2e63698c54c34ef3a5de2e101746e6f43949ba768eb8121ec9f82c",
    PARITY: "fe3138f5d99ef420f12b78c5850c87be9ffcf1bddc83a156719898128dae061b",
    REFIT_RESULT: "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25",
    D4_RESULT: "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1",
}
EXPECTED_MANIFEST_SHA256 = "2592bf5d55d21e6a42dfdd55935bd17b638f1204744bd021cc28ebcf87aad068"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path, wanted in EXPECTED.items():
        observed = sha256(path)
        if observed != wanted:
            raise AssertionError((str(path), observed, wanted))
    manifest = EXPECTED_DIR / "MANIFEST.json"
    if sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("expected-output manifest changed")
    spec = importlib.util.spec_from_file_location("d4_full688_full_gate_authority", AUTHORITY)
    if spec is None or spec.loader is None:
        raise ImportError(AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE = SOURCE
    module.BUILDER = BUILDER
    module.GENERATOR = GENERATOR
    module.PREREGISTRATION = PREREGISTRATION
    module.PARITY = PARITY
    module.EXPECTED_DIR = EXPECTED_DIR
    module.RESULT = RESULT
    module.OUTPUT_DIR = OUTPUT_DIR
    module.SOURCE_SHA256 = EXPECTED[SOURCE]
    module.BUILDER_SHA256 = EXPECTED[BUILDER]
    module.GENERATOR_SHA256 = EXPECTED[GENERATOR]
    module.PREREGISTRATION_SHA256 = EXPECTED[PREREGISTRATION]
    module.PARITY_SHA256 = EXPECTED[PARITY]
    module.EXPECTED_MANIFEST_SHA256 = EXPECTED_MANIFEST_SHA256
    module.INCUMBENT_REFERENCES = (
        (
            P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/candidate_rank35_first.npy",
            "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
        ),
        (
            P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/candidate_rank279.npy",
            "1586d3635eb865600b10418fffa61c89fbcad791e32c49e50fcf05fa783fd125",
        ),
        (
            P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/candidate_rank35_repeat.npy",
            "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
        ),
    )
    module.EXPECTED_CANDIDATE_FLOPS = (
        1_319_968_409_954,
        1_356_691_987_955,
        1_318_045_658_603,
    )
    module.EXPECTED_INCUMBENT_FLOPS = (
        1_319_871_710_583,
        1_356_595_288_584,
        1_317_948_959_232,
    )
    module.EXPECTED_INCREMENT = (96_699_371,) * 3
    module.INCUMBENT_RAW = 2.9409214512554626e-8
    module.INCUMBENT_ADJUSTED = 2.1258043062525995e-8
    module.PROTECTED_RAW_RATIO = 1.0251565972496817
    module.MAX_COMPUTE_RATIO = 0.8413057286182815
    module.PROJECTED_RAW = 3.0149050277476456e-8
    module.PROJECTED_ADJUSTED = 1.833442690852171e-8
    # Bind the wrapper which pins the promoted refit/D4 evidence and all
    # overridden literal bills, not merely the reused gate implementation.
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
