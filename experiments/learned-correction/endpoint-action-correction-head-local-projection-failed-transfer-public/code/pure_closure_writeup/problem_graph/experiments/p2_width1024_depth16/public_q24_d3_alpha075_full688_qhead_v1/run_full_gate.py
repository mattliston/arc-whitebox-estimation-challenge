#!/usr/bin/env python3
"""One-shot target-free [35,279,35] gate for D3 + promoted full688 q-head."""

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
D3_RESULT = P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1/D3_CANDIDATE_RESULT.json"
D4_FAILURE = P2 / "public_q24_d4_alpha075_full688_qhead_v1/FULL_GATE_RESULT.json"

EXPECTED = {
    AUTHORITY: "1a8f2db58e46ad047523967e888744eacaa3bfd4f274be44d4fe217696a961d6",
    SOURCE: "25f0a0df7a1910e94ff6d4a9947d7381a0e907a90b6bf24a2fef5013c8d689af",
    BUILDER: "8004729441bcddcad1d933756d6bb1f2be33db401433b31660e364c889ebc27e",
    GENERATOR: "ae7d085ab802de5a8f0aa9651a6fdedd404c7bc54341d2c917441422b90e1d28",
    PREREGISTRATION: "ed034d5c92b72a8ba5b6713200c34b47cf428cba0ed795c19bd309f30a364f15",
    PARITY: "ae83d1a68e6bd742a7a3dc7ec2209f281098e3fee6e4514d0e063b69facb2218",
    REFIT_RESULT: "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25",
    D3_RESULT: "31a54135d25f795cd9a463be4ac58971edc884e4a374fa0d7f51dc36f4466da3",
    D4_FAILURE: "742bef9476c84c5c6afb2d6e3dced0e3d94e2b4a1ea3aae6c5a2c71d75f5eca2",
}
EXPECTED_MANIFEST_SHA256 = "2c3c26401e6fd687323aefa2715488faf4d6354c40f0c048714103b54bc658b0"


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
    if sha256(EXPECTED_DIR / "MANIFEST.json") != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("expected-output manifest changed")
    spec = importlib.util.spec_from_file_location("d3_full688_full_gate_authority", AUTHORITY)
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
            P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1/candidate_rank35_first.npy",
            "eda68ee258e39b7b6bc164a416aa3ab5edf0f0c95dbc6689ec4f1cf14ac7f13e",
        ),
        (
            P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1/candidate_rank279.npy",
            "50eb5a162dd35ab9b16b6fa49a5b9006a2fdf40d8ddc1cd7a959063a7e99fb9e",
        ),
        (
            P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1/candidate_rank35_repeat.npy",
            "eda68ee258e39b7b6bc164a416aa3ab5edf0f0c95dbc6689ec4f1cf14ac7f13e",
        ),
    )
    module.EXPECTED_CANDIDATE_FLOPS = (
        1_387_818_002_786,
        1_425_446_354_419,
        1_386_800_025_067,
    )
    module.EXPECTED_INCUMBENT_FLOPS = (
        1_387_721_303_415,
        1_425_349_655_048,
        1_386_703_325_696,
    )
    module.EXPECTED_INCREMENT = (96_699_371,) * 3
    module.INCUMBENT_RAW = 2.9409214512554626e-8
    module.INCUMBENT_ADJUSTED = 2.1258043062525995e-8
    module.PROTECTED_RAW_RATIO = 1.0251565972496817
    module.MAX_COMPUTE_RATIO = 0.8839413768621204
    module.PROJECTED_RAW = 3.0149050277476456e-8
    module.PROJECTED_ADJUSTED = 1.9263578048034253e-8
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
