#!/usr/bin/env python3
"""Build the literal D4 + alpha=.75 J + promoted full688 q-head descendant.

The payload/runtime splicing law is reused from the independently audited D1
integration builder.  This wrapper pins that builder, the passed D4 source,
the promoted refit result/model, and changes no scientific parameter.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUTHORITY = P2 / "public_q24_d1_alpha075_qhead_bank_v1/build_candidate.py"
D4_SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
QHEAD_MODEL = P2 / "public_q24_qhead_full688_refit_v1/FULL688_Q_SUFFIX_HEAD.pt"
REFIT_RESULT = P2 / "public_q24_qhead_full688_refit_v1/RESULT.json"
DESTINATION = HERE / "estimator.py"

EXPECTED = {
    AUTHORITY: "a364ee45ad99de0fc227ac3cede521d2e33318449fed7a04767436ad856c747f",
    D4_SOURCE: "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e",
    QHEAD_MODEL: "d57407e0534e0be8b29a8b3f8bb80f617e83671584ddb32af757a3631e0fc58f",
    REFIT_RESULT: "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25",
}


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
    spec = importlib.util.spec_from_file_location("d4_full688_builder_authority", AUTHORITY)
    if spec is None or spec.loader is None:
        raise ImportError(AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE = D4_SOURCE
    module.SOURCE_SHA256 = EXPECTED[D4_SOURCE]
    module.QHEAD_MODEL = QHEAD_MODEL
    module.QHEAD_MODEL_SHA256 = EXPECTED[QHEAD_MODEL]
    module.DESTINATION = DESTINATION
    module.NEW_PREDICT = module.NEW_PREDICT.replace(
        "D1 plus alpha=.75 J/q-head", "D4 plus alpha=.75 J/full688 q-head"
    )
    module.main()


if __name__ == "__main__":
    main()
