#!/usr/bin/env python3
"""Target-free isolated arm runner for the frozen adaptive CP4 bracket."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_RUNNER = P2 / "public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py"
BASE_RUNNER_SHA256 = "b4fe8329e81b91db4670b3c5d62d7abb5ead0697d7f60d345164673c1621bdb8"
PREREGISTRATION_SHA256 = "6cb74cff24f014ef5fe101c4f8ddebff6ec4eb8a083bac116237e64aaec38f35"
SOURCE_SHA256 = {
    "cap640": "5b89058711507a327a08c48a7be508d5af9b0d7c6af54b2b4df59045598be080",
    "cap512": "3246a0238f397ad7b56f37551e6263120141152ae9affd7f22e3764da7603388",
    "cap384": "b2327145d0f7e1aceba70fd7a7b39c6caa9330cc79badaa98c148bf6135fea39",
    "cap256": "5356ea650cef99d0c8efecef4c99d8762dd3b472c99f1dccfc929ae903f6249f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(BASE_RUNNER) != BASE_RUNNER_SHA256:
        raise AssertionError("audited isolated arm runner changed")
    spec = importlib.util.spec_from_file_location("cp4_adaptive_isolated_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.PREREGISTRATION = HERE / "PREREGISTRATION.json"
    module.PREREGISTRATION_SHA256 = PREREGISTRATION_SHA256
    module.SOURCES = {
        arm: (HERE / f"estimator_{arm}.py", expected)
        for arm, expected in SOURCE_SHA256.items()
    }
    # Bind receipts to this wrapper, not to the imported implementation file.
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()
