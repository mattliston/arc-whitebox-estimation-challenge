#!/usr/bin/env python3
"""D4 rank-scope wrapper around the sealed D3 integrated host helper."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "public_cp16384_dynamic_multistate_pilot_v1/generate_host_row.py"
BASE_SHA256 = "f9e39975c3892adefceb2d749536d65b9e9d5216cf4e7148195d41f9225044a8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(BASE) != BASE_SHA256:
        raise AssertionError("sealed D3 integrated host helper changed")
    spec = importlib.util.spec_from_file_location("p2r0c1d4_base_host_helper", BASE)
    if spec is None or spec.loader is None:
        raise ImportError(BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.PILOT_ROWS != 200:
        raise AssertionError("D3 helper rank ceiling changed")
    module.PILOT_ROWS = 400
    module.main()


if __name__ == "__main__":
    main()
