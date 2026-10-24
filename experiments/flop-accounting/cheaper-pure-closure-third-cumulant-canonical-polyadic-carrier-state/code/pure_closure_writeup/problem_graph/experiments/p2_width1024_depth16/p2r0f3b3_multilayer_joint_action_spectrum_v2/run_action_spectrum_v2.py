#!/usr/bin/env python3
"""Mechanical repository-root correction for the frozen v1 discriminator.

All scientific code, inputs, blocks, gates, and output construction are loaded
from the hash-bound v1 runner.  This descendant changes only path binding and
adds explicit repository/feature-store sentinels before invoking v1 ``main``.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
V1 = HERE.parent / "p2r0f3b3_multilayer_joint_action_spectrum_v1/run_action_spectrum.py"
V1_SHA256 = "25195fff173972437e7e9d8bf58c58dff4e8eb590e0f43264f4a77382235847a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256_file(V1) != V1_SHA256:
        raise RuntimeError("frozen v1 scientific runner hash mismatch")
    if not (REPO_ROOT / ".git").exists():
        raise RuntimeError(("repository sentinel absent", REPO_ROOT))
    store = (
        REPO_ROOT
        / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16"
        / "learned_residual_observer_design/v4_feature_store_v3"
        / "sealed_target_blind_feature_store_v3/rows/observer_fit"
    )
    if not (store / "0000/row_manifest.json").is_file():
        raise RuntimeError(("feature-store sentinel absent", store))
    if "pure_closure_writeup/pure_closure_writeup" in str(store):
        raise RuntimeError(("doubled writeup path", store))

    spec = importlib.util.spec_from_file_location("frozen_action_spectrum_v1", V1)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Mechanical path/output rebinding only.  Setting __file__ makes the v1
    # integrity check bind this wrapper, whose hash is frozen by the v2 prereg.
    module.__file__ = str(Path(__file__).resolve())
    module.HERE = HERE
    module.ROOT = REPO_ROOT
    module.STORE = store
    module.PREREG = HERE / "PREREGISTRATION.json"
    module.RESULT = HERE / "RESULT.json"
    module.BASIS = HERE / "OPTIMAL_ENDPOINT_BASIS_R13.npy"
    module.main()


if __name__ == "__main__":
    main()
