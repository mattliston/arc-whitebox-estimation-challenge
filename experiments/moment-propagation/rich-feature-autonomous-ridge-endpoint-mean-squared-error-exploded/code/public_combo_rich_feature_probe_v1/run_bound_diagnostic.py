#!/usr/bin/env python3
"""Run the existing read-only Q24 seam tracer against a selected source.

This is deliberately a thin development wrapper: the diagnostic implementation
is reused byte-for-byte, while its estimator binding and output namespace are
changed before entering ``main``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DIAGNOSTIC = (
    P2
    / "public_observer_fit_q24_b38_late_blowup_diagnostic_v1"
    / "run_diagnostic_v1.py"
)
SOURCES = {
    "combo": (
        P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py",
        "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5",
    ),
    "global_b": (
        P2 / "fullbudget_k21_k4_closure_q24_raw_diag_load_rescale_v1/estimator.py",
        "5399e9140554fe676797d7bb6307bafbedf40a3ba9511b4eda8023d56732b160",
    ),
    "parent": (
        P2 / "fullbudget_k21_k4_closure_q24_v1/estimator.py",
        "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52",
    ),
}


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("combo_bound_diagnostic", DIAGNOSTIC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DIAGNOSTIC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--role-rank", type=int, required=True)
    args = parser.parse_args()
    source, digest = SOURCES[args.source]
    output = HERE / "runs" / args.source / f"row_{args.role_rank:04d}"
    diagnostic = load_diagnostic()
    diagnostic.Q24 = source
    diagnostic.Q24_SHA256 = digest
    diagnostic.RUNS = HERE / "runs" / args.source
    sys.argv = [
        str(DIAGNOSTIC),
        "--role-rank",
        str(args.role_rank),
        "--output-dir",
        str(output),
    ]
    return int(diagnostic.main())


if __name__ == "__main__":
    raise SystemExit(main())
