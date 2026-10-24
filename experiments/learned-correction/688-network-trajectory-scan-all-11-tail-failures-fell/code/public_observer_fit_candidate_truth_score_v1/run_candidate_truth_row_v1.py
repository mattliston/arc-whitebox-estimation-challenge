#!/usr/bin/env python3
"""Run one frozen candidate through the already-validated public truth scorer."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_PATH = P2 / "public_observer_fit_truth_scan_v1/run_truth_row_v1.py"
SOURCES = {
    "gaussian_m11": (
        P2 / "fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/gaussian_m11/estimator.py",
        "7946a7630088b0ebfacd59c974e78414b8317f174716fa516024ba5040cdf1be",
    ),
    "m11_no_l3sq": (
        P2 / "fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l3sq/estimator.py",
        "2568729a12a4eeba850c76592dbc126c3ef69973770ca43d5a14c653ca4500bb",
    ),
    "m11_no_l4": (
        P2 / "fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l4/estimator.py",
        "c8be3f3d71cb6a28686d5a39481b7680187293be0201c21f1a586353a2b70e65",
    ),
    "correlation_shrink": (
        P2 / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/correlation_shrink/estimator.py",
        "fca768809e6662b6fbe449a1b1e28476046a46f3e40c829fbb101d342bf279ad",
    ),
    "raw_diag_load_rescale": (
        P2 / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/raw_diag_load_rescale/estimator.py",
        "5399e9140554fe676797d7bb6307bafbedf40a3ba9511b4eda8023d56732b160",
    ),
    "spectral_negative_buffer_1_16": (
        P2 / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/spectral_negative_buffer_1_16/estimator.py",
        "4a9e179f3810af4f9878a2e57366cbaf9d614a35e8b59c6c6ef656cf69d54b14",
    ),
    "hybrid_exact_var_floor_trigger": (
        P2 / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/hybrid_exact_var_floor_trigger/estimator.py",
        "aa87cee5e26ab25590a693d9ac08d5c2bc79cdd43b0c643ca9176df21919addb",
    ),
    "manual_rho_hybrid_floor": (
        P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py",
        "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5",
    ),
    "manual_rho_hybrid_near_floor_min": (
        P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py",
        "26aca6ed1fb52ec431132f87739fe729ec6855cc2fa2c80642fadd1c4c786e5d",
    ),
    "final_ec34_product_probe": (
        P2 / "fullbudget_k21_k4_closure_q24_final_ec34_product_probe_v1/estimator.py",
        "9ee59d3822040c64408d6889a172b1a8c4bc224569e6167a4adf02fb4ebac108",
    ),
    "cs2": (
        P2 / "fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py",
        "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779",
    ),
    "a1_no_cp4": (
        P2 / "fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py",
        "5b222ef4f8120adb9443da5c2a1f89c79085a768ac2aef10c61c18413ef6a025",
    ),
}


def load_base():
    tag = "public_truth_base_runner"
    spec = importlib.util.spec_from_file_location(tag, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import base truth runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(SOURCES), required=True)
    parser.add_argument("--role-rank", type=int, required=True)
    args = parser.parse_args()
    source, source_hash = SOURCES[args.candidate]
    base = load_base()
    old_source = base.ESTIMATOR
    base.ESTIMATOR = source
    base.HASHES.pop(old_source)
    base.HASHES[source] = source_hash
    base.RUNS = HERE / "runs" / args.candidate
    base.RUNS.mkdir(parents=True, exist_ok=True)
    saved_argv = sys.argv
    try:
        sys.argv = [str(BASE_PATH), "--role-rank", str(args.role_rank)]
        return int(base.main())
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
