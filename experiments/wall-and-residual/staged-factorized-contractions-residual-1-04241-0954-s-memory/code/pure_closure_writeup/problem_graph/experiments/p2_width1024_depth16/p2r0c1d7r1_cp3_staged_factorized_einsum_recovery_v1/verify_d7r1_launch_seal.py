#!/usr/bin/env python3
"""Read-only verifier for the root-only D7R1 tiny launch seal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEAL = HERE / "D7R1_TINY_RUNTIME_LAUNCH_SEAL.json"
EXPECTED_SEAL = "22da348b8297ea65bf115ab6ff4090dffabbf33072df8a3731abf1cc68945148"
EXPECTED_SHA256 = {
    "../p2r0c1d5_cp3_winograd_hybrid_v1/KERNEL_SMOKE_RESULT.json": "a50ca69aae9e4b39fca8d3de9014bcc7688af63a1f5ced6aab1f94de957235c9",
    "../p2r0c1d5_cp3_winograd_hybrid_v1/run_kernel_smoke.py": "44940602fc2f6fdb336911821de024bddb94da021a01ddf0ba9570e251bfb996",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/INDEPENDENT_D7R1_TINY_PRERUN_REVIEW.md": "d9b0d0a35b4f5ca7fa2651a8a2791d2d588e46067792779ef0d7dc5cf5aec586",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/INDEPENDENT_D7_DUAL_LAUNCH_FAILURE_AND_R1_ELIGIBILITY_AUDIT.md": "98db28ab6c80c5fd012fc569ed123cfbcf48d33fea7ec16a380b04be53560998",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_ATTEMPT_001_FAILURE.json": "5bdcc2530ff208843de0dc169bbc8d8b17746e75f12a1829b626531d24d0bbff",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_ATTEMPT_001_TRANSCRIPT.txt": "0e244c00291c6a915f2129082df5bed2510f7ffae4df00c49a96086207281417",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_ATTEMPT_002_FAILURE.json": "6084d8f0e7f36be2c82ac5c67cf2ccb0b0dc3de719ee171adee6083587f33bcf",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_ATTEMPT_002_TRANSCRIPT.txt": "4ac804fd6b724871362fb274216ca640d0e19ea1806b01c512d3707540b949f2",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_SEAL.json": "72c7bc9e264f63de2f6bcbe5129b927ed006bfe41b4cbbb98e7282c35c379602",
    "../p2r0c1d7_cp3_staged_factorized_einsum_design_v1/TINY_RUNTIME_LAUNCH_SEAL_ATTEMPT_001.json": "8aeb6adc651a8292dea39be53f6d3c01f66c0a4a9e82be257e05bc9253fb31fa",
    "ACCOUNTING_AND_RESOURCE_MANIFEST.json": "94ad1b2e89a58417bf91cb4fd844ac1d02674b58a2738b0218bf7cc63eb145ec",
    "DESIGN.md": "3ccdefd80d69817f1ad8519124f2f0def2f03e95a5c96385d921edfdc97b1b2e",
    "run_tiny_runtime_discriminator.py": "1222bdbc067ae28bdf90c0baced1687c5dbeb3624fe24b4f6c8f9a21f3a4a439",
    "staged_winograd_kernel.py": "81d14f2ca983c2820ab078d15a3a8a861427ac5a8b1b5e82828300256648bbe9",
    "verify_design_static.py": "27a067afcee05558f2d066bfc7ee594752dce8fa62a5c7166353b97ebed9a816",
}
TERMINAL_PATHS = (
    "D7R1_TINY_ATTEMPT_MARKER.json",
    "D7R1_BIND_INVENTORY_PREASSERT.json",
    "D7R1_RUNTIME_EXCEPTION_RECEIPT.json",
    "D7R1_TINY_RUNTIME_RESULT.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert SEAL.is_file()
    assert sha256(SEAL) == EXPECTED_SEAL
    seal = json.loads(SEAL.read_text())
    assert seal["schema"] == "p2r0c1d7r1-tiny-runtime-launch-seal-v1"
    assert seal["status"] == "AUTHORIZED_ONE_SHOT_D7R1_TINY_RUNTIME_ONLY"
    assert seal["node"] == "P2R0C1D7R1"
    assert seal["authorized_by_root"] is True
    assert seal["launch_owner"] == "/root"
    assert seal["authorized_runs"] == 1
    assert seal["exact_shape_authorized"] is False
    assert seal["sha256"] == EXPECTED_SHA256
    for relative, expected in EXPECTED_SHA256.items():
        path = HERE / relative
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)
    for relative in TERMINAL_PATHS:
        assert not (HERE / relative).exists(), relative
    print("PASS_D7R1_ROOT_ONLY_LAUNCH_SEAL_READY_NOT_INVOKED")


if __name__ == "__main__":
    main()
