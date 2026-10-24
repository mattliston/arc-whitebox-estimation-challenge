#!/usr/bin/env python3
"""Mechanically deepen the passed D3 CP3 arena to the audited D4 ceiling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D3_DIR = P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1"
SOURCE = D3_DIR / "estimator.py"
D3_BUILDER = D3_DIR / "build_candidate.py"
DEFAULT_DESTINATION = HERE / "estimator.py"
EXPECTED_SOURCE_SHA256 = "1771a616c86507ce1aad75dced22be57caddb93f5121034c5022d4a11c62c8c4"
EXPECTED_D3_BUILDER_SHA256 = "8890c877e3c8ef44e6b2a104a85e934ee8f3b3715470bf5ac527058fff84f031"
EXPECTED_DEFAULT_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"

WIDTH = 1_024
LEVELS = 4
MAX_PUSH_RANK = 15_360
MAX_READ_RANK = 14_336
EXPECTED_PUSH_ELEMENTS = 844_165_120
EXPECTED_READ_ELEMENTS = 681_480_192
EXPECTED_ARENA_ROSTER = 591_553_830_912
EXPECTED_HOT_SAVING = 374_532_792_320


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_d3_builder():
    spec = importlib.util.spec_from_file_location("d3_builder_authority", D3_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(D3_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(cp4_cap: int) -> tuple[bytes, dict[str, int]]:
    if not 1 <= cp4_cap <= 2048:
        raise ValueError("cp4 cap must be in [1, 2048]")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("passed D3 source changed")
    if sha256(D3_BUILDER) != EXPECTED_D3_BUILDER_SHA256:
        raise RuntimeError("D3 layout authority changed")
    d3 = load_d3_builder()
    push_elements, push_specs, push_bill = d3.layout(
        WIDTH, WIDTH, MAX_PUSH_RANK, LEVELS
    )
    read_elements, read_specs, read_bill = d3.layout(
        WIDTH, MAX_READ_RANK, WIDTH, LEVELS
    )
    if (push_elements, read_elements) != (
        EXPECTED_PUSH_ELEMENTS,
        EXPECTED_READ_ELEMENTS,
    ):
        raise AssertionError((push_elements, read_elements))

    roster = (
        [(WIDTH, WIDTH, WIDTH * layer) for layer in range(1, 16) for _ in range(2)]
        + [(WIDTH, WIDTH * layer, WIDTH) for layer in range(1, 15) for _ in range(2)]
    )
    dense = sum(d3.mm(*shape) for shape in roster)
    arena = sum(d3.arena_flops(*shape, LEVELS) for shape in roster)
    if dense != 966_086_623_232 or arena != EXPECTED_ARENA_ROSTER:
        raise AssertionError((dense, arena))
    if dense - arena != EXPECTED_HOT_SAVING:
        raise AssertionError(dense - arena)

    text = SOURCE.read_text()
    constants_start = text.index("_CP3_CAP = 16384\n")
    constants_end = text.index("_Q24_TERMS = 7\n", constants_start)
    constants = (
        "_CP3_CAP = 16384\n"
        f"_CP3_WINOGRAD_LEVELS = {LEVELS}\n"
        f"_CP3_WINOGRAD_ARENA_ELEMENTS = {push_elements}\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {push_specs!r}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {read_specs!r}\n"
        f"_CP4_CAP = {cp4_cap}\n"
    )
    text = text[:constants_start] + constants + text[constants_end:]
    if text.count("_cp3_winograd_arena_depth3") != 5:
        raise AssertionError(text.count("_cp3_winograd_arena_depth3"))
    text = text.replace(
        "_cp3_winograd_arena_depth3", "_cp3_winograd_arena_depth4"
    )
    text = text.replace(
        "Audited three-level fixed-arena Winograd product with explicit out=.",
        "Audited four-level fixed-arena Winograd product with explicit out=.",
        1,
    )
    text = text.replace(
        'raise ValueError("D3 arena kernel requires compatible /8 matrices")',
        'raise ValueError("D4 arena kernel requires compatible /16 matrices")',
        1,
    )
    payload = text.encode("utf-8")
    return payload, {
        "push_arena_elements": push_elements,
        "push_arena_bytes": 4 * push_elements,
        "read_arena_elements": read_elements,
        "read_arena_bytes": 4 * read_elements,
        "push_binding_flops": push_bill,
        "read_binding_flops": read_bill,
        "cold_binding_flops": push_bill + read_bill,
        "hot_cp3_roster_flops": arena,
        "hot_cp3_saving_flops": dense - arena,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp4-cap", type=int, default=2048)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    payload, accounting = build(args.cp4_cap)
    observed = sha256_bytes(payload)
    if args.cp4_cap == 2048 and args.destination == DEFAULT_DESTINATION:
        if EXPECTED_DEFAULT_SHA256 != "TO_BE_FROZEN" and observed != EXPECTED_DEFAULT_SHA256:
            raise RuntimeError((observed, EXPECTED_DEFAULT_SHA256))
    args.destination.write_bytes(payload)
    print(json.dumps({"source_sha256": observed, "cp4_cap": args.cp4_cap, **accounting}, sort_keys=True))


if __name__ == "__main__":
    main()
