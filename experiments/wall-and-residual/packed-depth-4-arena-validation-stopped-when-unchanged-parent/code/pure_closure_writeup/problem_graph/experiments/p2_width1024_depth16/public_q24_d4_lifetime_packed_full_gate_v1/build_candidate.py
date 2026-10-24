#!/usr/bin/env python3
"""Mechanically build the full public D4 source with packed CP3 storage.

The sealed D4 arithmetic is not replaced by imported code.  The packed-kernel
artifact supplies only exact layout specs; this builder changes constants,
routes the two read outputs to independent allocations, and leaves every D4
kernel statement in its original order.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STATIC = P2 / "public_q24_d4_lifetime_packed_reinvestment_v1"
SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
D4_RESULT = (
    P2
    / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
    / "D4_CANDIDATE_RESULT.json"
)
PACKED_KERNEL = STATIC / "packed_d4_kernel.py"
PACKED_RECEIPT = STATIC / "PACKED_D4_STATIC_RECEIPT.json"
DESTINATION = HERE / "estimator.py"

EXPECTED_SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
EXPECTED_D4_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
EXPECTED_PACKED_KERNEL_SHA256 = "bc23fe4f8a46bd194390b53523d63d5ab136fc1feb0240d589fcaf3fad681a37"
EXPECTED_PACKED_RECEIPT_SHA256 = "832220c4c9467d8d57569649198e2465e97bc908ebb96103e549e683964c4b51"
EXPECTED_CANDIDATE_SHA256 = "982775b3296709156e1d05c24d703a0982db8bab6dce83dc16744fb09361b066"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_packed():
    spec = importlib.util.spec_from_file_location(
        "packed_d4_full_gate_layout_authority", PACKED_KERNEL
    )
    if spec is None or spec.loader is None:
        raise ImportError(PACKED_KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


OLD_READ_HEADER = "def _cp_read(cp_lambda, cp_p, cp_q, product_a, product_b, arena_views):"
NEW_READ_HEADER = """def _cp_read(
    cp_lambda, cp_p, cp_q, product_a, product_b, read_outputs, arena_views,
):"""


OLD_READ_OUTPUTS = """    first = arena_views[-2]
    _cp3_winograd_arena_depth4(
        active_a, fnp.transpose(active_b), first, arena_views
    )
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
    second = arena_views[-1]
"""


NEW_READ_OUTPUTS = """    first, second = read_outputs
    _cp3_winograd_arena_depth4(
        active_a, fnp.transpose(active_b), first, arena_views
    )
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
"""


OLD_CONTEXT = """        cp3_arena = fnp.empty(
            _CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32
        )
        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
        )
"""


NEW_CONTEXT = """        cp3_arena = fnp.empty(
            _CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32
        )
        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
            (
                fnp.empty((width, width), dtype=fnp.float32),
                fnp.empty((width, width), dtype=fnp.float32),
            ),
        )
"""


OLD_CALL = """        cp_pre_k21 = _cp_read(
            cp_lambda,
            pushed_p,
            pushed_q,
            cp3_context[3],
            cp3_context[4],
            cp3_context[7],
        )
"""


NEW_CALL = """        cp_pre_k21 = _cp_read(
            cp_lambda,
            pushed_p,
            pushed_q,
            cp3_context[3],
            cp3_context[4],
            cp3_context[8],
            cp3_context[7],
        )
"""


def compact_specs(specs):
    return tuple((start, stop, shape) for _owner, start, stop, shape in specs)


def build() -> bytes:
    pins = {
        SOURCE: EXPECTED_SOURCE_SHA256,
        D4_RESULT: EXPECTED_D4_RESULT_SHA256,
        PACKED_KERNEL: EXPECTED_PACKED_KERNEL_SHA256,
        PACKED_RECEIPT: EXPECTED_PACKED_RECEIPT_SHA256,
    }
    for path, expected in pins.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"authority changed: {path}: {observed}")
    packed = load_packed()
    push_elements, _push_regions, push_specs = packed.packed_layout(
        1024, 1024, 15360
    )
    read_elements, _read_regions, read_specs = packed.packed_layout(
        1024, 14336, 1024
    )
    if (push_elements, read_elements) != (331_563_008, 511_393_792):
        raise AssertionError((push_elements, read_elements))
    if (len(push_specs), len(read_specs)) != (24, 24):
        raise AssertionError((len(push_specs), len(read_specs)))

    text = SOURCE.read_text()
    constants_start = text.index("_CP3_CAP = 16384\n")
    constants_end = text.index("_Q24_TERMS = 7\n", constants_start)
    constants = (
        "_CP3_CAP = 16384\n"
        "_CP3_WINOGRAD_LEVELS = 4\n"
        f"_CP3_WINOGRAD_ARENA_ELEMENTS = {max(push_elements, read_elements)}\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {compact_specs(push_specs)!r}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {compact_specs(read_specs)!r}\n"
        "_CP4_CAP = 2048\n"
    )
    text = text[:constants_start] + constants + text[constants_end:]
    text = replace_once(
        text,
        '"""Bind the sealed D1 arena views once on the first prediction."""',
        '"""Bind lifetime-packed D4 arena views once on first prediction."""',
        "bind docstring",
    )
    text = replace_once(
        text,
        '"""Audited four-level fixed-arena Winograd product with explicit out=."""',
        '"""Audited four-level lifetime-packed product with explicit out=."""',
        "kernel docstring",
    )
    text = replace_once(text, OLD_READ_HEADER, NEW_READ_HEADER, "read signature")
    text = replace_once(text, OLD_READ_OUTPUTS, NEW_READ_OUTPUTS, "read outputs")
    text = replace_once(text, OLD_CONTEXT, NEW_CONTEXT, "scratch context")
    text = replace_once(text, OLD_CALL, NEW_CALL, "read call")
    return text.encode("utf-8")


def main() -> None:
    payload = build()
    observed = sha256_bytes(payload)
    if (
        EXPECTED_CANDIDATE_SHA256 != "TO_BE_FROZEN"
        and observed != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError((observed, EXPECTED_CANDIDATE_SHA256))
    if DESTINATION.exists():
        if DESTINATION.read_bytes() != payload:
            raise RuntimeError("existing candidate differs from mechanical build")
    else:
        DESTINATION.write_bytes(payload)
    print(observed)


if __name__ == "__main__":
    main()
