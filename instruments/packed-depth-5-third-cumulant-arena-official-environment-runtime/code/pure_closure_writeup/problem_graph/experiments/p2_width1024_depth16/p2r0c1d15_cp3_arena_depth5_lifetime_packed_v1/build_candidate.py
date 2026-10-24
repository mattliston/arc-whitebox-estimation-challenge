#!/usr/bin/env python3
"""Build the full Q24 estimator with the audited lifetime-packed D5 kernel."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
PACKED_KERNEL = HERE / "packed_d5_kernel.py"
DESTINATION = HERE / "estimator.py"
EXPECTED_SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
EXPECTED_PACKED_KERNEL_SHA256 = "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891"
EXPECTED_CANDIDATE_SHA256 = "74095215f842a0a993ad89c94cbaa27baf2cbc2f8cb2868e490972be8b9ab5e5"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_packed():
    spec = importlib.util.spec_from_file_location("packed_d5_build_authority", PACKED_KERNEL)
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


OLD_FUSION = '''    return (
        fnp.asarray(_CONNECTED_K21_WEIGHTS, dtype=fnp.float64),
        fnp.asarray(_CONNECTED_K31_WEIGHTS, dtype=fnp.float64),
        fnp.asarray(_CONNECTED_K22_WEIGHTS, dtype=fnp.float64),
        fnp.asarray(_CONNECTED_K4_WEIGHTS, dtype=fnp.float64),
        fnp.asarray(_EDGEWORTH_DENOMINATORS, dtype=fnp.float64),
        fnp.asarray(_EDGEWORTH_QUADRATIC_MAP, dtype=fnp.float64),
        fnp.asarray(_INV_FACTORIAL[:_Q24_TERMS], dtype=fnp.float64),
        fnp.asarray(_CUBIC_MATMUL_WEIGHTS, dtype=fnp.float32),
        fnp.asarray(_QUARTIC_MATMUL_MAP, dtype=fnp.float32),
        fnp.asarray(2.0, dtype=fnp.float64),
        fnp.asarray(3.0, dtype=fnp.float32),
        fnp.asarray(4.0, dtype=fnp.float32),
'''


NEW_FUSION = '''    return (
        None,
        None,
        None,
        None,
        fnp.asarray(_EDGEWORTH_DENOMINATORS, dtype=fnp.float64),
        None,
        fnp.asarray(_INV_FACTORIAL[:_Q24_TERMS], dtype=fnp.float64),
        None,
        None,
        None,
        None,
        None,
'''


OLD_READ_HEADER = '''def _cp_read(cp_lambda, cp_p, cp_q, product_a, product_b, arena_views):'''
NEW_READ_HEADER = '''def _cp_read(
    cp_lambda, cp_p, cp_q, product_a, product_b, read_outputs, arena_views,
):'''


OLD_READ_OUTPUTS = '''    first = arena_views[-2]
    _cp3_winograd_arena_depth5(
        active_a, fnp.transpose(active_b), first, arena_views
    )
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
    second = arena_views[-1]
'''


NEW_READ_OUTPUTS = '''    first, second = read_outputs
    _cp3_winograd_arena_depth5(
        active_a, fnp.transpose(active_b), first, arena_views
    )
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
'''


OLD_CONTEXT = '''        cp3_arena = fnp.empty(
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
'''


NEW_CONTEXT = '''        cp3_arena = fnp.empty(
            _CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32
        )
        product_a = fnp.empty(width * _CP3_CAP, dtype=fnp.float32)
        product_b = fnp.empty(width * _CP3_CAP, dtype=fnp.float32)
        read_output_start = width * (_CP3_CAP - width)
        read_outputs = (
            fnp.reshape(product_a[read_output_start:], (width, width)),
            fnp.reshape(product_b[read_output_start:], (width, width)),
        )
        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            product_a,
            product_b,
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
            read_outputs,
        )
'''


OLD_CALL = '''        cp_pre_k21 = _cp_read(
            cp_lambda,
            pushed_p,
            pushed_q,
            cp3_context[3],
            cp3_context[4],
            cp3_context[7],
        )
'''


NEW_CALL = '''        cp_pre_k21 = _cp_read(
            cp_lambda,
            pushed_p,
            pushed_q,
            cp3_context[3],
            cp3_context[4],
            cp3_context[8],
            cp3_context[7],
        )
'''


def compact_specs(specs):
    return tuple((start, stop, shape) for _owner, start, stop, shape in specs)


def build() -> bytes:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("passed D4 parent changed")
    if sha256(PACKED_KERNEL) != EXPECTED_PACKED_KERNEL_SHA256:
        raise RuntimeError("packed kernel authority changed")
    packed = load_packed()
    push_elements, _push_regions, push_specs = packed.packed_layout(
        1024, 1024, 15360
    )
    read_elements, _read_regions, read_specs = packed.packed_layout(
        1024, 14336, 1024
    )
    if (push_elements, read_elements) != (580_235_264, 894_939_136):
        raise AssertionError((push_elements, read_elements))

    text = SOURCE.read_text()
    constants_start = text.index("_CP3_CAP = 16384\n")
    constants_end = text.index("_Q24_TERMS = 7\n", constants_start)
    constants = (
        "_CP3_CAP = 16384\n"
        "_CP3_WINOGRAD_LEVELS = 5\n"
        f"_CP3_WINOGRAD_ARENA_ELEMENTS = {max(push_elements, read_elements)}\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {compact_specs(push_specs)!r}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {compact_specs(read_specs)!r}\n"
        "_CP4_CAP = 2048\n"
    )
    text = text[:constants_start] + constants + text[constants_end:]
    if text.count("_cp3_winograd_arena_depth4") != 5:
        raise AssertionError(text.count("_cp3_winograd_arena_depth4"))
    text = text.replace("_cp3_winograd_arena_depth4", "_cp3_winograd_arena_depth5")
    text = replace_once(
        text,
        "Audited four-level fixed-arena Winograd product with explicit out=.",
        "Audited five-level lifetime-packed Winograd product with explicit out=.",
        "kernel docstring",
    )
    text = replace_once(
        text,
        'raise ValueError("D4 arena kernel requires compatible /16 matrices")',
        'raise ValueError("packed D5 kernel requires compatible /32 matrices")',
        "kernel shape gate",
    )
    text = replace_once(text, OLD_FUSION, NEW_FUSION, "dead fusion payloads")
    text = replace_once(text, OLD_READ_HEADER, NEW_READ_HEADER, "read signature")
    text = replace_once(text, OLD_READ_OUTPUTS, NEW_READ_OUTPUTS, "external read outputs")
    text = replace_once(text, OLD_CONTEXT, NEW_CONTEXT, "packed scratch context")
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
    DESTINATION.write_bytes(payload)
    print(observed)


if __name__ == "__main__":
    main()
