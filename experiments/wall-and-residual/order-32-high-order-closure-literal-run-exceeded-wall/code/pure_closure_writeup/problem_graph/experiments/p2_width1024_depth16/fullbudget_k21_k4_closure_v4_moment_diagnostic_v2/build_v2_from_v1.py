#!/usr/bin/env python3
"""Deterministically derive normalized diagnostic v2 from frozen V1 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4_DIR = ROOT / "fullbudget_k21_k4_closure_v4"
V1_DIR = ROOT / "fullbudget_k21_k4_closure_v4_moment_diagnostic"
V4_ESTIMATOR = V4_DIR / "estimator.py"
V4_AUDIT = V4_DIR / "INDEPENDENT_SOURCE_AUDIT_V4.json"
V1_ESTIMATOR = V1_DIR / "estimator.py"
V1_MANIFEST = V1_DIR / "SOURCE_MANIFEST.json"
V1_FREEZE = V1_DIR / "FREEZE_RECEIPT.json"
V1_HOLD_RECEIPT = (
    V1_DIR / "runs" / "v4_target_free_moment_seed20260824" / "receipt.json"
)
OUTPUT = HERE / "estimator.py"

V4_ESTIMATOR_SHA256 = (
    "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
)
V4_AUDIT_SHA256 = (
    "c9553e1596e8e61b29c862256487586d38f66c92a8389cd1541c5b1e9ffa5a9d"
)
V1_ESTIMATOR_SHA256 = (
    "3428b9e37641bcae9c7fcc867a55672f1160fecd225dc084f8a980d07ffe4256"
)
V1_MANIFEST_SHA256 = (
    "ab774cb80025309913e2f2b25a0de3b59615073d72dfd25df4f7c80f50914c29"
)
V1_FREEZE_SHA256 = (
    "843d7f47a7348b95b3fa07f6e4c14bf318e60128c79930e72bfc27bff67461ce"
)
V1_HOLD_RECEIPT_SHA256 = (
    "93124980ee134d2b0bad4e6f00b46ced6a7329fcc356945c12736443f4723ee8"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def build_source(parent: str) -> str:
    source = parent
    source = _replace_once(
        source,
        '"""FB-EC4-32 V4 moment diagnostic: target-free trajectory audit.',
        '"""FB-EC4-32 V4 normalized-moment diagnostic: target-free audit v2.',
        "module title",
    )
    source = _replace_once(
        source,
        "The additive observation path retains only nine metered scalars per\n"
        "layer. It never repairs or feeds diagnostic values back into the\n"
        "closure. The research-only class returns a (depth, 9) diagnostic\n"
        "array and must not be submitted.\n",
        "The additive observation path retains fourteen metered scalars per layer:\n"
        "the worst scale-relative A and B margins, their unit indices, their\n"
        "cancellation scales, and raw r1..r4 at each index. It never clips, repairs,\n"
        "or feeds diagnostic values back into the closure. The research-only class\n"
        "returns a (depth, 14) diagnostic array and must not be submitted.\n",
        "module diagnostic scope",
    )
    source = _replace_once(
        source,
        "_DIAGNOSTIC_CHANNELS = (\n"
        '    "min_r1", "count_r1_lt_zero",\n'
        '    "min_v", "count_v_lt_zero",\n'
        '    "min_A", "count_A_lt_zero",\n'
        '    "min_B", "count_B_lt_zero",\n'
        '    "finite_count_r1_r2_r3_r4_v_A_B",\n'
        ")\n",
        "_DIAGNOSTIC_CHANNELS = (\n"
        '    "min_relative_A", "worst_A_index", "worst_A_scale",\n'
        '    "worst_A_r1", "worst_A_r2", "worst_A_r3", "worst_A_r4",\n'
        '    "min_relative_B", "worst_B_index", "worst_B_scale",\n'
        '    "worst_B_r1", "worst_B_r2", "worst_B_r3", "worst_B_r4",\n'
        ")\n",
        "channel order",
    )
    source = _replace_once(
        source,
        "def _moment_feasibility_row(raw1, raw2, raw3, raw4):\n"
        '    """Meter necessary Stieltjes margins without clipping or repair."""\n'
        "    raw1_squared = raw1 * raw1\n"
        "    variance = raw2 - raw1_squared\n"
        "    raw2_squared = raw2 * raw2\n"
        "    margin_a = raw1 * raw3 - raw2_squared\n"
        "    centered_21 = raw3 - raw1 * raw2\n"
        "    margin_b = (\n"
        "        (raw4 - raw2_squared) * variance\n"
        "        - centered_21 * centered_21\n"
        "    )\n"
        "    finite_count = fnp.count_nonzero(fnp.isfinite(raw1))\n"
        "    for field in (raw2, raw3, raw4, variance, margin_a, margin_b):\n"
        "        finite_count = finite_count + fnp.count_nonzero(\n"
        "            fnp.isfinite(field)\n"
        "        )\n"
        "    return fnp.stack(\n"
        "        (\n"
        "            fnp.min(raw1),\n"
        "            fnp.count_nonzero(raw1 < 0.0),\n"
        "            fnp.min(variance),\n"
        "            fnp.count_nonzero(variance < 0.0),\n"
        "            fnp.min(margin_a),\n"
        "            fnp.count_nonzero(margin_a < 0.0),\n"
        "            fnp.min(margin_b),\n"
        "            fnp.count_nonzero(margin_b < 0.0),\n"
        "            finite_count,\n"
        "        ),\n"
        "        axis=0,\n"
        "    )\n",
        "def _normalized_moment_worst_row(raw1, raw2, raw3, raw4):\n"
        '    """Report worst cancellation-relative A/B margins without repair."""\n'
        "    raw1_squared = raw1 * raw1\n"
        "    variance = raw2 - raw1_squared\n"
        "    raw2_squared = raw2 * raw2\n"
        "    a_left = raw1 * raw3\n"
        "    margin_a = a_left - raw2_squared\n"
        "    centered_21 = raw3 - raw1 * raw2\n"
        "    b_left = (raw4 - raw2_squared) * variance\n"
        "    centered_21_squared = centered_21 * centered_21\n"
        "    margin_b = b_left - centered_21_squared\n"
        "    scale_a = fnp.abs(a_left) + fnp.abs(raw2_squared)\n"
        "    scale_b = fnp.abs(b_left) + fnp.abs(centered_21_squared)\n"
        "    relative_a = margin_a / scale_a\n"
        "    relative_b = margin_b / scale_b\n"
        "    index_a = fnp.argmin(relative_a)\n"
        "    index_b = fnp.argmin(relative_b)\n"
        "    return fnp.stack(\n"
        "        (\n"
        "            relative_a[index_a], index_a, scale_a[index_a],\n"
        "            raw1[index_a], raw2[index_a], raw3[index_a], raw4[index_a],\n"
        "            relative_b[index_b], index_b, scale_b[index_b],\n"
        "            raw1[index_b], raw2[index_b], raw3[index_b], raw4[index_b],\n"
        "        ),\n"
        "        axis=0,\n"
        "    )\n",
        "normalized diagnostic helper",
    )
    call_count = source.count("_moment_feasibility_row(")
    if call_count != 3:
        raise RuntimeError(
            f"diagnostic call sites: expected three anchors, found {call_count}"
        )
    source = source.replace(
        "_moment_feasibility_row(", "_normalized_moment_worst_row("
    )
    source = _replace_once(
        source,
        "return fnp.zeros((smoke_depth, 9), dtype=fnp.float64)",
        "return fnp.zeros((smoke_depth, 14), dtype=fnp.float64)",
        "diagnostic smoke shape",
    )
    source = _replace_once(
        source,
        "The additive observation path retains fourteen metered scalars per layer:\n"
        "the worst scale-relative A and B margins, their unit indices, their\n"
        "cancellation scales, and raw r1..r4 at each index. It never clips, repairs,\n"
        "or feeds diagnostic values back into the closure. The research-only class\n"
        "returns a (depth, 14) diagnostic array and must not be submitted.\n",
        "The additive observation path retains twenty-four metered scalars per layer:\n"
        "the worst scale-relative A and B margins, certified outward-rounded error\n"
        "bounds, class counts, unit indices, cancellation scales, and raw r1..r4 at\n"
        "each index. It never clips, floors, repairs, or feeds diagnostic values back\n"
        "into the closure. The research-only class returns a (depth, 24) diagnostic\n"
        "array and must not be submitted.\n",
        "certified diagnostic scope",
    )
    source = _replace_once(
        source,
        "_INV_TWO_PI = 0.15915494309189535\n",
        "_INV_TWO_PI = 0.15915494309189535\n"
        "_TWO_U = 2.220446049250313e-16  # 2^-52\n",
        "certified local error constant",
    )
    source = _replace_once(
        source,
        "_DIAGNOSTIC_CHANNELS = (\n"
        '    "min_relative_A", "worst_A_index", "worst_A_scale",\n'
        '    "worst_A_r1", "worst_A_r2", "worst_A_r3", "worst_A_r4",\n'
        '    "min_relative_B", "worst_B_index", "worst_B_scale",\n'
        '    "worst_B_r1", "worst_B_r2", "worst_B_r3", "worst_B_r4",\n'
        ")\n",
        "_DIAGNOSTIC_CHANNELS = (\n"
        '    "min_relative_A", "worst_A_relative_error_bound",\n'
        '    "worst_A_index", "worst_A_scale",\n'
        '    "worst_A_r1", "worst_A_r2", "worst_A_r3", "worst_A_r4",\n'
        '    "count_A_CERT_NEG", "count_A_ZERO_COMPATIBLE",\n'
        '    "count_A_CERT_POS", "count_A_scale_zero",\n'
        '    "min_relative_B", "worst_B_relative_error_bound",\n'
        '    "worst_B_index", "worst_B_scale",\n'
        '    "worst_B_r1", "worst_B_r2", "worst_B_r3", "worst_B_r4",\n'
        '    "count_B_CERT_NEG", "count_B_ZERO_COMPATIBLE",\n'
        '    "count_B_CERT_POS", "count_B_scale_zero",\n'
        ")\n",
        "certified channel order",
    )
    helper_start = source.index("def _normalized_moment_worst_row(")
    helper_end = source.index("\n\ndef _q32_context(", helper_start)
    certified_helper = '''def _up(nonnegative):
    """Round a nonnegative fp64 error bound one representable value outward."""
    return fnp.nextafter(nonnegative, fnp.inf)


def _add_sub_error(error_x, error_y, result):
    """Certified outward error for one stored fp64 add or subtract."""
    inherited = _up(error_x + error_y)
    local = _up(_TWO_U * fnp.abs(result))
    return _up(inherited + local)


def _multiply_error(x, error_x, y, error_y, result):
    """Certified outward error for one stored fp64 multiplication."""
    left = _up(fnp.abs(x) * error_y)
    right = _up(fnp.abs(y) * error_x)
    inherited = _up(left + right)
    cross = _up(error_x * error_y)
    inherited = _up(inherited + cross)
    local = _up(_TWO_U * fnp.abs(result))
    return _up(inherited + local)


def _normalized_moment_worst_row(raw1, raw2, raw3, raw4):
    """Certify stored-tuple A/B signs and report worst relative margins."""
    zero_error = fnp.zeros_like(raw1)
    raw1_squared = raw1 * raw1
    error_raw1_squared = _multiply_error(
        raw1, zero_error, raw1, zero_error, raw1_squared
    )
    variance = raw2 - raw1_squared
    error_variance = _add_sub_error(
        zero_error, error_raw1_squared, variance
    )
    raw2_squared = raw2 * raw2
    error_raw2_squared = _multiply_error(
        raw2, zero_error, raw2, zero_error, raw2_squared
    )
    a_left = raw1 * raw3
    error_a_left = _multiply_error(
        raw1, zero_error, raw3, zero_error, a_left
    )
    margin_a = a_left - raw2_squared
    error_margin_a = _add_sub_error(
        error_a_left, error_raw2_squared, margin_a
    )
    raw1_raw2 = raw1 * raw2
    error_raw1_raw2 = _multiply_error(
        raw1, zero_error, raw2, zero_error, raw1_raw2
    )
    centered_21 = raw3 - raw1_raw2
    error_centered_21 = _add_sub_error(
        zero_error, error_raw1_raw2, centered_21
    )
    raw4_minus_raw2_squared = raw4 - raw2_squared
    error_raw4_minus_raw2_squared = _add_sub_error(
        zero_error, error_raw2_squared, raw4_minus_raw2_squared
    )
    b_left = raw4_minus_raw2_squared * variance
    error_b_left = _multiply_error(
        raw4_minus_raw2_squared,
        error_raw4_minus_raw2_squared,
        variance,
        error_variance,
        b_left,
    )
    centered_21_squared = centered_21 * centered_21
    error_centered_21_squared = _multiply_error(
        centered_21,
        error_centered_21,
        centered_21,
        error_centered_21,
        centered_21_squared,
    )
    margin_b = b_left - centered_21_squared
    error_margin_b = _add_sub_error(
        error_b_left, error_centered_21_squared, margin_b
    )
    scale_a = fnp.abs(a_left) + fnp.abs(raw2_squared)
    scale_b = fnp.abs(b_left) + fnp.abs(centered_21_squared)
    zero_scale_a = scale_a == 0.0
    zero_scale_b = scale_b == 0.0
    relative_a = fnp.where(zero_scale_a, 0.0, margin_a / scale_a)
    relative_b = fnp.where(zero_scale_b, 0.0, margin_b / scale_b)
    relative_error_a = fnp.where(
        zero_scale_a, 0.0, error_margin_a / scale_a
    )
    relative_error_b = fnp.where(
        zero_scale_b, 0.0, error_margin_b / scale_b
    )
    cert_negative_a = margin_a < -error_margin_a
    cert_positive_a = margin_a > error_margin_a
    cert_negative_b = margin_b < -error_margin_b
    cert_positive_b = margin_b > error_margin_b
    count_negative_a = fnp.count_nonzero(cert_negative_a)
    count_positive_a = fnp.count_nonzero(cert_positive_a)
    count_zero_compatible_a = int(raw1.shape[0]) - count_negative_a
    count_zero_compatible_a = count_zero_compatible_a - count_positive_a
    count_negative_b = fnp.count_nonzero(cert_negative_b)
    count_positive_b = fnp.count_nonzero(cert_positive_b)
    count_zero_compatible_b = int(raw1.shape[0]) - count_negative_b
    count_zero_compatible_b = count_zero_compatible_b - count_positive_b
    index_a = fnp.argmin(relative_a)
    index_b = fnp.argmin(relative_b)
    return fnp.stack(
        (
            relative_a[index_a], relative_error_a[index_a],
            index_a, scale_a[index_a],
            raw1[index_a], raw2[index_a], raw3[index_a], raw4[index_a],
            count_negative_a, count_zero_compatible_a, count_positive_a,
            fnp.count_nonzero(zero_scale_a),
            relative_b[index_b], relative_error_b[index_b],
            index_b, scale_b[index_b],
            raw1[index_b], raw2[index_b], raw3[index_b], raw4[index_b],
            count_negative_b, count_zero_compatible_b, count_positive_b,
            fnp.count_nonzero(zero_scale_b),
        ),
        axis=0,
    )
'''
    source = source[:helper_start] + certified_helper + source[helper_end:]
    source = _replace_once(
        source,
        "return fnp.zeros((smoke_depth, 14), dtype=fnp.float64)",
        "return fnp.zeros((smoke_depth, 24), dtype=fnp.float64)",
        "certified diagnostic smoke shape",
    )
    source = _replace_once(
        source,
        "The additive observation path retains twenty-four metered scalars per layer:\n"
        "the worst scale-relative A and B margins, certified outward-rounded error\n"
        "bounds, class counts, unit indices, cancellation scales, and raw r1..r4 at\n"
        "each index. It never clips, floors, repairs, or feeds diagnostic values back\n"
        "into the closure. The research-only class returns a (depth, 24) diagnostic\n",
        "The additive observation path retains twenty-eight metered scalars per layer:\n"
        "the worst scale-relative A and B margins, certified outward-rounded error\n"
        "bounds, raw margins, class counts, unit indices, cancellation scales, and raw\n"
        "r1..r4 at each index. It never clips, floors, repairs, or feeds diagnostic\n"
        "values back into the closure. The research-only class returns a (depth, 28) diagnostic\n",
        "self-contained certified diagnostic scope",
    )
    source = _replace_once(
        source,
        '    "min_relative_A", "worst_A_relative_error_bound",\n'
        '    "worst_A_index", "worst_A_scale",\n',
        '    "min_relative_A", "worst_A_relative_error_bound",\n'
        '    "worst_A_raw_margin", "worst_A_absolute_error_bound",\n'
        '    "worst_A_index", "worst_A_scale",\n',
        "A absolute channels",
    )
    source = _replace_once(
        source,
        '    "min_relative_B", "worst_B_relative_error_bound",\n'
        '    "worst_B_index", "worst_B_scale",\n',
        '    "min_relative_B", "worst_B_relative_error_bound",\n'
        '    "worst_B_raw_margin", "worst_B_absolute_error_bound",\n'
        '    "worst_B_index", "worst_B_scale",\n',
        "B absolute channels",
    )
    source = _replace_once(
        source,
        "    zero_scale_a = scale_a == 0.0\n"
        "    zero_scale_b = scale_b == 0.0\n"
        "    relative_a = fnp.where(zero_scale_a, 0.0, margin_a / scale_a)\n"
        "    relative_b = fnp.where(zero_scale_b, 0.0, margin_b / scale_b)\n"
        "    relative_error_a = fnp.where(\n"
        "        zero_scale_a, 0.0, error_margin_a / scale_a\n"
        "    )\n"
        "    relative_error_b = fnp.where(\n"
        "        zero_scale_b, 0.0, error_margin_b / scale_b\n"
        "    )\n",
        "    positive_scale_a = scale_a > 0.0\n"
        "    positive_scale_b = scale_b > 0.0\n"
        "    relative_a = fnp.zeros_like(scale_a)\n"
        "    relative_b = fnp.zeros_like(scale_b)\n"
        "    relative_error_a = fnp.zeros_like(scale_a)\n"
        "    relative_error_b = fnp.zeros_like(scale_b)\n"
        "    fnp.divide(margin_a, scale_a, out=relative_a, where=positive_scale_a)\n"
        "    fnp.divide(margin_b, scale_b, out=relative_b, where=positive_scale_b)\n"
        "    fnp.divide(\n"
        "        error_margin_a, scale_a, out=relative_error_a,\n"
        "        where=positive_scale_a,\n"
        "    )\n"
        "    fnp.divide(\n"
        "        error_margin_b, scale_b, out=relative_error_b,\n"
        "        where=positive_scale_b,\n"
        "    )\n",
        "zero-scale masked normalization",
    )
    source = _replace_once(
        source,
        "    index_a = fnp.argmin(relative_a)\n"
        "    index_b = fnp.argmin(relative_b)\n",
        "    reduction_a = fnp.where(positive_scale_a, relative_a, fnp.inf)\n"
        "    reduction_b = fnp.where(positive_scale_b, relative_b, fnp.inf)\n"
        "    index_a = fnp.argmin(reduction_a)\n"
        "    index_b = fnp.argmin(reduction_b)\n",
        "zero-scale reduction sentinel",
    )
    source = _replace_once(
        source,
        "            relative_a[index_a], relative_error_a[index_a],\n"
        "            index_a, scale_a[index_a],\n",
        "            relative_a[index_a], relative_error_a[index_a],\n"
        "            margin_a[index_a], error_margin_a[index_a],\n"
        "            index_a, scale_a[index_a],\n",
        "A absolute output",
    )
    source = _replace_once(
        source,
        "            fnp.count_nonzero(zero_scale_a),\n"
        "            relative_b[index_b], relative_error_b[index_b],\n"
        "            index_b, scale_b[index_b],\n",
        "            fnp.count_nonzero(~positive_scale_a),\n"
        "            relative_b[index_b], relative_error_b[index_b],\n"
        "            margin_b[index_b], error_margin_b[index_b],\n"
        "            index_b, scale_b[index_b],\n",
        "B absolute output and A zero scale",
    )
    source = _replace_once(
        source,
        "            fnp.count_nonzero(zero_scale_b),\n",
        "            fnp.count_nonzero(~positive_scale_b),\n",
        "B zero scale",
    )
    source = _replace_once(
        source,
        "return fnp.zeros((smoke_depth, 24), dtype=fnp.float64)",
        "return fnp.zeros((smoke_depth, 28), dtype=fnp.float64)",
        "self-contained certified diagnostic smoke shape",
    )
    return source


def _verify_bindings() -> None:
    bindings = (
        (V4_ESTIMATOR, V4_ESTIMATOR_SHA256, "frozen V4 estimator"),
        (V4_AUDIT, V4_AUDIT_SHA256, "frozen V4 audit"),
        (V1_ESTIMATOR, V1_ESTIMATOR_SHA256, "frozen diagnostic v1 estimator"),
        (V1_MANIFEST, V1_MANIFEST_SHA256, "frozen diagnostic v1 manifest"),
        (V1_FREEZE, V1_FREEZE_SHA256, "frozen diagnostic v1 freeze receipt"),
        (V1_HOLD_RECEIPT, V1_HOLD_RECEIPT_SHA256, "actual diagnostic v1 HOLD"),
    )
    for path, expected, label in bindings:
        if not path.is_file() or _sha256(path) != expected:
            raise SystemExit(f"{label} hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write estimator.py; the default verifies exact derived bytes.",
    )
    args = parser.parse_args()
    _verify_bindings()
    expected = build_source(V1_ESTIMATOR.read_text(encoding="utf-8"))
    if args.write:
        OUTPUT.write_text(expected, encoding="utf-8")
    if not OUTPUT.is_file():
        raise SystemExit("estimator.py is absent; use --write initially")
    actual = OUTPUT.read_text(encoding="utf-8")
    if actual != expected:
        raise SystemExit("diagnostic v2 source differs from exact v1 transform")
    print(
        {
            "status": "PASS_EXACT_V1_HOLD_BOUND_NORMALIZED_V2_TRANSFORM",
            "v4_estimator_sha256": V4_ESTIMATOR_SHA256,
            "v4_audit_sha256": V4_AUDIT_SHA256,
            "v1_estimator_sha256": V1_ESTIMATOR_SHA256,
            "v1_hold_receipt_sha256": V1_HOLD_RECEIPT_SHA256,
            "v2_estimator_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
