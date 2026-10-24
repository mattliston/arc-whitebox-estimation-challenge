#!/usr/bin/env python3
"""Second pass: thread the k5 state through init, layer loop, and final read."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "estimator.py"

REPLACEMENTS = [
    # W1: reset emits gaussian-part post-kappa5 and returns it.
    (
        "    return (\n"
        "        raw1, covariance, k21, k4, k31, k22, alpha, pdf, sigma,\n"
        "        repaired_variance, covariance2, variance_outer,\n"
        "    )\n",
        "    k5_post = _relu_gaussian_kappa5(pre_mean, gaussian_raw, sigma, pdf)\n"
        "    return (\n"
        "        raw1, covariance, k21, k4, k31, k22, alpha, pdf, sigma,\n"
        "        repaired_variance, covariance2, variance_outer, k5_post,\n"
        "    )\n",
    ),
    # W2: layer-loop unpack gains k5_post; the call passes the pushed pre_k5.
    (
        "            repaired_variance,\n"
        "            covariance2,\n"
        "            variance_outer,\n"
        "        ) = _full_relu_reset(\n"
        "            pre_mean,\n"
        "            pre_covariance,\n"
        "            pre_k21,\n"
        "            pre_k4,\n"
        "            pre_k31,\n"
        "            pre_k22,\n",
        "            repaired_variance,\n"
        "            covariance2,\n"
        "            variance_outer,\n"
        "            k5_post,\n"
        "        ) = _full_relu_reset(\n"
        "            pre_mean,\n"
        "            pre_covariance,\n"
        "            pre_k21,\n"
        "            pre_k4,\n"
        "            pre_k31,\n"
        "            pre_k22,\n"
        "            _push_kappa5(k5_post, weight_context),\n",
    ),
    # W3: final read consumes the pushed k5.
    (
        "            final_mean = _univariate_edgeworth(\n"
        "                coefficients,\n"
        "                gaussian_raw,\n"
        "                sigma,\n"
        "                alpha,\n"
        "                pdf,\n"
        "                final_k3,\n"
        "                final_k4,\n"
        "                sigma2,\n"
        "                sigma3,\n"
        "            )[0]\n",
        "            final_mean = _univariate_edgeworth(\n"
        "                coefficients,\n"
        "                gaussian_raw,\n"
        "                sigma,\n"
        "                alpha,\n"
        "                pdf,\n"
        "                final_k3,\n"
        "                final_k4,\n"
        "                sigma2,\n"
        "                sigma3,\n"
        "                k5=_push_kappa5(k5_post, weight_context),\n"
        "            )[0]\n",
    ),
    # W4: h1 init emits and returns the first k5_post.
    (
        "        cp4_u,\n"
        "        cubic,\n"
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "    )\n",
        "        cp4_u,\n"
        "        cubic,\n"
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "        _relu_gaussian_kappa5(pre_mean, gaussian_raw, sigma, pdf),\n"
        "    )\n",
    ),
    # W5: init caller unpacks the extra element.
    (
        "        pairing_alpha,\n"
        "        initial_native_context,\n"
        "    ) = _initialize_h1(\n",
        "        pairing_alpha,\n"
        "        initial_native_context,\n"
        "        k5_post,\n"
        "    ) = _initialize_h1(\n",
    ),
]


def main():
    source = TARGET.read_text()
    patched = source
    for index, (old, new) in enumerate(REPLACEMENTS, start=1):
        count = patched.count(old)
        assert count == 1, f"wire anchor W{index} matched {count} times"
        patched = patched.replace(old, new)
    TARGET.write_text(patched)
    receipt = {
        "schema": "k5-probe-wire-receipt-v1",
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "anchors": len(REPLACEMENTS),
    }
    (HERE / "BUILD_RECEIPT_STAGE2.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
