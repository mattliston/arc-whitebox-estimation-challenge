#!/usr/bin/env python3
"""Build the k5 probe: diagonal 5th-cumulant carrier + L5 univariate term.

Attacks the pair model's per-step information limit with 5th-order marginal
information. Gaussian-part kappa5 is emitted at every reset from closed
forms, pushed diagonally through W^5 (exact for the diagonal family), and
consumed as the +k5/(120 sigma^5) He5 column in the univariate Edgeworth
correction at every layer and the final read. All vector ops (<1e9 FLOPs).
The ladder c[p,q] = p*sigma*c[p-1,q-1] gives the p=4 row's q=5 entry
c4_5 = 24 sigma^4 pdf (verified against the three existing c4 columns).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "public_q24_native_transport_energy_final_workspace_v4"
    / "estimator.py"
)
BASE_SHA = "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76"
OUT = HERE / "estimator.py"

APPENDED = '''

# --- diagonal kappa5 carrier (P2R0F3A probe) ------------------------------

def _relu_gaussian_kappa5(mean, raw, sigma, pdf):
    """Gaussian-part 5th cumulant of relu(N(mean, sigma^2)), closed form."""
    mass, raw1, raw2, raw3, raw4 = raw
    mean2 = mean * mean
    sigma2_l = sigma * sigma
    raw5 = (
        mean * (mean2 * mean2 + 10.0 * mean2 * sigma2_l + 15.0 * sigma2_l * sigma2_l) * mass
        + sigma * (mean2 * mean2 + 9.0 * mean2 * sigma2_l + 8.0 * sigma2_l * sigma2_l) * pdf
    )
    m1, m2, m3, m4, m5 = raw1, raw2, raw3, raw4, raw5
    m1_2 = m1 * m1
    return (
        m5
        - 5.0 * m4 * m1
        - 10.0 * m3 * m2
        + 20.0 * m3 * m1_2
        + 30.0 * m2 * m2 * m1
        - 60.0 * m2 * m1_2 * m1
        + 24.0 * m1_2 * m1_2 * m1
    )


def _push_kappa5(k5_post, weight_context):
    weight32, _wt, weight2, _w2t, weight3, _w3t = weight_context
    weight5 = weight2 * weight3
    return fnp.transpose(weight5) @ k5_post
'''

REPLACEMENTS = [
    # R1: L5 column in the univariate Edgeworth (signature + table + weights).
    (
        "def _univariate_edgeworth(\n"
        "    coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,\n"
        "    inverse_context=None,\n"
        "):\n"
        '    """Exact frozen 1+L3+L4+1/2L3^2 univariate specialization."""\n',
        "def _univariate_edgeworth(\n"
        "    coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,\n"
        "    inverse_context=None, k5=None,\n"
        "):\n"
        '    """Frozen 1+L3+L4+1/2L3^2 univariate specialization, plus an\n'
        "    optional +k5/(120 sigma^5) He5 column when a k5 state exists.\"\"\"\n",
    ),
    (
        "    k3_squared = k3 * k3\n"
        "\n"
        "    c1, c2, c3 = coefficients[0], coefficients[1], coefficients[2]\n"
        "    sigma4 = sigma2 * sigma2\n"
        "    c4_3 = 24.0 * sigma3 * raw1\n"
        "    c4_4 = 24.0 * sigma4 * mass\n"
        "    c4_6 = -24.0 * sigma4 * alpha * pdf\n"
        "    standardized = fnp.stack(\n"
        "        (\n"
        "            k3 * inv_sigma3 / 6.0,\n"
        "            k4 * inv_sigma4 / 24.0,\n"
        "            k3_squared * inv_sigma6 / 72.0,\n"
        "        ),\n"
        "        axis=0,\n"
        "    )\n",
        "    k3_squared = k3 * k3\n"
        "\n"
        "    c1, c2, c3 = coefficients[0], coefficients[1], coefficients[2]\n"
        "    sigma4 = sigma2 * sigma2\n"
        "    c4_3 = 24.0 * sigma3 * raw1\n"
        "    c4_4 = 24.0 * sigma4 * mass\n"
        "    c4_6 = -24.0 * sigma4 * alpha * pdf\n"
        "    if k5 is None:\n"
        "        k5_row = fnp.zeros_like(k3)\n"
        "    else:\n"
        "        inv_sigma5 = inv_sigma4 * _inv_sigma\n"
        "        k5_row = k5 * inv_sigma5 / 120.0\n"
        "    c4_5 = 24.0 * sigma4 * pdf\n"
        "    standardized = fnp.stack(\n"
        "        (\n"
        "            k3 * inv_sigma3 / 6.0,\n"
        "            k4 * inv_sigma4 / 24.0,\n"
        "            k3_squared * inv_sigma6 / 72.0,\n"
        "            k5_row,\n"
        "        ),\n"
        "        axis=0,\n"
        "    )\n",
    ),
    (
        "    coefficient_table = fnp.reshape(\n"
        "        fnp.stack(\n"
        "            (\n"
        "                c1[0], c2[0], c3[0], raw4,\n"
        "                c1[3], c2[3], c3[3], c4_3,\n"
        "                c1[4], c2[4], c3[4], c4_4,\n"
        "                c1[6], c2[6], c3[6], c4_6,\n"
        "            ),\n"
        "            axis=0,\n"
        "        ),\n"
        "        (4, 4, int(sigma.shape[0])),\n"
        "    )\n",
        "    coefficient_table = fnp.reshape(\n"
        "        fnp.stack(\n"
        "            (\n"
        "                c1[0], c2[0], c3[0], raw4,\n"
        "                c1[3], c2[3], c3[3], c4_3,\n"
        "                c1[4], c2[4], c3[4], c4_4,\n"
        "                c1[6], c2[6], c3[6], c4_6,\n"
        "                c1[5], c2[5], c3[5], c4_5,\n"
        "            ),\n"
        "            axis=0,\n"
        "        ),\n"
        "        (5, 4, int(sigma.shape[0])),\n"
        "    )\n",
    ),
    (
        "    correction_terms = coefficient_table[1:] * standardized[:, None, :]\n"
        "    corrected = coefficient_table[0] + correction_terms[0]\n"
        "    corrected = corrected + correction_terms[1]\n"
        "    corrected = corrected + correction_terms[2]\n",
        "    correction_terms = coefficient_table[1:] * standardized[:, None, :]\n"
        "    corrected = coefficient_table[0] + correction_terms[0]\n"
        "    corrected = corrected + correction_terms[1]\n"
        "    corrected = corrected + correction_terms[2]\n"
        "    corrected = corrected + correction_terms[3]\n",
    ),
    # R2: reset emits post-kappa5 and consumes the pushed pre-kappa5.
    (
        "    k3 = fnp.diag(pre_k21)\n"
        "    inverse_context = _inverse_sigma_context(sigma)\n"
        "    raw1, raw2, raw3, raw4 = _univariate_edgeworth(\n"
        "        coefficients,\n"
        "        gaussian_raw,\n"
        "        sigma,\n"
        "        alpha,\n"
        "        pdf,\n"
        "        k3,\n"
        "        pre_k4,\n"
        "        sigma2,\n"
        "        sigma3,\n"
        "        inverse_context,\n"
        "    )\n",
        "    k3 = fnp.diag(pre_k21)\n"
        "    inverse_context = _inverse_sigma_context(sigma)\n"
        "    raw1, raw2, raw3, raw4 = _univariate_edgeworth(\n"
        "        coefficients,\n"
        "        gaussian_raw,\n"
        "        sigma,\n"
        "        alpha,\n"
        "        pdf,\n"
        "        k3,\n"
        "        pre_k4,\n"
        "        sigma2,\n"
        "        sigma3,\n"
        "        inverse_context,\n"
        "        k5=pre_k5,\n"
        "    )\n",
    ),
    (
        "def _full_relu_reset(\n"
        "    pre_mean,\n"
        "    pre_covariance,\n"
        "    pre_k21,\n"
        "    pre_k4,\n"
        "    pre_k31,\n"
        "    pre_k22,\n",
        "def _full_relu_reset(\n"
        "    pre_mean,\n"
        "    pre_covariance,\n"
        "    pre_k21,\n"
        "    pre_k4,\n"
        "    pre_k31,\n"
        "    pre_k22,\n"
        "    pre_k5,\n",
    ),
]


def main():
    source = BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA
    patched = source
    for index, (old, new) in enumerate(REPLACEMENTS, start=1):
        count = patched.count(old)
        assert count == 1, f"anchor K5-R{index} matched {count} times"
        patched = patched.replace(old, new)
    patched += APPENDED
    OUT.write_text(patched)
    receipt = {
        "schema": "k5-probe-builder-receipt-v1",
        "base_sha256": BASE_SHA,
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "note": "call-site wiring (reset caller, final read, state threading, h1 init) is applied by wire_k5.py as a second pass",
    }
    (HERE / "BUILD_RECEIPT_STAGE1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
