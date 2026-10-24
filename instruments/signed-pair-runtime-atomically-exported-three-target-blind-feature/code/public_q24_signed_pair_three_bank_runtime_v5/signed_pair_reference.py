#!/usr/bin/env python3
"""Structurally independent reference for the accepted signed-pair law.

This target-blind implementation uses a different workspace schema, helper
decomposition, while-loop dictionary construction, and sector-block packing.
It deliberately does not call or import the literal implementation.  The
adapter executes it on the same live parent state and byte-compares every
H16/H64/H256 A2:A14 tape and A15-zero buffer with the literal execution.
"""

from __future__ import annotations

from q24_three_bank_runtime_token_v2 import require_runtime_token


REFERENCE_IMPLEMENTATION_ID = "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
WIDTH = 1024
CAPTURE_LAYERS = 13
SECTORS = 8
H16 = 16
H64 = 64
H256 = 256
RECEIVERS_H16 = 2
RECEIVERS_H64 = 8
RECEIVERS_H256 = 32
PAIR_SCALE_F32 = 0.031265269964933395
TAU_FLOOR_F32 = 9.313225746154785e-10
MGS_DOT_FLOOR_F64 = 9.313225746154785e-10
MGS_RMS_FLOOR_F64 = 9.5367431640625e-07
CAPTURE_LAYERS_ZERO_BASED = tuple(range(2, 15))
PROBE_SELECTOR_BY_SECTOR = (3, 3, 0, 1, 1, 2, 0, 3)
MONOMIAL_EXPONENTS = (
    (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
    (2, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),
    (0, 2, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 2, 0),
    (0, 0, 1, 1), (0, 0, 0, 2),
    (3, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 0, 0, 1),
    (1, 2, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 2, 0),
    (1, 0, 1, 1), (1, 0, 0, 2), (0, 3, 0, 0), (0, 2, 1, 0),
    (0, 2, 0, 1), (0, 1, 2, 0), (0, 1, 1, 1), (0, 1, 0, 2),
    (0, 0, 3, 0), (0, 0, 2, 1), (0, 0, 1, 2), (0, 0, 0, 3),
)
DEGREE_ONE_UNIT = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)


def reference_candidate_exponents(sector, runtime_token=None):
    """Build the accepted ordered 30-tuple list without literal-source code."""
    require_runtime_token(runtime_token)
    skipped = DEGREE_ONE_UNIT[PROBE_SELECTOR_BY_SECTOR[sector]]
    answer = []
    index = 0
    while index < len(MONOMIAL_EXPONENTS) and len(answer) < 30:
        exponent = MONOMIAL_EXPONENTS[index]
        if exponent != skipped:
            answer.append(exponent)
        index += 1
    return tuple(answer)


def allocate_reference_workspace(fnp, runtime_token=None):
    """Allocate an independent buffer graph for one complete reference row."""
    require_runtime_token(runtime_token)
    workspace = {
        "planes": fnp.empty((12, WIDTH, WIDTH), dtype=fnp.float32),
        "vectors": fnp.empty((21, WIDTH), dtype=fnp.float32),
        "probe32": fnp.empty((4, WIDTH), dtype=fnp.float32),
        "dictionary32": fnp.empty(
            (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
        ),
        "contraction": fnp.empty(
            (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
        ),
        "scaled_contraction": fnp.empty(
            (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
        ),
        "bounded_contraction": fnp.empty(
            (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
        ),
        "A16": fnp.empty((CAPTURE_LAYERS, WIDTH, H16), dtype=fnp.float32),
        "A64": fnp.empty((CAPTURE_LAYERS, WIDTH, H64), dtype=fnp.float32),
        "A256": fnp.empty((CAPTURE_LAYERS, WIDTH, H256), dtype=fnp.float32),
        "A15_16": fnp.zeros((WIDTH, H16), dtype=fnp.float32),
        "A15_64": fnp.zeros((WIDTH, H64), dtype=fnp.float32),
        "A15_256": fnp.zeros((WIDTH, H256), dtype=fnp.float32),
        "probe64": fnp.empty((4, WIDTH), dtype=fnp.float64),
        "candidate_a": fnp.empty(WIDTH, dtype=fnp.float64),
        "candidate_b": fnp.empty(WIDTH, dtype=fnp.float64),
        "prior64": fnp.empty(WIDTH, dtype=fnp.float64),
        "product64": fnp.empty(WIDTH, dtype=fnp.float64),
        "projection64": fnp.empty(WIDTH, dtype=fnp.float64),
        "square64": fnp.empty(WIDTH, dtype=fnp.float64),
        "dot_numerator": fnp.empty((), dtype=fnp.float64),
        "dot_denominator_raw": fnp.empty((), dtype=fnp.float64),
        "dot_denominator": fnp.empty((), dtype=fnp.float64),
        "coefficient": fnp.empty((), dtype=fnp.float64),
        "rms_square": fnp.empty((), dtype=fnp.float64),
        "rms_raw": fnp.empty((), dtype=fnp.float64),
        "rms_denominator": fnp.empty((), dtype=fnp.float64),
        "mean_tau": fnp.empty((), dtype=fnp.float32),
    }
    return workspace


def _reference_monomial(fnp, exponents, workspace, runtime_token):
    require_runtime_token(runtime_token)
    current = workspace["candidate_a"]
    alternate = workspace["candidate_b"]
    fnp.copyto(current, 1.0)
    probe_index = 0
    while probe_index < 4:
        multiplication = 0
        while multiplication < exponents[probe_index]:
            fnp.multiply(
                current,
                workspace["probe64"][probe_index],
                out=alternate,
            )
            current, alternate = alternate, current
            multiplication += 1
        probe_index += 1
    return current, alternate


def _reference_orthogonal_receiver(
    fnp,
    sector,
    receiver_index,
    exponents,
    workspace,
    runtime_token,
):
    require_runtime_token(runtime_token)
    current, alternate = _reference_monomial(
        fnp, exponents, workspace, runtime_token
    )
    prior_index = 0
    while prior_index < receiver_index:
        fnp.copyto(
            workspace["prior64"],
            workspace["dictionary32"][sector, :, prior_index],
        )
        fnp.multiply(
            workspace["prior64"], current, out=workspace["product64"]
        )
        fnp.sum(
            workspace["product64"],
            dtype=fnp.float64,
            out=workspace["dot_numerator"],
        )
        fnp.multiply(
            workspace["prior64"],
            workspace["prior64"],
            out=workspace["square64"],
        )
        fnp.sum(
            workspace["square64"],
            dtype=fnp.float64,
            out=workspace["dot_denominator_raw"],
        )
        fnp.maximum(
            workspace["dot_denominator_raw"],
            MGS_DOT_FLOOR_F64,
            out=workspace["dot_denominator"],
        )
        fnp.divide(
            workspace["dot_numerator"],
            workspace["dot_denominator"],
            out=workspace["coefficient"],
        )
        fnp.multiply(
            workspace["prior64"],
            workspace["coefficient"],
            out=workspace["projection64"],
        )
        fnp.subtract(current, workspace["projection64"], out=alternate)
        current, alternate = alternate, current
        prior_index += 1
    fnp.multiply(current, current, out=workspace["square64"])
    fnp.mean(
        workspace["square64"],
        dtype=fnp.float64,
        out=workspace["rms_square"],
    )
    fnp.sqrt(workspace["rms_square"], out=workspace["rms_raw"])
    fnp.maximum(
        workspace["rms_raw"],
        MGS_RMS_FLOOR_F64,
        out=workspace["rms_denominator"],
    )
    fnp.divide(current, workspace["rms_denominator"], out=alternate)
    fnp.copyto(workspace["dictionary32"][sector, :, receiver_index], alternate)


def _reference_dictionary(fnp, workspace, runtime_token):
    require_runtime_token(runtime_token)
    fnp.copyto(workspace["dictionary32"][:, :, 0], 1.0)
    sector = 0
    while sector < SECTORS:
        selector = PROBE_SELECTOR_BY_SECTOR[sector]
        fnp.copyto(
            workspace["dictionary32"][sector, :, 1],
            workspace["probe32"][selector],
        )
        exponents = reference_candidate_exponents(sector, runtime_token)
        receiver_index = 2
        while receiver_index < RECEIVERS_H256:
            _reference_orthogonal_receiver(
                fnp,
                sector,
                receiver_index,
                exponents[receiver_index - 2],
                workspace,
                runtime_token,
            )
            receiver_index += 1
        sector += 1


def _reference_standardized_pair_planes(
    fnp,
    post_covariance_f64,
    post_k31_f64,
    post_k22_f64,
    repaired_variance_f64,
    workspace,
    runtime_token,
):
    require_runtime_token(runtime_token)
    planes = workspace["planes"]
    vectors = workspace["vectors"]
    raw = planes[0:3]
    scaled = planes[3:6]
    auxiliary = planes[6:9]
    tau_safe = vectors[0]
    tau_raw = vectors[1]
    inverse_tau = vectors[2]
    inverse_tau2 = vectors[3]
    inverse_tau3 = vectors[4]
    inverse_tau4 = vectors[5]
    fnp.copyto(raw[0], post_covariance_f64)
    fnp.copyto(raw[1], post_k31_f64)
    fnp.copyto(raw[2], post_k22_f64)
    fnp.copyto(tau_safe, repaired_variance_f64)
    fnp.sqrt(tau_safe, out=tau_raw)
    fnp.maximum(tau_raw, TAU_FLOOR_F32, out=tau_safe)
    fnp.divide(1.0, tau_safe, out=inverse_tau)
    fnp.multiply(inverse_tau, inverse_tau, out=inverse_tau2)
    fnp.multiply(inverse_tau2, inverse_tau, out=inverse_tau3)
    fnp.multiply(inverse_tau2, inverse_tau2, out=inverse_tau4)
    fnp.multiply(raw[0], inverse_tau[:, None], out=scaled[0])
    fnp.multiply(scaled[0], inverse_tau[None, :], out=auxiliary[0])
    fnp.multiply(raw[1], inverse_tau3[:, None], out=scaled[1])
    fnp.multiply(scaled[1], inverse_tau[None, :], out=auxiliary[1])
    fnp.multiply(raw[2], inverse_tau2[:, None], out=scaled[2])
    fnp.multiply(scaled[2], inverse_tau2[None, :], out=auxiliary[2])
    fnp.abs(auxiliary, out=raw)
    fnp.add(1.0, raw, out=scaled)
    fnp.divide(auxiliary, scaled, out=raw)
    return raw[0], raw[1], raw[2], inverse_tau3, inverse_tau4, tau_safe


def _reference_probe_dictionary_inputs(
    fnp,
    post_k21_f64,
    post_k4_f64,
    same_layer_alpha_f64,
    inverse_tau3,
    inverse_tau4,
    tau_safe,
    workspace,
    runtime_token,
):
    require_runtime_token(runtime_token)
    vectors = workspace["vectors"]
    post_k3 = vectors[6]
    post_k4 = vectors[7]
    pre_alpha = vectors[8]
    probe_pre = vectors[9:13]
    probe_abs = vectors[13:17]
    probe_denominator = vectors[17:21]
    fnp.copyto(post_k3, fnp.diagonal(post_k21_f64))
    fnp.copyto(post_k4, post_k4_f64)
    fnp.copyto(pre_alpha, same_layer_alpha_f64)
    fnp.mean(tau_safe, out=workspace["mean_tau"])
    fnp.divide(tau_safe, workspace["mean_tau"], out=probe_denominator[0])
    fnp.subtract(probe_denominator[0], 1.0, out=probe_pre[0])
    fnp.multiply(2.0, pre_alpha, out=probe_denominator[1])
    fnp.subtract(probe_denominator[1], 1.0, out=probe_pre[1])
    fnp.multiply(post_k3, inverse_tau3, out=probe_pre[2])
    fnp.multiply(post_k4, inverse_tau4, out=probe_pre[3])
    fnp.abs(probe_pre, out=probe_abs)
    fnp.add(1.0, probe_abs, out=probe_denominator)
    fnp.divide(probe_pre, probe_denominator, out=workspace["probe32"])
    fnp.copyto(workspace["probe64"], workspace["probe32"])


def _reference_sector_planes(fnp, r, q, t, workspace, runtime_token):
    require_runtime_token(runtime_token)
    sectors = workspace["planes"][3:11]
    scratch = workspace["planes"][11]
    fnp.copyto(sectors[0], q)
    q_transpose = fnp.transpose(q)
    fnp.copyto(sectors[1], q_transpose)
    fnp.copyto(sectors[2], t)
    fnp.multiply(r, q, out=sectors[3])
    fnp.multiply(r, q_transpose, out=sectors[4])
    fnp.multiply(r, t, out=sectors[5])
    fnp.multiply(q, q_transpose, out=sectors[6])
    fnp.multiply(r, r, out=scratch)
    fnp.multiply(scratch, t, out=sectors[7])
    return sectors


def _reference_contract_and_pack(fnp, sectors, slot, workspace, runtime_token):
    require_runtime_token(runtime_token)
    fnp.matmul(sectors, workspace["dictionary32"], out=workspace["contraction"])
    fnp.multiply(
        PAIR_SCALE_F32,
        workspace["contraction"],
        out=workspace["scaled_contraction"],
    )
    fnp.abs(
        workspace["scaled_contraction"], out=workspace["bounded_contraction"]
    )
    fnp.add(1.0, workspace["bounded_contraction"], out=workspace["contraction"])
    fnp.divide(
        workspace["scaled_contraction"],
        workspace["contraction"],
        out=workspace["bounded_contraction"],
    )
    # Independent sector-block packing replaces the literal receiver-stride loop.
    sector = 0
    while sector < SECTORS:
        wide_start = sector * RECEIVERS_H256
        fnp.copyto(
            workspace["A256"][slot, :, wide_start:wide_start + RECEIVERS_H256],
            workspace["bounded_contraction"][sector],
        )
        fnp.copyto(
            workspace["A64"][
                slot,
                :,
                sector * RECEIVERS_H64:(sector + 1) * RECEIVERS_H64,
            ],
            workspace["bounded_contraction"][sector, :, :RECEIVERS_H64],
        )
        fnp.copyto(
            workspace["A16"][
                slot,
                :,
                sector * RECEIVERS_H16:(sector + 1) * RECEIVERS_H16,
            ],
            workspace["bounded_contraction"][sector, :, :RECEIVERS_H16],
        )
        sector += 1


def capture_reference_one_layer(
    fnp,
    layer_zero_based,
    post_covariance_f64,
    post_k21_f64,
    post_k4_f64,
    post_k31_f64,
    post_k22_f64,
    repaired_variance_f64,
    same_layer_preactivation_alpha_f64,
    workspace,
    runtime_token=None,
):
    """Compute one reference A_l through independent helper decomposition."""
    require_runtime_token(runtime_token)
    if layer_zero_based not in CAPTURE_LAYERS_ZERO_BASED:
        raise ValueError("capture layer must be zero-based 2..14")
    slot = layer_zero_based - 2
    r, q, t, inverse_tau3, inverse_tau4, tau_safe = (
        _reference_standardized_pair_planes(
            fnp,
            post_covariance_f64,
            post_k31_f64,
            post_k22_f64,
            repaired_variance_f64,
            workspace,
            runtime_token,
        )
    )
    _reference_probe_dictionary_inputs(
        fnp,
        post_k21_f64,
        post_k4_f64,
        same_layer_preactivation_alpha_f64,
        inverse_tau3,
        inverse_tau4,
        tau_safe,
        workspace,
        runtime_token,
    )
    sectors = _reference_sector_planes(
        fnp, r, q, t, workspace, runtime_token
    )
    _reference_dictionary(fnp, workspace, runtime_token)
    _reference_contract_and_pack(
        fnp, sectors, slot, workspace, runtime_token
    )


def reference_bank_views(workspace, runtime_token=None):
    """Return the independently computed tapes and exact A15 zeros."""
    require_runtime_token(runtime_token)
    return {
        "H16": (workspace["A16"], workspace["A15_16"]),
        "H64": (workspace["A64"], workspace["A15_64"]),
        "H256": (workspace["A256"], workspace["A15_256"]),
    }
