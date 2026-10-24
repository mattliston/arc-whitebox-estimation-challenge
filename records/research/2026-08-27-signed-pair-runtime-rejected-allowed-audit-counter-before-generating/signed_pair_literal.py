#!/usr/bin/env python3
"""Token-gated literal H16/H64/H256 signed-pair action source V2.

This module imports no estimator, FlopScope, NumPy, row loader, or target
reader. It contains no runner, review, anchor, capability, release, scope, or
transaction digest. Every callable validates an opaque in-process token before
its first source operation.
"""

from __future__ import annotations

from q24_three_bank_runtime_token_v2 import require_runtime_token


LITERAL_IMPLEMENTATION_ID = "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
WIDTH = 1024
CAPTURE_LAYERS = 13
SECTORS = 8
H16 = 16
H64 = 64
H256 = 256
RECEIVERS_H16 = 2
RECEIVERS_H64 = 8
RECEIVERS_H256 = 32
PAIR_SCALE_F32 = 0.031265269964933395  # float32(1/sqrt(1023))
TAU_FLOOR_F32 = 9.313225746154785e-10  # float32(2^-30)
MGS_DOT_FLOOR_F64 = 9.313225746154785e-10  # float64(2^-30)
MGS_RMS_FLOOR_F64 = 9.5367431640625e-07  # float64(2^-20)

CAPTURE_LAYERS_ZERO_BASED = tuple(range(2, 15))
ACTION_TAPE_SLOT_BY_LAYER_ZERO_BASED = tuple(
    (layer, layer - 2) for layer in CAPTURE_LAYERS_ZERO_BASED
)
ZERO_ACTION_LAYER_ZERO_BASED = 15
PROBE_SELECTOR_BY_SECTOR = (3, 3, 0, 1, 1, 2, 0, 3)

# Degree 1, then 2, then 3; descending lexicographic exponent tuple within
# degree.  Each sector removes the one degree-one tuple equal to p_s and takes
# the first thirty remaining tuples for receivers 2..31.  H64 is the exact
# first-eight receiver prefix; H16 is the exact first-two prefix.
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

H16_PREFIX_INDEX_MAP = tuple(
    (sector, 2 * sector, 8 * sector, 32 * sector)
    for sector in range(SECTORS)
)

def candidate_exponents_for_sector(sector, runtime_token=None):
    """Return the exact 30 host-metadata exponent tuples for one sector."""
    require_runtime_token(runtime_token)
    selector = PROBE_SELECTOR_BY_SECTOR[sector]
    skipped = DEGREE_ONE_UNIT[selector]
    remaining = tuple(item for item in MONOMIAL_EXPONENTS if item != skipped)
    return remaining[:30]


def allocate_three_bank_workspace(fnp, runtime_token=None):
    """Allocate every persistent and scratch buffer exactly once per row."""
    require_runtime_token(runtime_token)

    # The 12 covariance planes preserve the accepted V2 phase-liveness law.
    plane_arena_f32 = fnp.empty((12, WIDTH, WIDTH), dtype=fnp.float32)
    vector_arena_f32 = fnp.empty((21, WIDTH), dtype=fnp.float32)
    probe_f32 = fnp.empty((4, WIDTH), dtype=fnp.float32)
    receiver_f32 = fnp.empty(
        (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
    )

    # One H256 contraction is packed sector-major; smaller banks are copied
    # only from its exact receiver prefixes.
    contract_raw_f32 = fnp.empty(
        (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
    )
    contract_scaled_f32 = fnp.empty(
        (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
    )
    contract_aux_f32 = fnp.empty(
        (SECTORS, WIDTH, RECEIVERS_H256), dtype=fnp.float32
    )

    action_H16_f32 = fnp.empty(
        (CAPTURE_LAYERS, WIDTH, H16), dtype=fnp.float32
    )
    action_H64_f32 = fnp.empty(
        (CAPTURE_LAYERS, WIDTH, H64), dtype=fnp.float32
    )
    action_H256_f32 = fnp.empty(
        (CAPTURE_LAYERS, WIDTH, H256), dtype=fnp.float32
    )
    action15_zero_H16_f32 = fnp.zeros((WIDTH, H16), dtype=fnp.float32)
    action15_zero_H64_f32 = fnp.zeros((WIDTH, H64), dtype=fnp.float32)
    action15_zero_H256_f32 = fnp.zeros((WIDTH, H256), dtype=fnp.float32)

    # Modified Gram-Schmidt is float64.  Prior receivers are copied from their
    # one-time f32 cast, so subsequent candidates see the exact retained bytes.
    probe_f64 = fnp.empty((4, WIDTH), dtype=fnp.float64)
    mgs_a_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    mgs_b_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    prior_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    product_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    projection_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    square_f64 = fnp.empty(WIDTH, dtype=fnp.float64)
    dot_numerator_f64 = fnp.empty((), dtype=fnp.float64)
    dot_denominator_raw_f64 = fnp.empty((), dtype=fnp.float64)
    dot_denominator_f64 = fnp.empty((), dtype=fnp.float64)
    coefficient_f64 = fnp.empty((), dtype=fnp.float64)
    rms_square_f64 = fnp.empty((), dtype=fnp.float64)
    rms_raw_f64 = fnp.empty((), dtype=fnp.float64)
    rms_denominator_f64 = fnp.empty((), dtype=fnp.float64)
    mean_tau_f32 = fnp.empty((), dtype=fnp.float32)
    return locals()


def _build_monomial(
    fnp,
    exponents,
    probe_f64,
    workspace,
    runtime_token=None,
):
    """Build one raw monomial in fixed p0,p1,p2,p3 multiplication order."""
    require_runtime_token(runtime_token)
    current = workspace["mgs_a_f64"]
    alternate = workspace["mgs_b_f64"]
    fnp.copyto(current, 1.0)
    for probe_index, exponent in enumerate(exponents):
        for _ in range(exponent):
            fnp.multiply(current, probe_f64[probe_index], out=alternate)
            current, alternate = alternate, current
    return current, alternate


def _orthogonalize_one_receiver(
    fnp,
    sector,
    receiver_index,
    exponents,
    workspace,
    runtime_token=None,
):
    """Exact float64 modified Gram-Schmidt followed by one f32 cast."""
    require_runtime_token(runtime_token)
    receiver_f32 = workspace["receiver_f32"]
    current, alternate = _build_monomial(
        fnp,
        exponents,
        workspace["probe_f64"],
        workspace,
        runtime_token,
    )
    for prior_index in range(receiver_index):
        fnp.copyto(
            workspace["prior_f64"],
            receiver_f32[sector, :, prior_index],
        )
        fnp.multiply(
            workspace["prior_f64"],
            current,
            out=workspace["product_f64"],
        )
        fnp.sum(
            workspace["product_f64"],
            dtype=fnp.float64,
            out=workspace["dot_numerator_f64"],
        )
        fnp.multiply(
            workspace["prior_f64"],
            workspace["prior_f64"],
            out=workspace["square_f64"],
        )
        fnp.sum(
            workspace["square_f64"],
            dtype=fnp.float64,
            out=workspace["dot_denominator_raw_f64"],
        )
        fnp.maximum(
            workspace["dot_denominator_raw_f64"],
            MGS_DOT_FLOOR_F64,
            out=workspace["dot_denominator_f64"],
        )
        fnp.divide(
            workspace["dot_numerator_f64"],
            workspace["dot_denominator_f64"],
            out=workspace["coefficient_f64"],
        )
        fnp.multiply(
            workspace["prior_f64"],
            workspace["coefficient_f64"],
            out=workspace["projection_f64"],
        )
        fnp.subtract(current, workspace["projection_f64"], out=alternate)
        current, alternate = alternate, current

    fnp.multiply(current, current, out=workspace["square_f64"])
    fnp.mean(
        workspace["square_f64"],
        dtype=fnp.float64,
        out=workspace["rms_square_f64"],
    )
    fnp.sqrt(workspace["rms_square_f64"], out=workspace["rms_raw_f64"])
    fnp.maximum(
        workspace["rms_raw_f64"],
        MGS_RMS_FLOOR_F64,
        out=workspace["rms_denominator_f64"],
    )
    fnp.divide(current, workspace["rms_denominator_f64"], out=alternate)
    fnp.copyto(receiver_f32[sector, :, receiver_index], alternate)


def _build_receiver_dictionary(
    fnp,
    workspace,
    runtime_token=None,
):
    """Build all eight sector dictionaries with exact nested prefix order."""
    require_runtime_token(runtime_token)
    receiver_f32 = workspace["receiver_f32"]
    probe_f32 = workspace["probe_f32"]
    fnp.copyto(receiver_f32[:, :, 0], 1.0)
    for sector, selector in enumerate(PROBE_SELECTOR_BY_SECTOR):
        fnp.copyto(receiver_f32[sector, :, 1], probe_f32[selector])
        exponents = candidate_exponents_for_sector(
            sector, runtime_token
        )
        for receiver_index, exponent_tuple in enumerate(exponents, start=2):
            _orthogonalize_one_receiver(
                fnp,
                sector,
                receiver_index,
                exponent_tuple,
                workspace,
                runtime_token,
            )


def capture_one_layer(
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
    """Capture one A_l for all three banks at the accepted V2 hook."""
    require_runtime_token(runtime_token)
    if layer_zero_based not in CAPTURE_LAYERS_ZERO_BASED:
        raise ValueError("capture layer must be zero-based 2..14")
    slot = layer_zero_based - 2

    arena = workspace["plane_arena_f32"]
    raw = arena[0:3]
    scaled = arena[3:6]
    aux = arena[6:9]
    sectors = arena[3:11]
    sector_scratch = arena[11]
    vectors = workspace["vector_arena_f32"]
    tau_safe = vectors[0]
    tau_raw = vectors[1]
    inverse_tau = vectors[2]
    inverse_tau2 = vectors[3]
    inverse_tau3 = vectors[4]
    inverse_tau4 = vectors[5]
    post_k3 = vectors[6]
    post_k4 = vectors[7]
    pre_alpha = vectors[8]
    probe_pre = vectors[9:13]
    probe_abs = vectors[13:17]
    probe_denominator = vectors[17:21]

    # Exact V2 f64-source to f32 boundary.  Parent objects remain read-only.
    fnp.copyto(raw[0], post_covariance_f64)
    fnp.copyto(raw[1], post_k31_f64)
    fnp.copyto(raw[2], post_k22_f64)
    post_k3_view_f64 = fnp.diagonal(post_k21_f64)
    fnp.copyto(post_k3, post_k3_view_f64)
    fnp.copyto(post_k4, post_k4_f64)
    fnp.copyto(pre_alpha, same_layer_preactivation_alpha_f64)
    fnp.copyto(tau_safe, repaired_variance_f64)

    fnp.sqrt(tau_safe, out=tau_raw)
    fnp.maximum(tau_raw, TAU_FLOOR_F32, out=tau_safe)
    fnp.divide(1.0, tau_safe, out=inverse_tau)
    fnp.multiply(inverse_tau, inverse_tau, out=inverse_tau2)
    fnp.multiply(inverse_tau2, inverse_tau, out=inverse_tau3)
    fnp.multiply(inverse_tau2, inverse_tau2, out=inverse_tau4)

    fnp.multiply(raw[0], inverse_tau[:, None], out=scaled[0])
    fnp.multiply(scaled[0], inverse_tau[None, :], out=aux[0])
    fnp.multiply(raw[1], inverse_tau3[:, None], out=scaled[1])
    fnp.multiply(scaled[1], inverse_tau[None, :], out=aux[1])
    fnp.multiply(raw[2], inverse_tau2[:, None], out=scaled[2])
    fnp.multiply(scaled[2], inverse_tau2[None, :], out=aux[2])
    fnp.abs(aux, out=raw)
    fnp.add(1.0, raw, out=scaled)
    fnp.divide(aux, scaled, out=raw)
    r = raw[0]
    q = raw[1]
    t = raw[2]

    fnp.mean(tau_safe, out=workspace["mean_tau_f32"])
    fnp.divide(tau_safe, workspace["mean_tau_f32"], out=probe_denominator[0])
    fnp.subtract(probe_denominator[0], 1.0, out=probe_pre[0])
    fnp.multiply(2.0, pre_alpha, out=probe_denominator[1])
    fnp.subtract(probe_denominator[1], 1.0, out=probe_pre[1])
    fnp.multiply(post_k3, inverse_tau3, out=probe_pre[2])
    fnp.multiply(post_k4, inverse_tau4, out=probe_pre[3])
    fnp.abs(probe_pre, out=probe_abs)
    fnp.add(1.0, probe_abs, out=probe_denominator)
    fnp.divide(probe_pre, probe_denominator, out=workspace["probe_f32"])
    fnp.copyto(workspace["probe_f64"], workspace["probe_f32"])

    fnp.copyto(sectors[0], q)
    q_transpose = fnp.transpose(q)
    fnp.copyto(sectors[1], q_transpose)
    fnp.copyto(sectors[2], t)
    fnp.multiply(r, q, out=sectors[3])
    fnp.multiply(r, q_transpose, out=sectors[4])
    fnp.multiply(r, t, out=sectors[5])
    fnp.multiply(q, q_transpose, out=sectors[6])
    fnp.multiply(r, r, out=sector_scratch)
    fnp.multiply(sector_scratch, t, out=sectors[7])

    _build_receiver_dictionary(fnp, workspace, runtime_token)
    fnp.matmul(
        sectors,
        workspace["receiver_f32"],
        out=workspace["contract_raw_f32"],
    )
    fnp.multiply(
        PAIR_SCALE_F32,
        workspace["contract_raw_f32"],
        out=workspace["contract_scaled_f32"],
    )
    fnp.abs(
        workspace["contract_scaled_f32"],
        out=workspace["contract_aux_f32"],
    )
    fnp.add(
        1.0,
        workspace["contract_aux_f32"],
        out=workspace["contract_raw_f32"],
    )
    fnp.divide(
        workspace["contract_scaled_f32"],
        workspace["contract_raw_f32"],
        out=workspace["contract_aux_f32"],
    )

    # Pack sector-major without reshape/stack/concatenate allocations.
    for receiver_index in range(RECEIVERS_H256):
        receiver_view = fnp.transpose(
            workspace["contract_aux_f32"][:, :, receiver_index]
        )
        fnp.copyto(
            workspace["action_H256_f32"][
                slot, :, receiver_index::RECEIVERS_H256
            ],
            receiver_view,
        )
    for sector in range(SECTORS):
        fnp.copyto(
            workspace["action_H64_f32"][
                slot, :, sector * RECEIVERS_H64:(sector + 1) * RECEIVERS_H64
            ],
            workspace["action_H256_f32"][
                slot,
                :,
                sector * RECEIVERS_H256:
                sector * RECEIVERS_H256 + RECEIVERS_H64,
            ],
        )
        fnp.copyto(
            workspace["action_H16_f32"][
                slot, :, sector * RECEIVERS_H16:(sector + 1) * RECEIVERS_H16
            ],
            workspace["action_H256_f32"][
                slot,
                :,
                sector * RECEIVERS_H256:
                sector * RECEIVERS_H256 + RECEIVERS_H16,
            ],
        )


def bank_views(workspace, runtime_token=None):
    """Return exact tape/A15-zero objects after all 13 captures."""
    require_runtime_token(runtime_token)
    return {
        "H16": (
            workspace["action_H16_f32"],
            workspace["action15_zero_H16_f32"],
        ),
        "H64": (
            workspace["action_H64_f32"],
            workspace["action15_zero_H64_f32"],
        ),
        "H256": (
            workspace["action_H256_f32"],
            workspace["action15_zero_H256_f32"],
        ),
    }
