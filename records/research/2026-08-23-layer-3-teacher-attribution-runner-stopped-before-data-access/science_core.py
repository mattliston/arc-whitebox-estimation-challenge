"""Target-free V5 Layer-3 attribution science implementation.

This module is pure computation. It has no filesystem, loader, release, claim,
target, fit, score, submission, child-process, or graph interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import flopscope as flops
import numpy as np


SQRT_2PI = math.sqrt(2.0 * math.pi)
VAR_FLOOR = 1.0e-30
BLOCK_ROWS = 256
PRODUCTION_WIDTH = 1024
PRODUCTION_CAP = 1792
PRODUCTION_PREFIXES = (1024, 4096, 16384)
T_CRITICAL_M2 = (14.478720640339768, 11.453164801899343, 9.041609643757877)
T_CRITICAL_M6 = (20.943811618166106, 16.59641392982402, 13.138897143142432)
SIGNED_MECHANISMS = (
    "recurrence_state",
    "remaining_premean_source",
    "diagonal_covariance_action",
    "offdiagonal_covariance_action",
    "first_cp_truncation",
    "k3_source_after_cp2048",
    "k4_readout_contribution",
)
DIRECT_ENDPOINTS = (
    "premean_source",
    "diagonal_covariance_action",
    "offdiagonal_covariance_action",
    "k3_source",
    "k4_energy",
    "k21_forward_source",
    "k21_transpose_source",
    "z3_affine_mean_closure",
)


def canonical_f32(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(result).all():
        raise AssertionError("non-finite float32 array")
    return result


def canonical_f64(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f8")
    if not np.isfinite(result).all():
        raise AssertionError("non-finite float64 array")
    return result


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def normal_cdf(value: np.ndarray) -> np.ndarray:
    return canonical_f64(np.asarray(flops.stats.norm.cdf(value), dtype=np.float64))


def walsh_probe(width: int = PRODUCTION_WIDTH, rank: int = 8) -> np.ndarray:
    if width <= 0 or width & (width - 1) or rank <= 0 or rank > width:
        raise ValueError("Walsh width must be a power of two and rank admissible")
    rows = np.arange(width, dtype=np.uint32)[:, None]
    columns = np.arange(rank, dtype=np.uint32)[None, :]
    value = np.bitwise_and(rows, columns)
    parity = np.zeros((width, rank), dtype=np.uint8)
    for _ in range(width.bit_length() - 1):
        parity ^= (value & 1).astype(np.uint8)
        value >>= 1
    probe = np.where(parity == 0, 1.0, -1.0).astype("<f8") / math.sqrt(width)
    if not np.allclose(probe.T @ probe, np.eye(rank), rtol=0.0, atol=2e-15):
        raise AssertionError("Walsh probe is not orthonormal")
    if width == 1024 and rank == 8:
        expected = "186a89cf1344af1d963171b632ac8a596475fff12ba76c9cb74029e691038a74"
        if hash_array(probe) != expected:
            raise AssertionError("production Walsh payload changed")
    return probe


def signed_decimal(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("seed coordinate must be an exact Python int")
    encoded = str(value).encode("ascii")
    if value == 0 and encoded != b"0":
        raise AssertionError("zero spelling drift")
    if value > 0 and (encoded.startswith(b"+") or encoded.startswith(b"0")):
        raise AssertionError("positive spelling drift")
    if value < 0 and (not encoded.startswith(b"-") or encoded[1:2] == b"0"):
        raise AssertionError("negative spelling drift")
    return encoded


def seed_material(mlp_seed: int, replicate: int) -> tuple[bytes, int]:
    if type(replicate) is not int or not 0 <= replicate < 16:
        raise ValueError("replicate must be an exact integer in 0..15")
    preimage = (
        b"p2-l3-attribution-v5|v2-phase2|mlp_seed|"
        + signed_decimal(mlp_seed)
        + b"|replicate|"
        + signed_decimal(replicate)
    )
    digest = hashlib.sha256(preimage).digest()
    return preimage, int.from_bytes(digest[:16], "little", signed=False)


def new_stream_hasher(preimage: bytes):
    digest = hashlib.sha256()
    digest.update(b"p2-l3-attribution-v5|stream-fingerprint-v1\x00")
    digest.update(len(preimage).to_bytes(4, "little"))
    digest.update(preimage)
    return digest


def update_stream_hasher(digest, index: int, block: np.ndarray) -> None:
    value = canonical_f32(block)
    if value.ndim != 2:
        raise AssertionError("stream block must be rank two")
    digest.update(index.to_bytes(8, "little"))
    digest.update(value.shape[0].to_bytes(4, "little"))
    digest.update(value.shape[1].to_bytes(4, "little"))
    digest.update(memoryview(value).cast("B"))


def hermite_coefficients(alpha: np.ndarray, pdf: np.ndarray, cdf: np.ndarray):
    coefficients = [pdf + alpha * cdf, cdf, pdf / math.sqrt(2.0)]
    previous = np.ones_like(alpha)
    current = -alpha
    factorial = 2.0
    for order in range(3, 11):
        factorial *= order
        coefficients.append(current * pdf / math.sqrt(factorial))
        previous, current = current, (-alpha) * current - (order - 2) * previous
    return coefficients


def gaussian_relu_layer(
    mean: np.ndarray, covariance: np.ndarray, weight: np.ndarray
) -> dict[str, np.ndarray]:
    w = canonical_f64(weight)
    pre_mean = w.T @ canonical_f64(mean)
    pre_covariance = np.einsum(
        "ij,ia,jb->ab", canonical_f64(covariance), w, w
    )
    pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
    variance = np.maximum(np.diag(pre_covariance), VAR_FLOOR)
    sigma = np.sqrt(variance)
    alpha = pre_mean / sigma
    pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
    cdf = normal_cdf(alpha)
    coefficients = hermite_coefficients(alpha, pdf, cdf)
    post_mean = sigma * coefficients[0]
    correlation = pre_covariance / np.outer(sigma, sigma)
    post_covariance = np.zeros_like(pre_covariance)
    power = correlation.copy()
    for order in range(1, 11):
        post_covariance += np.outer(coefficients[order], coefficients[order]) * power
        power *= correlation
    post_covariance *= np.outer(sigma, sigma)
    exact_variance = np.maximum(
        variance * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
        - post_mean * post_mean,
        VAR_FLOOR,
    )
    diagonal = np.arange(mean.size)
    post_covariance[diagonal, diagonal] = exact_variance
    post_covariance = 0.5 * (post_covariance + post_covariance.T)
    return {
        "pre_mean": canonical_f64(pre_mean),
        "pre_covariance": canonical_f64(pre_covariance),
        "pre_variance": canonical_f64(variance),
        "sigma": canonical_f64(sigma),
        "pdf": canonical_f64(pdf),
        "cdf": canonical_f64(cdf),
        "post_mean": canonical_f64(post_mean),
        "post_covariance": canonical_f64(post_covariance),
    }


def exact_zero_gaussian_relu(pre_covariance: np.ndarray):
    covariance = canonical_f64(pre_covariance)
    variance = np.maximum(np.diag(covariance), VAR_FLOOR)
    sigma = np.sqrt(variance)
    correlation = np.clip(covariance / np.outer(sigma, sigma), -1.0, 1.0)
    raw2 = np.outer(sigma, sigma) * (
        np.sqrt(np.maximum(1.0 - correlation * correlation, 0.0))
        + (math.pi - np.arccos(correlation)) * correlation
    ) / (2.0 * math.pi)
    mean = sigma / SQRT_2PI
    post_covariance = raw2 - np.outer(mean, mean)
    diagonal = np.arange(mean.size)
    raw2[diagonal, diagonal] = 0.5 * variance
    post_covariance[diagonal, diagonal] = variance * (
        0.5 - 1.0 / (2.0 * math.pi)
    )
    return canonical_f64(mean), canonical_f64(
        0.5 * (post_covariance + post_covariance.T)
    ), canonical_f64(0.5 * (raw2 + raw2.T))


def scalar_edgeworth4(
    mean: np.ndarray, variance: np.ndarray, k3: np.ndarray, k4: np.ndarray
) -> np.ndarray:
    mu = canonical_f64(mean)
    var = np.maximum(canonical_f64(variance), VAR_FLOOR)
    third = canonical_f64(k3)
    fourth = canonical_f64(k4)
    if not (mu.shape == var.shape == third.shape == fourth.shape):
        raise AssertionError("Edgeworth field shape mismatch")
    sigma = np.sqrt(var)
    alpha = mu / sigma
    pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
    cdf = normal_cdf(alpha)
    gaussian = sigma * pdf + mu * cdf
    delta3 = (third / 6.0) * (-alpha * pdf / var)
    delta4 = (fourth / 24.0) * (
        (alpha * alpha - 1.0) * pdf / (sigma * var)
    )
    return canonical_f64(np.maximum(gaussian + delta3 + delta4, 0.0))


def cp_k3_host32(lam: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    lam32, p32, q32 = canonical_f32(lam), canonical_f32(p), canonical_f32(q)
    if lam32.size == 0:
        return np.zeros(p32.shape[0], dtype=np.float64)
    return canonical_f64(np.float32(3.0) * ((p32 * p32 * q32) @ lam32))


def cp_k3_factor64(lam: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    lam64, p64, q64 = canonical_f64(lam), canonical_f64(p), canonical_f64(q)
    if lam64.size == 0:
        return np.zeros(p64.shape[0], dtype=np.float64)
    return canonical_f64(3.0 * ((p64 * p64 * q64) @ lam64))


def cp_k21_actions(
    lam: np.ndarray, p: np.ndarray, q: np.ndarray, probe: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lam64, p64, q64, Q = (
        canonical_f64(lam), canonical_f64(p), canonical_f64(q), canonical_f64(probe)
    )
    if lam64.size == 0:
        zero = np.zeros((p64.shape[0], Q.shape[1]), dtype=np.float64)
        return zero.copy(), zero.copy()
    forward = (p64 * p64 * lam64[None, :]) @ (q64.T @ Q)
    forward += 2.0 * (p64 * q64 * lam64[None, :]) @ (p64.T @ Q)
    transpose = q64 @ ((p64 * p64 * lam64[None, :]).T @ Q)
    transpose += 2.0 * p64 @ ((p64 * q64 * lam64[None, :]).T @ Q)
    return canonical_f64(forward), canonical_f64(transpose)


def grow_cp(
    lam: np.ndarray,
    pushed_p: np.ndarray,
    pushed_q: np.ndarray,
    gate: np.ndarray,
    pre_covariance: np.ndarray,
    sigma: np.ndarray,
    density: np.ndarray,
    cap: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    width = gate.size
    post_p = np.concatenate(
        [canonical_f32(gate)[:, None] * canonical_f32(pushed_p),
         canonical_f32(gate)[:, None] * canonical_f32(pre_covariance)], axis=1
    )
    post_q = np.concatenate(
        [canonical_f32(gate)[:, None] * canonical_f32(pushed_q),
         np.eye(width, dtype=np.float32)], axis=1
    )
    post_lam = np.concatenate(
        [canonical_f32(lam), canonical_f32(density / sigma)]
    )
    full = (canonical_f32(post_lam), canonical_f32(post_p), canonical_f32(post_q))
    if post_lam.size <= cap:
        return full, full
    p_norm2 = np.sum(post_p * post_p, axis=0)
    q_norm2 = np.sum(post_q * post_q, axis=0)
    p_dot_q = np.sum(post_p * post_q, axis=0)
    energy = (post_lam * post_lam) * (
        np.float32(3.0) * p_norm2 * p_norm2 * q_norm2
        + np.float32(6.0) * p_norm2 * p_dot_q * p_dot_q
    )
    keep = np.argsort(-energy)[:cap]
    capped = (
        canonical_f32(post_lam[keep]), canonical_f32(post_p[:, keep]),
        canonical_f32(post_q[:, keep]),
    )
    return full, capped


@dataclass(frozen=True)
class HostState:
    reported_rows: np.ndarray
    mu3_gaussian: np.ndarray
    mu3_reported: np.ndarray
    variance3_host: np.ndarray
    covariance2_host_diagonal: np.ndarray
    k3_1792_host32: np.ndarray
    k3_2048_host32: np.ndarray
    k3_1792_factor64: np.ndarray
    k3_2048_factor64: np.ndarray
    k21_1792_forward: np.ndarray
    k21_1792_transpose: np.ndarray
    k21_2048_forward: np.ndarray
    k21_2048_transpose: np.ndarray
    exact_controls: dict[str, object]


def build_host_state(
    weights: np.ndarray, probe: np.ndarray, *, cp_cap: int = PRODUCTION_CAP
) -> HostState:
    w32 = canonical_f32(weights)
    if w32.ndim != 3 or w32.shape[0] < 3 or w32.shape[1] != w32.shape[2]:
        raise AssertionError("host needs at least three square weight layers")
    width = w32.shape[1]
    Q = canonical_f64(probe)
    if Q.shape[0] != width:
        raise AssertionError("probe/host width mismatch")
    w64 = canonical_f64(w32[:3])
    mean0 = np.zeros(width, dtype=np.float64)
    covariance0 = np.eye(width, dtype=np.float64)

    layer1 = gaussian_relu_layer(mean0, covariance0, w64[0])
    zero_p = np.zeros((width, 0), dtype=np.float32)
    zero_lam = np.zeros(0, dtype=np.float32)
    reported1 = scalar_edgeworth4(
        layer1["pre_mean"], layer1["pre_variance"],
        np.zeros(width), np.zeros(width)
    )
    _, cp1 = grow_cp(
        zero_lam, zero_p, zero_p, layer1["cdf"], layer1["pre_covariance"],
        layer1["sigma"], layer1["pdf"], cp_cap,
    )

    lam1, p1, q1 = cp1
    pushed_p2 = canonical_f32(w32[1].T @ p1)
    pushed_q2 = canonical_f32(w32[1].T @ q1)
    layer2 = gaussian_relu_layer(layer1["post_mean"], layer1["post_covariance"], w64[1])
    k3_z2 = cp_k3_host32(lam1, pushed_p2, pushed_q2)
    reported2 = scalar_edgeworth4(
        layer2["pre_mean"], layer2["pre_variance"], k3_z2, np.zeros(width)
    )
    cp2_full, cp2_cap = grow_cp(
        lam1, pushed_p2, pushed_q2, layer2["cdf"], layer2["pre_covariance"],
        layer2["sigma"], layer2["pdf"], cp_cap,
    )

    lam_cap, p_cap, q_cap = cp2_cap
    lam_full, p_full, q_full = cp2_full
    pushed_p3_cap = canonical_f32(w32[2].T @ p_cap)
    pushed_q3_cap = canonical_f32(w32[2].T @ q_cap)
    pushed_p3_full = canonical_f32(w32[2].T @ p_full)
    pushed_q3_full = canonical_f32(w32[2].T @ q_full)
    layer3 = gaussian_relu_layer(layer2["post_mean"], layer2["post_covariance"], w64[2])
    k3_cap_host32 = cp_k3_host32(lam_cap, pushed_p3_cap, pushed_q3_cap)
    k3_full_host32 = cp_k3_host32(lam_full, pushed_p3_full, pushed_q3_full)
    reported3 = scalar_edgeworth4(
        layer3["pre_mean"], layer3["pre_variance"], k3_cap_host32,
        np.zeros(width),
    )

    k3_cap64 = cp_k3_factor64(lam_cap, pushed_p3_cap, pushed_q3_cap)
    k3_full64 = cp_k3_factor64(lam_full, pushed_p3_full, pushed_q3_full)
    k21_cap_f, k21_cap_t = cp_k21_actions(
        lam_cap, pushed_p3_cap, pushed_q3_cap, Q
    )
    k21_full_f, k21_full_t = cp_k21_actions(
        lam_full, pushed_p3_full, pushed_q3_full, Q
    )
    if not np.allclose(k3_cap64, 3.0 * (canonical_f64(pushed_p3_cap) ** 2
                                       * canonical_f64(pushed_q3_cap)) @ canonical_f64(lam_cap),
                       rtol=2e-13, atol=2e-13):
        raise AssertionError("CP1792 factor identity failed")

    exact_h1_mean, exact_h1_covariance, exact_h1_raw2 = exact_zero_gaussian_relu(
        w64[0].T @ w64[0]
    )
    exact_z2_mean = w64[1].T @ exact_h1_mean
    exact_z2_raw2 = w64[1].T @ exact_h1_raw2 @ w64[1]
    exact_z2_covariance = exact_z2_raw2 - np.outer(exact_z2_mean, exact_z2_mean)
    controls = {
        "z1_mean_max_abs": float(np.max(np.abs(layer1["pre_mean"]))),
        "z1_K3": 0.0,
        "z1_K4": 0.0,
        "h1_unit_sigma_K3": 0.3264584120691983,
        "h1_unit_sigma_K4": 0.2797083383527027,
        "h1_cumulants_nonzero": True,
        "exact_z2_premean_host_max_abs": float(
            np.max(np.abs(exact_z2_mean - layer2["pre_mean"]))
        ),
        "exact_z2_covariance_host_frobenius": float(
            np.linalg.norm(exact_z2_covariance - layer2["pre_covariance"])
        ),
        "cp_columns_after_l1": int(lam1.size),
        "cp_columns_before_first_cap": int(lam_full.size),
        "cp_columns_after_first_cap": int(lam_cap.size),
        "cp_columns_removed": int(lam_full.size - lam_cap.size),
        "host32_factor64_k3_gap_max_abs": float(
            np.max(np.abs(k3_cap_host32 - k3_cap64))
        ),
        "exact_h1_covariance_sha256": hash_array(exact_h1_covariance),
    }
    if width == PRODUCTION_WIDTH:
        expected_counts = (1024, 2048, 1792, 256)
        observed_counts = (
            controls["cp_columns_after_l1"], controls["cp_columns_before_first_cap"],
            controls["cp_columns_after_first_cap"], controls["cp_columns_removed"],
        )
        if observed_counts != expected_counts:
            raise AssertionError(f"first-cap count drift: {observed_counts}")
    return HostState(
        reported_rows=canonical_f64(np.stack([reported1, reported2, reported3])),
        mu3_gaussian=layer3["pre_mean"],
        mu3_reported=canonical_f64(w64[2].T @ reported2),
        variance3_host=layer3["pre_variance"],
        covariance2_host_diagonal=canonical_f64(np.diag(layer2["post_covariance"])),
        k3_1792_host32=k3_cap_host32,
        k3_2048_host32=k3_full_host32,
        k3_1792_factor64=k3_cap64,
        k3_2048_factor64=k3_full64,
        k21_1792_forward=k21_cap_f,
        k21_1792_transpose=k21_cap_t,
        k21_2048_forward=k21_full_f,
        k21_2048_transpose=k21_full_t,
        exact_controls=controls,
    )


def forward_to_h3(weights3: np.ndarray, inputs: np.ndarray):
    weights = canonical_f32(weights3)
    value = canonical_f32(inputs)
    if weights.shape[0] != 3 or value.shape[1] != weights.shape[1]:
        raise AssertionError("three-layer forward shape mismatch")
    z1 = canonical_f32(value @ weights[0])
    h1 = canonical_f32(np.maximum(z1, np.float32(0.0)))
    z2 = canonical_f32(h1 @ weights[1])
    h2 = canonical_f32(np.maximum(z2, np.float32(0.0)))
    z3 = canonical_f32(h2 @ weights[2])
    h3 = canonical_f32(np.maximum(z3, np.float32(0.0)))
    return h2, z3, h3


@dataclass(frozen=True)
class StreamStats:
    count: int
    mean_h2: np.ndarray
    variance_h2: np.ndarray
    mean_z3: np.ndarray
    variance_z3: np.ndarray
    k3_z3: np.ndarray
    k4_z3: np.ndarray
    k21_forward: np.ndarray
    k21_transpose: np.ndarray
    mean_h3: np.ndarray
    stream_fingerprint: str


def stream_prefix_statistics(
    weights3: np.ndarray,
    mlp_seed: int,
    replicate: int,
    probe: np.ndarray,
    prefixes: tuple[int, ...] = PRODUCTION_PREFIXES,
) -> dict[int, StreamStats]:
    weights = canonical_f32(weights3)
    Q = canonical_f64(probe)
    width = weights.shape[1]
    if weights.shape != (3, width, width) or Q.shape[0] != width:
        raise AssertionError("stream science shape mismatch")
    levels = tuple(int(value) for value in prefixes)
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError("prefixes must be unique increasing integers")
    if any(value <= 3 or value % BLOCK_ROWS for value in levels):
        raise ValueError("prefixes must be block-aligned and exceed three")
    maximum = levels[-1]
    preimage, seed_int = seed_material(mlp_seed, replicate)

    sums = {
        "h2": np.zeros(width, dtype=np.float64),
        "z3": np.zeros(width, dtype=np.float64),
        "h3": np.zeros(width, dtype=np.float64),
    }
    means: dict[int, dict[str, np.ndarray]] = {}
    pass1_fingerprints: dict[int, str] = {}
    generator = np.random.Generator(np.random.PCG64DXSM(seed_int))
    digest = new_stream_hasher(preimage)
    count = 0
    for index in range(maximum // BLOCK_ROWS):
        block = generator.standard_normal((BLOCK_ROWS, width), dtype=np.float32)
        update_stream_hasher(digest, index, block)
        h2, z3, h3 = forward_to_h3(weights, block)
        for label, value in (("h2", h2), ("z3", z3), ("h3", h3)):
            sums[label] += np.sum(value.astype(np.float64), axis=0, dtype=np.float64)
        count += BLOCK_ROWS
        if count in levels:
            means[count] = {
                label: canonical_f64(total / np.float64(count))
                for label, total in sums.items()
            }
            pass1_fingerprints[count] = digest.hexdigest()
    if count != maximum or tuple(means) != levels:
        raise AssertionError("pass-1 prefix traversal failed")

    accumulators = {
        level: {
            "h2_s2": np.zeros(width, dtype=np.float64),
            "z3_s2": np.zeros(width, dtype=np.float64),
            "z3_s3": np.zeros(width, dtype=np.float64),
            "z3_s4": np.zeros(width, dtype=np.float64),
            "k21_f": np.zeros((width, Q.shape[1]), dtype=np.float64),
            "k21_t": np.zeros((width, Q.shape[1]), dtype=np.float64),
        }
        for level in levels
    }
    generator = np.random.Generator(np.random.PCG64DXSM(seed_int))
    digest = new_stream_hasher(preimage)
    pass2_fingerprints: dict[int, str] = {}
    count = 0
    for index in range(maximum // BLOCK_ROWS):
        block = generator.standard_normal((BLOCK_ROWS, width), dtype=np.float32)
        update_stream_hasher(digest, index, block)
        h2, z3, _ = forward_to_h3(weights, block)
        count += BLOCK_ROWS
        h2_64 = h2.astype(np.float64)
        z3_64 = z3.astype(np.float64)
        for level in levels:
            if count > level:
                continue
            acc = accumulators[level]
            dh2 = h2_64 - means[level]["h2"][None, :]
            dz3 = z3_64 - means[level]["z3"][None, :]
            dz3_2 = dz3 * dz3
            acc["h2_s2"] += np.sum(dh2 * dh2, axis=0, dtype=np.float64)
            acc["z3_s2"] += np.sum(dz3_2, axis=0, dtype=np.float64)
            acc["z3_s3"] += np.sum(dz3_2 * dz3, axis=0, dtype=np.float64)
            acc["z3_s4"] += np.sum(dz3_2 * dz3_2, axis=0, dtype=np.float64)
            acc["k21_f"] += dz3_2.T @ (dz3 @ Q)
            acc["k21_t"] += dz3.T @ (dz3_2 @ Q)
        if count in levels:
            pass2_fingerprints[count] = digest.hexdigest()
            if pass2_fingerprints[count] != pass1_fingerprints[count]:
                raise AssertionError("complete stream pass fingerprint mismatch")

    result: dict[int, StreamStats] = {}
    for level in levels:
        n = np.float64(level)
        acc = accumulators[level]
        m2 = acc["z3_s2"] / n
        m4 = acc["z3_s4"] / n
        factor3 = n / (np.float64(level - 1) * np.float64(level - 2))
        k4 = (
            n * n * ((n + 1.0) * m4 - 3.0 * (n - 1.0) * m2 * m2)
            / (np.float64(level - 1) * np.float64(level - 2)
               * np.float64(level - 3))
        )
        result[level] = StreamStats(
            count=level,
            mean_h2=means[level]["h2"],
            variance_h2=canonical_f64(acc["h2_s2"] / np.float64(level - 1)),
            mean_z3=means[level]["z3"],
            variance_z3=canonical_f64(acc["z3_s2"] / np.float64(level - 1)),
            k3_z3=canonical_f64(factor3 * acc["z3_s3"]),
            k4_z3=canonical_f64(k4),
            k21_forward=canonical_f64(factor3 * acc["k21_f"]),
            k21_transpose=canonical_f64(factor3 * acc["k21_t"]),
            mean_h3=means[level]["h3"],
            stream_fingerprint=pass1_fingerprints[level],
        )
    return result


def cross_product_mse(a: np.ndarray, b: np.ndarray, analytic: np.ndarray) -> float:
    left = canonical_f64(a) - canonical_f64(analytic)
    right = canonical_f64(b) - canonical_f64(analytic)
    if left.shape != right.shape:
        raise AssertionError("cross-product shape mismatch")
    return float(np.sum(left * right, dtype=np.float64) / np.float64(left.size))


def source_chain(host: HostState, teacher: StreamStats, weight3: np.ndarray):
    w = canonical_f64(weight3)
    mu_teacher = canonical_f64(w.T @ teacher.mean_h2)
    delta_diag = canonical_f64(
        (w * w).T @ (teacher.variance_h2 - host.covariance2_host_diagonal)
    )
    delta_total = canonical_f64(teacher.variance_z3 - host.variance3_host)
    delta_offdiag = canonical_f64(delta_total - delta_diag)
    means = (
        host.mu3_gaussian,
        host.mu3_reported,
        mu_teacher,
        mu_teacher,
        mu_teacher,
        mu_teacher,
        mu_teacher,
        mu_teacher,
    )
    variances = (
        host.variance3_host,
        host.variance3_host,
        host.variance3_host,
        host.variance3_host + delta_diag,
        teacher.variance_z3,
        teacher.variance_z3,
        teacher.variance_z3,
        teacher.variance_z3,
    )
    thirds = (
        host.k3_1792_host32,
        host.k3_1792_host32,
        host.k3_1792_host32,
        host.k3_1792_host32,
        host.k3_1792_host32,
        host.k3_2048_host32,
        teacher.k3_z3,
        teacher.k3_z3,
    )
    zeros = np.zeros_like(host.mu3_gaussian)
    fourths = (zeros, zeros, zeros, zeros, zeros, zeros, zeros, teacher.k4_z3)
    predictions = canonical_f64(np.stack([
        scalar_edgeworth4(mu, var, k3, k4)
        for mu, var, k3, k4 in zip(means, variances, thirds, fourths, strict=True)
    ]))
    closure = canonical_f64(teacher.mean_z3 - mu_teacher)
    if np.max(np.abs((delta_diag + delta_offdiag) - delta_total)) > 2e-12:
        raise AssertionError("covariance split closure failed")
    return {
        "predictions": predictions,
        "mu_teacher": mu_teacher,
        "delta_diag": delta_diag,
        "delta_offdiag": delta_offdiag,
        "delta_total": delta_total,
        "z3_affine_closure": closure,
    }


def quartet_metrics(
    host: HostState, streams: list[StreamStats], weight3: np.ndarray
) -> dict[str, object]:
    if len(streams) != 16 or [item.count for item in streams] != [streams[0].count] * 16:
        raise AssertionError("exactly sixteen equal-prefix streams required")
    source = [source_chain(host, item, weight3) for item in streams]
    arm_values = {name: [] for name in SIGNED_MECHANISMS}
    tail_values: list[float] = []
    direct_values = {name: [] for name in DIRECT_ENDPOINTS}
    role_map = []
    for group in range(4):
        A, B, C, D = 4 * group, 4 * group + 1, 4 * group + 2, 4 * group + 3
        role_map.append({"g": group, "A": A, "B": B, "C": C, "D": D})
        chain_a = source[A]["predictions"]
        chain_b = source[B]["predictions"]
        ref_c, ref_d = streams[C].mean_h3, streams[D].mean_h3
        for index, name in enumerate(SIGNED_MECHANISMS):
            values = []
            for chain in (chain_a, chain_b):
                for reference in (ref_c, ref_d):
                    values.append(np.mean(
                        (reference - chain[index]) ** 2
                        - (reference - chain[index + 1]) ** 2,
                        dtype=np.float64,
                    ))
            arm_values[name].append(float(math.fsum(map(float, values)) / 4.0))
        tail = 0.5 * np.mean(
            (ref_c - chain_a[7]) * (ref_d - chain_b[7])
            + (ref_c - chain_b[7]) * (ref_d - chain_a[7]),
            dtype=np.float64,
        )
        tail_values.append(float(tail))

        direct_values["premean_source"].append(cross_product_mse(
            source[A]["mu_teacher"], source[B]["mu_teacher"], host.mu3_reported
        ))
        zero = np.zeros_like(host.mu3_gaussian)
        for label in ("delta_diag", "delta_offdiag", "z3_affine_closure"):
            endpoint = {
                "delta_diag": "diagonal_covariance_action",
                "delta_offdiag": "offdiagonal_covariance_action",
                "z3_affine_closure": "z3_affine_mean_closure",
            }[label]
            direct_values[endpoint].append(cross_product_mse(
                source[A][label], source[B][label], zero
            ))
        direct_values["k3_source"].append(cross_product_mse(
            streams[A].k3_z3, streams[B].k3_z3, host.k3_2048_host32
        ))
        direct_values["k4_energy"].append(cross_product_mse(
            streams[A].k4_z3, streams[B].k4_z3, zero
        ))
        direct_values["k21_forward_source"].append(cross_product_mse(
            streams[A].k21_forward, streams[B].k21_forward,
            host.k21_2048_forward,
        ))
        direct_values["k21_transpose_source"].append(cross_product_mse(
            streams[A].k21_transpose, streams[B].k21_transpose,
            host.k21_2048_transpose,
        ))

    deterministic = {
        "report_vs_carried_premean_mse": float(np.mean(
            (host.mu3_reported - host.mu3_gaussian) ** 2, dtype=np.float64
        )),
        "cp_truncation_k3_host32_mse": float(np.mean(
            (host.k3_2048_host32 - host.k3_1792_host32) ** 2,
            dtype=np.float64,
        )),
        "cp_truncation_k3_factor64_mse": float(np.mean(
            (host.k3_2048_factor64 - host.k3_1792_factor64) ** 2,
            dtype=np.float64,
        )),
        "cp_truncation_k21_forward_mse": float(np.mean(
            (host.k21_2048_forward - host.k21_1792_forward) ** 2,
            dtype=np.float64,
        )),
        "cp_truncation_k21_transpose_mse": float(np.mean(
            (host.k21_2048_transpose - host.k21_1792_transpose) ** 2,
            dtype=np.float64,
        )),
    }
    return {
        "count_per_stream": streams[0].count,
        "quartet_role_map": role_map,
        "arm_effects": {key: values for key, values in arm_values.items()},
        "consumer_tail": tail_values,
        "direct": {key: values for key, values in direct_values.items()},
        "deterministic": deterministic,
        "stream_fingerprints": [item.stream_fingerprint for item in streams],
    }


def student_interval(values: list[float], critical: float) -> dict[str, float]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise AssertionError("four finite quartet values required")
    mean = math.fsum(values) / 4.0
    variance = math.fsum((value - mean) ** 2 for value in values) / 3.0
    standard_error = math.sqrt(max(variance, 0.0)) / 2.0
    radius = critical * standard_error
    return {"mean": mean, "lower": mean - radius, "upper": mean + radius,
            "quartet_sd": math.sqrt(max(variance, 0.0)), "critical": critical}


def classify_signed(intervals: list[dict[str, float]], final: bool) -> str | None:
    epsilon = 1.0e-8
    if all(item["lower"] > epsilon for item in intervals):
        return "APPROX_MATERIAL_BENEFICIAL"
    if all(item["upper"] < -epsilon for item in intervals):
        return "APPROX_MATERIAL_ADVERSE_OR_COMPENSATING"
    if all(item["lower"] >= -epsilon and item["upper"] <= epsilon for item in intervals):
        return "APPROX_BELOW_PILOT_RESOLUTION"
    return "UNRESOLVED_APPROX_NOISE_LIMIT" if final else None


def classify_tail(intervals: list[dict[str, float]], final: bool) -> str | None:
    epsilon = 1.0e-8
    if all(item["lower"] > epsilon for item in intervals):
        return "APPROX_MATERIAL"
    if all(item["upper"] < epsilon for item in intervals):
        return "APPROX_BELOW_PILOT_RESOLUTION"
    return "UNRESOLVED_APPROX_NOISE_LIMIT" if final else None


def analyze_two_ordinals(
    raw: dict[int, dict[int, dict[str, object]]],
    prefixes: tuple[int, ...] = PRODUCTION_PREFIXES,
) -> dict[str, object]:
    if tuple(sorted(raw)) != (0, 1):
        raise AssertionError("analysis requires exactly ordinals zero and one")
    signed_results = {}
    for mechanism in SIGNED_MECHANISMS:
        looks = []
        label = None
        for look_index, prefix in enumerate(prefixes):
            intervals = [student_interval(
                raw[ordinal][prefix]["arm_effects"][mechanism],
                T_CRITICAL_M2[look_index],
            ) for ordinal in (0, 1)]
            looks.append({"N_total": prefix * 16, "N_per_stream": prefix,
                          "ordinal_intervals": intervals})
            label = classify_signed(intervals, look_index == len(prefixes) - 1)
            if label is not None:
                break
        signed_results[mechanism] = {
            "label": label,
            "stopped_N_total": looks[-1]["N_total"],
            "inference_qualification": "APPROXIMATE_GAUSSIAN_FOUR_QUARTET_T_DF3",
            "looks": looks,
        }

    tail_looks = []
    tail_label = None
    for look_index, prefix in enumerate(prefixes):
        intervals = [student_interval(
            raw[ordinal][prefix]["consumer_tail"], T_CRITICAL_M2[look_index]
        ) for ordinal in (0, 1)]
        tail_looks.append({"N_total": prefix * 16, "N_per_stream": prefix,
                           "ordinal_intervals": intervals})
        tail_label = classify_tail(intervals, look_index == len(prefixes) - 1)
        if tail_label is not None:
            break

    direct = {}
    third_order = {"k3_source", "k21_forward_source", "k21_transpose_source"}
    for endpoint in DIRECT_ENDPOINTS:
        criticals = T_CRITICAL_M6 if endpoint in third_order else T_CRITICAL_M2
        direct[endpoint] = []
        for look_index, prefix in enumerate(prefixes):
            direct[endpoint].append({
                "N_total": prefix * 16,
                "N_per_stream": prefix,
                "ordinal_intervals": [student_interval(
                    raw[ordinal][prefix]["direct"][endpoint], criticals[look_index]
                ) for ordinal in (0, 1)],
                "material_threshold_applied": False,
            })
    return {
        "signed_mechanisms": signed_results,
        "consumer_tail": {
            "label": tail_label,
            "stopped_N_total": tail_looks[-1]["N_total"],
            "inference_qualification": "APPROXIMATE_GAUSSIAN_FOUR_QUARTET_T_DF3",
            "looks": tail_looks,
        },
        "direct_diagnostics": direct,
        "coverage_scope": "99_PERCENT_WITHIN_EACH_DECLARED_FAMILY_ACROSS_LOOKS_AND_MEMBERS_ONLY",
        "omnibus_or_winner_claim": False,
    }


def synthetic_self_test() -> dict[str, object]:
    rng = np.random.Generator(np.random.PCG64DXSM(2026082401))
    width, rank = 16, 4
    weights = canonical_f32(rng.standard_normal((3, width, width), dtype=np.float32)
                              / np.float32(math.sqrt(width)))
    probe = walsh_probe(width, rank)
    host = build_host_state(weights, probe, cp_cap=24)
    streams = [
        stream_prefix_statistics(weights, -271828, replicate, probe, (256,)) [256]
        for replicate in range(16)
    ]
    metrics = quartet_metrics(host, streams, weights[2])
    if tuple(metrics["arm_effects"]) != SIGNED_MECHANISMS:
        raise AssertionError("synthetic arm namespace drift")
    if tuple(metrics["direct"]) != DIRECT_ENDPOINTS:
        raise AssertionError("synthetic direct namespace drift")
    if host.exact_controls["cp_columns_before_first_cap"] != 32:
        raise AssertionError("synthetic first-cap pre-count drift")
    if host.exact_controls["cp_columns_after_first_cap"] != 24:
        raise AssertionError("synthetic first-cap post-count drift")
    return {
        "status": "PASS_V7_SMALL_THREE_LAYER_SIXTEEN_STREAM_RUNTIME_REHEARSAL",
        "width": width,
        "rank": rank,
        "streams": 16,
        "prefix": 256,
        "arm_mechanisms": list(SIGNED_MECHANISMS),
        "direct_endpoints": list(DIRECT_ENDPOINTS),
        "first_cap_counts": [16, 32, 24, 8],
        "host_reported_sha256": hash_array(host.reported_rows),
        "first_stream_fingerprint": streams[0].stream_fingerprint,
        "quartet_metric_sha256": hashlib.sha256(
            repr(metrics).encode("utf-8")
        ).hexdigest(),
    }
