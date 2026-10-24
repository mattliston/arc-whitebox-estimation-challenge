"""Target-free mathematical core for the Phase 2 teacher feasibility pilot.

This module contains no dataset loader, target name, scoring code, fitting
code, or execution authority.  It is written so the static audit can exercise
all identities at small width without touching a public row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


LITERAL = "p2-teacher-v1"
FIELD_ORDER = ("mean", "variance", "K3", "K4")
ARM_MASKS = tuple(f"{mask:04b}" for mask in range(16))
TOTAL_LEVELS = (1 << 14, 1 << 16, 1 << 18)
PER_REPLICATE_PREFIXES = tuple(level // 8 for level in TOTAL_LEVELS)
REPLICATES = 8
BANKS = {"A": (0, 1, 2, 3), "B": (4, 5, 6, 7)}
VAR_FLOOR = 1.0e-30
SQRT_2PI = math.sqrt(2.0 * math.pi)


def canonical_f32le(array: np.ndarray) -> np.ndarray:
    """Return a finite C-contiguous little-endian float32 array."""

    result = np.ascontiguousarray(array, dtype="<f4")
    if not np.isfinite(result).all():
        raise ValueError("non-finite float32 payload")
    return result


def canonical_f64le(array: np.ndarray) -> np.ndarray:
    """Return a finite C-contiguous little-endian float64 array."""

    result = np.ascontiguousarray(array, dtype="<f8")
    if not np.isfinite(result).all():
        raise ValueError("non-finite float64 payload")
    return result


def payload_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def signed_decimal_ascii(value: int) -> bytes:
    """Audited signed-decimal encoding: no plus, padding, or whitespace."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("boolean is not a signed integer seed")
    integer = int(value)
    if integer != value:
        raise ValueError("seed must be an exact integer")
    encoded = str(integer).encode("ascii")
    if encoded == b"-0" or encoded.startswith(b"+") or b" " in encoded:
        raise AssertionError("noncanonical signed decimal")
    return encoded


def stream_seed_preimage(network_seed: int, sampler: str, replicate: int) -> bytes:
    if sampler not in ("rqmc", "iid"):
        raise ValueError(f"unknown sampler: {sampler}")
    if replicate not in range(REPLICATES):
        raise ValueError(f"bad replicate: {replicate}")
    return b"|".join(
        (
            LITERAL.encode("ascii"),
            b"v2-phase2",
            signed_decimal_ascii(network_seed),
            sampler.encode("ascii"),
            str(replicate).encode("ascii"),
        )
    )


def stream_seed(network_seed: int, sampler: str, replicate: int) -> int:
    """Derive a target-independent 64-bit stream seed."""

    material = stream_seed_preimage(network_seed, sampler, replicate)
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def iter_normal_blocks(
    *,
    width: int,
    network_seed: int,
    sampler: str,
    replicate: int,
    count: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield one nested stream in order; callers snapshot prefixes in place."""

    if count <= 0 or batch_size <= 0 or count % batch_size:
        raise ValueError("count must be a positive multiple of batch_size")
    seed64 = stream_seed(network_seed, sampler, replicate)
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    almost_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    if sampler == "rqmc":
        engine = qmc.Sobol(d=width, scramble=True, bits=64, seed=seed64)
        uniform: Callable[[int], np.ndarray] = engine.random
    else:
        generator = np.random.Generator(np.random.PCG64DXSM(seed64))
        uniform = lambda n: generator.random((n, width), dtype=np.float64)

    emitted = 0
    while emitted < count:
        take = min(batch_size, count - emitted)
        u = uniform(take)
        np.clip(u, tiny, almost_one, out=u)
        block = canonical_f32le(ndtri(u))
        emitted += take
        yield block
    if emitted != count:
        raise AssertionError("stream length mismatch")


@dataclass
class RawTape:
    """Mergeable raw sums; normalization and nonlinear maps happen later."""

    count: int
    pre_powers: np.ndarray  # [depth, 4, width], sums of z^1...z^4
    post_sum: np.ndarray  # [depth, width]
    post_raw2: np.ndarray  # [depth, width, width], sum h_i h_j
    pre_sq_cross: np.ndarray  # [depth, width, width], sum z_i^2 z_j

    @classmethod
    def zeros(cls, depth: int, width: int) -> "RawTape":
        return cls(
            count=0,
            pre_powers=np.zeros((depth, 4, width), dtype=np.float64),
            post_sum=np.zeros((depth, width), dtype=np.float64),
            post_raw2=np.zeros((depth, width, width), dtype=np.float64),
            pre_sq_cross=np.zeros((depth, width, width), dtype=np.float64),
        )

    @property
    def depth(self) -> int:
        return int(self.pre_powers.shape[0])

    @property
    def width(self) -> int:
        return int(self.pre_powers.shape[2])

    def validate(self) -> None:
        d, powers, n = self.pre_powers.shape
        if powers != 4:
            raise AssertionError("expected four raw preactivation powers")
        if self.post_sum.shape != (d, n):
            raise AssertionError("post_sum shape mismatch")
        if self.post_raw2.shape != (d, n, n):
            raise AssertionError("post_raw2 shape mismatch")
        if self.pre_sq_cross.shape != (d, n, n):
            raise AssertionError("pre_sq_cross shape mismatch")
        if self.count < 0:
            raise AssertionError("negative sample count")
        for value in (
            self.pre_powers,
            self.post_sum,
            self.post_raw2,
            self.pre_sq_cross,
        ):
            if value.dtype != np.float64 or not np.isfinite(value).all():
                raise AssertionError("raw tape must be finite float64")

    def copy(self) -> "RawTape":
        return RawTape(
            count=self.count,
            pre_powers=self.pre_powers.copy(),
            post_sum=self.post_sum.copy(),
            post_raw2=self.post_raw2.copy(),
            pre_sq_cross=self.pre_sq_cross.copy(),
        )

    def add_(self, other: "RawTape") -> None:
        self.validate()
        other.validate()
        if self.pre_powers.shape != other.pre_powers.shape:
            raise AssertionError("cannot merge different tape shapes")
        self.count += other.count
        self.pre_powers += other.pre_powers
        self.post_sum += other.post_sum
        self.post_raw2 += other.post_raw2
        self.pre_sq_cross += other.pre_sq_cross


def merge_raw_tapes(tapes: Sequence[RawTape]) -> RawTape:
    """Pool raw sums.  Never average already-centered moments."""

    if not tapes:
        raise ValueError("at least one tape is required")
    merged = RawTape.zeros(tapes[0].depth, tapes[0].width)
    for tape in tapes:
        merged.add_(tape)
    return merged


def accumulate_cpu_block(tape: RawTape, weights: np.ndarray, inputs: np.ndarray) -> None:
    """Authoritative CPU block: float32 forward, float64 raw products."""

    tape.validate()
    weights32 = canonical_f32le(weights)
    h = canonical_f32le(inputs)
    if weights32.shape != (tape.depth, tape.width, tape.width):
        raise AssertionError("weight shape mismatch")
    if h.ndim != 2 or h.shape[1] != tape.width:
        raise AssertionError("input shape mismatch")

    for layer in range(tape.depth):
        z = canonical_f32le(h @ weights32[layer])
        z64 = z.astype(np.float64)
        z2 = z64 * z64
        tape.pre_powers[layer, 0] += np.sum(z64, axis=0)
        tape.pre_powers[layer, 1] += np.sum(z2, axis=0)
        tape.pre_powers[layer, 2] += np.sum(z2 * z64, axis=0)
        tape.pre_powers[layer, 3] += np.sum(z2 * z2, axis=0)
        tape.pre_sq_cross[layer] += z2.T @ z64
        h = np.maximum(z, np.float32(0.0), dtype=np.float32)
        h64 = h.astype(np.float64)
        tape.post_sum[layer] += np.sum(h64, axis=0)
        tape.post_raw2[layer] += h64.T @ h64
    tape.count += int(inputs.shape[0])


def accumulate_mps_stream(
    weights: np.ndarray,
    blocks: Iterable[np.ndarray],
    prefixes: Sequence[int],
    guard: Callable[[str], None] | None = None,
) -> dict[int, RawTape]:
    """MPS-f32 accelerator path; CPU-f64 normalization remains authoritative.

    Torch is imported lazily so a CPU-only preflight cannot initialize Metal.
    Products are accumulated on MPS for at most 2,048 samples, transferred,
    and merged into a CPU float64 tape.  This makes CPU authority literal and
    charges the transfers to the end-to-end speed gate.  The path is usable
    only after the explicit saved-input parity and throughput gate.
    """

    import torch  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    weights32 = canonical_f32le(weights)
    depth, width, width2 = weights32.shape
    if width != width2:
        raise AssertionError("MPS weights must be square")
    prefix_tuple = tuple(int(value) for value in prefixes)
    if tuple(sorted(set(prefix_tuple))) != prefix_tuple:
        raise AssertionError("MPS prefixes must be unique and increasing")

    device = torch.device("mps")
    if guard is not None:
        guard("mps_before_weight_transfer")
    w_device = torch.from_numpy(weights32).to(device=device, dtype=torch.float32)
    def device_zeros():
        if guard is not None:
            guard("mps_before_accumulator_allocation")
        powers = torch.zeros((depth, 4, width), device=device, dtype=torch.float32)
        h_sum = torch.zeros((depth, width), device=device, dtype=torch.float32)
        h_raw2 = torch.zeros(
            (depth, width, width), device=device, dtype=torch.float32
        )
        z2_cross = torch.zeros_like(h_raw2)
        return powers, h_sum, h_raw2, z2_cross

    pre_powers, post_sum, post_raw2, pre_sq_cross = device_zeros()
    cpu_total = RawTape.zeros(depth, width)
    snapshots: dict[int, RawTape] = {}
    count = 0
    chunk_count = 0

    def flush_chunk() -> None:
        nonlocal pre_powers, post_sum, post_raw2, pre_sq_cross, chunk_count
        if chunk_count == 0:
            return
        if guard is not None:
            guard("mps_before_chunk_transfer")
        torch.mps.synchronize()
        chunk = RawTape(
            count=chunk_count,
            pre_powers=pre_powers.detach().cpu().numpy().astype(np.float64),
            post_sum=post_sum.detach().cpu().numpy().astype(np.float64),
            post_raw2=post_raw2.detach().cpu().numpy().astype(np.float64),
            pre_sq_cross=pre_sq_cross.detach().cpu().numpy().astype(np.float64),
        )
        cpu_total.add_(chunk)
        if guard is not None:
            guard("mps_after_CPU_float64_chunk_merge")
        pre_powers, post_sum, post_raw2, pre_sq_cross = device_zeros()
        chunk_count = 0

    for block in blocks:
        if guard is not None:
            guard("mps_before_sample_block")
        block32 = canonical_f32le(block)
        if block32.ndim != 2 or block32.shape[1] != width:
            raise AssertionError("MPS input shape mismatch")
        h = torch.from_numpy(block32).to(device=device, dtype=torch.float32)
        for layer in range(depth):
            z = h @ w_device[layer]
            z2 = z * z
            pre_powers[layer, 0] += torch.sum(z, dim=0)
            pre_powers[layer, 1] += torch.sum(z2, dim=0)
            pre_powers[layer, 2] += torch.sum(z2 * z, dim=0)
            pre_powers[layer, 3] += torch.sum(z2 * z2, dim=0)
            pre_sq_cross[layer] += z2.T @ z
            h = torch.relu(z)
            post_sum[layer] += torch.sum(h, dim=0)
            post_raw2[layer] += h.T @ h
        count += int(block32.shape[0])
        chunk_count += int(block32.shape[0])
        if guard is not None:
            guard("mps_after_sample_block")
        if chunk_count > 2048:
            raise AssertionError("MPS accumulation chunk exceeded 2,048 samples")
        if chunk_count == 2048 or count in prefix_tuple:
            flush_chunk()
        if count in prefix_tuple:
            if cpu_total.count != count:
                raise AssertionError("CPU-f64 MPS merge count mismatch")
            snapshots[count] = cpu_total.copy()
            if guard is not None:
                guard(f"mps_after_nested_prefix_{count}")
        elif any(count > prefix for prefix in prefix_tuple if prefix not in snapshots):
            raise AssertionError("a block crossed an MPS prefix")
    flush_chunk()
    if cpu_total.count != count:
        raise AssertionError("final CPU-f64 MPS merge count mismatch")
    if tuple(snapshots) != prefix_tuple:
        raise AssertionError("MPS stream did not hit every prefix")
    return snapshots


@dataclass(frozen=True)
class TeacherTape:
    count: int
    pre_mean: np.ndarray
    pre_covariance: np.ndarray
    pre_variance: np.ndarray
    pre_K3: np.ndarray
    pre_K4: np.ndarray
    pre_K21: np.ndarray
    post_mean: np.ndarray
    post_covariance: np.ndarray


def zero_mean_gaussian_relu_moments(
    pre_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact mean, covariance, and raw second moment of ReLU(z), z Gaussian.

    This is the layer-1 analytic foothold.  It is also the exact source for the
    layer-2 pre-mean/covariance after the second real weight is applied.
    """

    covariance = np.asarray(pre_covariance, dtype=np.float64)
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
    exact_variance = variance * (0.5 - 1.0 / (2.0 * math.pi))
    post_covariance[diagonal, diagonal] = exact_variance
    raw2[diagonal, diagonal] = 0.5 * variance
    post_covariance = 0.5 * (post_covariance + post_covariance.T)
    raw2 = 0.5 * (raw2 + raw2.T)
    return (
        canonical_f64le(mean),
        canonical_f64le(post_covariance),
        canonical_f64le(raw2),
    )


def normalize_raw_tape(
    raw: RawTape,
    weights: np.ndarray,
    guard: Callable[[str], None] | None = None,
) -> TeacherTape:
    """Center one pooled bank with a bank-compatible Gaussian source."""

    raw.validate()
    if raw.count <= 0:
        raise ValueError("cannot normalize an empty tape")
    weights64 = canonical_f64le(weights)
    d, n, _ = weights64.shape
    if (d, n) != (raw.depth, raw.width):
        raise AssertionError("weight/tape mismatch")
    inverse_count = 1.0 / raw.count
    sampled_pre_mean = raw.pre_powers[:, 0] * inverse_count
    sampled_m2 = raw.pre_powers[:, 1] * inverse_count
    sampled_m3 = raw.pre_powers[:, 2] * inverse_count
    sampled_m4 = raw.pre_powers[:, 3] * inverse_count
    post_mean = raw.post_sum * inverse_count
    post_raw2 = raw.post_raw2 * inverse_count
    post_covariance = post_raw2 - np.einsum("li,lj->lij", post_mean, post_mean)
    post_covariance = 0.5 * (
        post_covariance + np.transpose(post_covariance, (0, 2, 1))
    )

    pre_mean = sampled_pre_mean.copy()
    pre_covariance = np.empty((d, n, n), dtype=np.float64)
    pre_raw2 = np.empty_like(pre_covariance)
    # Teacher-compatible analytic Gaussian source for layer 1.
    pre_mean[0] = 0.0
    if guard is not None:
        guard("normalize_before_layer1_source_covariance")
    pre_raw2[0] = weights64[0].T @ weights64[0]
    pre_covariance[0] = pre_raw2[0]
    exact_post_mean1, exact_post_covariance1, exact_post_raw2_1 = (
        zero_mean_gaussian_relu_moments(pre_covariance[0])
    )
    post_mean[0] = exact_post_mean1
    post_covariance[0] = exact_post_covariance1
    post_raw2[0] = exact_post_raw2_1
    if d > 1:
        # Layer 2 has an exact pre-mean/covariance even though its ReLU mean is
        # already a non-Gaussian readout problem.
        if guard is not None:
            guard("normalize_before_exact_WtCW_layer_2")
        pre_mean[1] = weights64[1].T @ exact_post_mean1
        pre_raw2[1] = weights64[1].T @ exact_post_raw2_1 @ weights64[1]
        pre_covariance[1] = pre_raw2[1] - np.outer(pre_mean[1], pre_mean[1])
        if guard is not None:
            guard("normalize_after_exact_WtCW_layer_2")
    for layer in range(2, d):
        if guard is not None:
            guard(f"normalize_before_WtCW_layer_{layer + 1}")
        pre_raw2[layer] = weights64[layer].T @ post_raw2[layer - 1] @ weights64[layer]
        pre_covariance[layer] = pre_raw2[layer] - np.outer(
            pre_mean[layer], pre_mean[layer]
        )
        if guard is not None:
            guard(f"normalize_after_WtCW_layer_{layer + 1}")
    pre_covariance = 0.5 * (
        pre_covariance + np.transpose(pre_covariance, (0, 2, 1))
    )
    pre_variance = np.maximum(
        np.diagonal(pre_covariance, axis1=1, axis2=2), VAR_FLOOR
    )

    # Use the diagonal of the same W^T E[hh^T] W source that supplies the full
    # covariance; the directly accumulated z^2 sum is retained as an audit
    # cross-check, not mixed into the authoritative centered state.
    compatible_m2 = np.diagonal(pre_raw2, axis1=1, axis2=2)
    centered3 = (
        sampled_m3
        - 3.0 * pre_mean * compatible_m2
        + 2.0 * pre_mean**3
    )
    centered4 = (
        sampled_m4
        - 4.0 * pre_mean * sampled_m3
        + 6.0 * pre_mean**2 * compatible_m2
        - 3.0 * pre_mean**4
    )
    pre_K3 = centered3
    pre_K4 = centered4 - 3.0 * np.maximum(
        compatible_m2 - pre_mean**2, VAR_FLOOR
    ) ** 2

    # Replace every layer-1 sampled cumulant by the exact Gaussian source.
    pre_K3[0] = 0.0
    pre_K4[0] = 0.0

    pre_K21 = np.empty((d, n, n), dtype=np.float64)
    raw_sq_cross = raw.pre_sq_cross * inverse_count
    for layer in range(d):
        if guard is not None:
            guard(f"normalize_before_K21_layer_{layer + 1}")
        mu = pre_mean[layer]
        m2_diag = np.diag(pre_raw2[layer])
        pre_K21[layer] = (
            raw_sq_cross[layer]
            - m2_diag[:, None] * mu[None, :]
            - 2.0 * mu[:, None] * pre_raw2[layer]
            + 2.0 * (mu * mu)[:, None] * mu[None, :]
        )
        if guard is not None:
            guard(f"normalize_after_K21_layer_{layer + 1}")
    pre_K21[0] = 0.0
    # Floating contractions may differ on the diagonal; the convention is exact.
    diagonal = np.arange(n)
    pre_K21[:, diagonal, diagonal] = pre_K3

    result = TeacherTape(
        count=raw.count,
        pre_mean=canonical_f64le(pre_mean),
        pre_covariance=canonical_f64le(pre_covariance),
        pre_variance=canonical_f64le(pre_variance),
        pre_K3=canonical_f64le(pre_K3),
        pre_K4=canonical_f64le(pre_K4),
        pre_K21=canonical_f64le(pre_K21),
        post_mean=canonical_f64le(post_mean),
        post_covariance=canonical_f64le(post_covariance),
    )
    return result


def scalar_edgeworth4(
    mean: np.ndarray,
    variance: np.ndarray,
    K3: np.ndarray,
    K4: np.ndarray,
) -> np.ndarray:
    """Frozen fourth-order scalar ReLU readout, with no fitted coefficient."""

    mu = np.asarray(mean, dtype=np.float64)
    var = np.maximum(np.asarray(variance, dtype=np.float64), VAR_FLOOR)
    k3 = np.asarray(K3, dtype=np.float64)
    k4 = np.asarray(K4, dtype=np.float64)
    if not (mu.shape == var.shape == k3.shape == k4.shape):
        raise AssertionError("readout field shape mismatch")
    sigma = np.sqrt(var)
    alpha = mu / sigma
    pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
    cdf = ndtr(alpha)
    gaussian = sigma * pdf + mu * cdf
    correction3 = (k3 / 6.0) * (-alpha * pdf / var)
    correction4 = (k4 / 24.0) * (
        (alpha * alpha - 1.0) * pdf / (sigma * var)
    )
    return canonical_f64le(np.maximum(gaussian + correction3 + correction4, 0.0))


@dataclass(frozen=True)
class HostTape:
    pre_mean: np.ndarray
    pre_covariance: np.ndarray
    pre_variance: np.ndarray
    pre_K3: np.ndarray
    pre_K21_probe: np.ndarray
    pre_K4: np.ndarray
    post_mean: np.ndarray
    post_covariance: np.ndarray


def _hermite_coefficients(alpha: np.ndarray, pdf: np.ndarray, cdf: np.ndarray) -> list[np.ndarray]:
    coefficients = [pdf + alpha * cdf, cdf, pdf / math.sqrt(2.0)]
    previous = np.ones_like(alpha)
    current = -alpha
    factorial = 2.0
    for order in range(3, 11):
        factorial *= order
        coefficients.append(current * pdf / math.sqrt(factorial))
        previous, current = current, (-alpha) * current - (order - 2) * previous
    return coefficients


def _cp_k3(lam: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if lam.size == 0:
        return np.zeros(p.shape[0], dtype=np.float64)
    return 3.0 * ((p * p * q) @ lam)


def _cp_K21_probe(
    lam: np.ndarray, p: np.ndarray, q: np.ndarray, probe: np.ndarray
) -> np.ndarray:
    """Read T[i,i,j] @ probe without materializing the dense K21 plane."""

    if lam.size == 0:
        return np.zeros((p.shape[0], probe.shape[1]), dtype=np.float64)
    p64 = p.astype(np.float64)
    q64 = q.astype(np.float64)
    lam64 = lam.astype(np.float64)
    term1 = (p64 * p64 * lam64[None, :]) @ (q64.T @ probe)
    term2 = 2.0 * (p64 * q64 * lam64[None, :]) @ (p64.T @ probe)
    return term1 + term2


def frozen_cp1792_host(
    weights: np.ndarray,
    probe: np.ndarray,
    guard: Callable[[str], None] | None = None,
) -> HostTape:
    """Literal NumPy port of the frozen cap-1792/Hermite-10 host state."""

    weights32 = canonical_f32le(weights)
    weights64 = canonical_f64le(weights)
    probe64 = canonical_f64le(probe)
    depth, width, width2 = weights32.shape
    if width != width2 or probe64.shape[0] != width:
        raise AssertionError("host shape mismatch")

    mean = np.zeros(width, dtype=np.float64)
    covariance = np.eye(width, dtype=np.float64)
    identity32 = np.eye(width, dtype=np.float32)
    cp_lambda = np.zeros(0, dtype=np.float32)
    cp_p = np.zeros((width, 0), dtype=np.float32)
    cp_q = np.zeros((width, 0), dtype=np.float32)

    pre_means: list[np.ndarray] = []
    pre_covariances: list[np.ndarray] = []
    pre_variances: list[np.ndarray] = []
    pre_k3s: list[np.ndarray] = []
    pre_k21_probes: list[np.ndarray] = []
    post_means: list[np.ndarray] = []
    post_covariances: list[np.ndarray] = []

    for layer, (w32, w64) in enumerate(zip(weights32, weights64, strict=True)):
        if guard is not None:
            guard(f"host_before_layer_{layer + 1}")
        pre_mean = w64.T @ mean
        # Preserve the frozen host's contraction path, not merely its algebra.
        pre_covariance = np.einsum("ij,ia,jb->ab", covariance, w64, w64)
        pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
        pre_variance = np.maximum(np.diag(pre_covariance), VAR_FLOOR)
        sigma = np.sqrt(pre_variance)
        alpha = pre_mean / sigma
        pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
        cdf = ndtr(alpha)
        coefficients = _hermite_coefficients(alpha, pdf, cdf)
        gaussian_mean = sigma * coefficients[0]

        if cp_lambda.size:
            pushed_p = canonical_f32le(w32.T @ cp_p)
            pushed_q = canonical_f32le(w32.T @ cp_q)
        else:
            pushed_p = cp_p
            pushed_q = cp_q
        k3 = _cp_k3(cp_lambda, pushed_p, pushed_q)
        k21_probe = _cp_K21_probe(cp_lambda, pushed_p, pushed_q, probe64)

        correlation = pre_covariance / np.outer(sigma, sigma)
        post_covariance = np.zeros((width, width), dtype=np.float64)
        power = correlation.copy()
        for order in range(1, 11):
            coefficient = coefficients[order]
            post_covariance += np.outer(coefficient, coefficient) * power
            power *= correlation
        post_covariance *= np.outer(sigma, sigma)
        exact_variance = np.maximum(
            pre_variance
            * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
            - gaussian_mean * gaussian_mean,
            VAR_FLOOR,
        )
        tail = np.maximum(exact_variance - np.diag(post_covariance), 0.0)
        post_covariance += np.diag(tail)
        post_covariance = 0.5 * (post_covariance + post_covariance.T)

        pre_means.append(pre_mean)
        pre_covariances.append(pre_covariance)
        pre_variances.append(pre_variance)
        pre_k3s.append(k3)
        pre_k21_probes.append(k21_probe)
        post_means.append(gaussian_mean)
        post_covariances.append(post_covariance)
        mean = gaussian_mean
        covariance = post_covariance

        if layer + 1 < depth:
            post_p = np.concatenate(
                [cdf.astype(np.float32)[:, None] * pushed_p,
                 cdf.astype(np.float32)[:, None] * pre_covariance.astype(np.float32)],
                axis=1,
            )
            post_q = np.concatenate(
                [cdf.astype(np.float32)[:, None] * pushed_q, identity32], axis=1
            )
            post_lambda = np.concatenate(
                [cp_lambda, (pdf / sigma).astype(np.float32)]
            )
            if post_lambda.size > 1792:
                p_norm2 = np.sum(post_p * post_p, axis=0)
                q_norm2 = np.sum(post_q * post_q, axis=0)
                p_dot_q = np.sum(post_p * post_q, axis=0)
                energy = (post_lambda * post_lambda) * (
                    3.0 * p_norm2 * p_norm2 * q_norm2
                    + 6.0 * p_norm2 * p_dot_q * p_dot_q
                )
                keep = np.argsort(-energy)[:1792]
                post_lambda = post_lambda[keep]
                post_p = post_p[:, keep]
                post_q = post_q[:, keep]
            cp_lambda, cp_p, cp_q = post_lambda, post_p, post_q
        if guard is not None:
            guard(f"host_after_layer_{layer + 1}")

    return HostTape(
        pre_mean=canonical_f64le(np.stack(pre_means)),
        pre_covariance=canonical_f64le(np.stack(pre_covariances)),
        pre_variance=canonical_f64le(np.stack(pre_variances)),
        pre_K3=canonical_f64le(np.stack(pre_k3s)),
        pre_K21_probe=canonical_f64le(np.stack(pre_k21_probes)),
        pre_K4=np.zeros((depth, width), dtype=np.float64),
        post_mean=canonical_f64le(np.stack(post_means)),
        post_covariance=canonical_f64le(np.stack(post_covariances)),
    )


def factorial_predictions(host: HostTape, teacher: TeacherTape) -> np.ndarray:
    """Return all masks in numeric order, shape [16, depth, width]."""

    predictions: list[np.ndarray] = []
    for mask in ARM_MASKS:
        fields = []
        host_fields = (
            host.pre_mean,
            host.pre_variance,
            host.pre_K3,
            host.pre_K4,
        )
        teacher_fields = (
            teacher.pre_mean,
            teacher.pre_variance,
            teacher.pre_K3,
            teacher.pre_K4,
        )
        for bit, host_field, teacher_field in zip(
            mask, host_fields, teacher_fields, strict=True
        ):
            fields.append(teacher_field if bit == "1" else host_field)
        predictions.append(scalar_edgeworth4(*fields))
    return canonical_f64le(np.stack(predictions))


@dataclass(frozen=True)
class CovarianceAction:
    diagonal: np.ndarray
    offdiagonal: np.ndarray
    total: np.ndarray
    probe: np.ndarray


def covariance_action_split(
    host_post_covariance: np.ndarray,
    teacher_post_covariance: np.ndarray,
    next_weight: np.ndarray,
    probe: np.ndarray,
) -> CovarianceAction:
    """Exact diagonal/off-diagonal split of diag(W.T @ DeltaC @ W)."""

    delta = np.asarray(host_post_covariance, dtype=np.float64) - np.asarray(
        teacher_post_covariance, dtype=np.float64
    )
    weight = np.asarray(next_weight, dtype=np.float64)
    probe64 = np.asarray(probe, dtype=np.float64)
    diagonal = np.sum((weight * weight) * np.diag(delta)[:, None], axis=0)
    pushed = delta @ weight
    total = np.sum(weight * pushed, axis=0)
    offdiagonal = total - diagonal
    projected = probe64.T @ (weight.T @ pushed) @ probe64
    if not np.allclose(diagonal + offdiagonal, total, rtol=2e-13, atol=2e-13):
        raise AssertionError("covariance action split identity failed")
    return CovarianceAction(
        diagonal=canonical_f64le(diagonal),
        offdiagonal=canonical_f64le(offdiagonal),
        total=canonical_f64le(total),
        probe=canonical_f64le(projected),
    )


def gaussian_post_covariance_h10(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Frozen Hermite-10 Gaussian covariance consumer with exact diagonal."""

    mu = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    var = np.maximum(np.diag(cov), VAR_FLOOR)
    sigma = np.sqrt(var)
    alpha = mu / sigma
    pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
    cdf = ndtr(alpha)
    coefficients = _hermite_coefficients(alpha, pdf, cdf)
    post_mean = sigma * coefficients[0]
    correlation = cov / np.outer(sigma, sigma)
    result = np.zeros_like(cov)
    power = correlation.copy()
    for order in range(1, 11):
        result += np.outer(coefficients[order], coefficients[order]) * power
        power *= correlation
    result *= np.outer(sigma, sigma)
    exact_variance = np.maximum(
        var * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
        - post_mean * post_mean,
        VAR_FLOOR,
    )
    diagonal = np.arange(mu.size)
    result[diagonal, diagonal] = exact_variance
    return canonical_f64le(0.5 * (result + result.T))


def teacher_compatible_K21_consumer(
    mean: np.ndarray,
    covariance: np.ndarray,
    K3: np.ndarray,
    K21: np.ndarray,
) -> np.ndarray:
    """One-step bivariate correction using one coherent teacher source."""

    mu = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    k3 = np.asarray(K3, dtype=np.float64)
    k21 = np.asarray(K21, dtype=np.float64)
    var = np.maximum(np.diag(cov), VAR_FLOOR)
    sigma = np.sqrt(var)
    alpha = mu / sigma
    pdf = np.exp(-0.5 * alpha * alpha) / SQRT_2PI
    density_zero = pdf / sigma
    regression = cov / var[:, None]
    conditional_mean = mu[None, :] - regression * mu[:, None]
    conditional_variance = np.maximum(
        var[None, :] - regression * cov, VAR_FLOOR
    )
    conditional_sigma = np.sqrt(conditional_variance)
    conditional_alpha = conditional_mean / conditional_sigma
    conditional_pdf = np.exp(-0.5 * conditional_alpha**2) / SQRT_2PI
    conditional_cdf = ndtr(conditional_alpha)
    conditional_relu_mean = (
        conditional_sigma * conditional_pdf
        + conditional_mean * conditional_cdf
    )
    delta_indicator = density_zero[:, None] * conditional_cdf
    delta_prime_relu = -density_zero[:, None] * (
        (mu / var)[:, None] * conditional_relu_mean
        + regression * conditional_cdf
    )
    one_sided = (
        (k3 / 6.0)[:, None] * delta_prime_relu
        + (k21 / 2.0) * delta_indicator
    )
    correction = one_sided + one_sided.T
    result = gaussian_post_covariance_h10(mu, cov) + correction
    return canonical_f64le(0.5 * (result + result.T))


def next_variance_action(post_covariance: np.ndarray, weight: np.ndarray) -> np.ndarray:
    cov = np.asarray(post_covariance, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    return canonical_f64le(np.sum(w * (cov @ w), axis=0))


@dataclass(frozen=True)
class BankError:
    difference: np.ndarray
    full_eight_proxy: np.ndarray
    conservative: np.ndarray


def bank_error(bank_a: np.ndarray, bank_b: np.ndarray) -> BankError:
    difference = np.asarray(bank_a, dtype=np.float64) - np.asarray(
        bank_b, dtype=np.float64
    )
    square = difference * difference
    return BankError(
        difference=canonical_f64le(difference),
        full_eight_proxy=canonical_f64le(square / 4.0),
        conservative=canonical_f64le(square / 2.0),
    )


def extrapolate_certificate(U14: float, U16: float, U18: float) -> dict[str, float | bool]:
    values = [float(U14), float(U16), float(U18)]
    if any((not math.isfinite(value) or value < 0.0) for value in values):
        raise ValueError("certificate inputs must be finite and nonnegative")

    def ratio(numerator: float, denominator: float) -> float:
        if denominator == 0.0:
            return 0.0 if numerator == 0.0 else math.inf
        return numerator / denominator

    rho = max(ratio(U16, U14), ratio(U18, U16))
    U22 = U18 * rho * rho
    return {
        "U14": U14,
        "U16": U16,
        "U18": U18,
        "rho": rho,
        "U22": U22,
        "rho_below_one": rho < 1.0,
        "U22_at_most_1e_9": U22 <= 1.0e-9,
        "pass": rho < 1.0 and U22 <= 1.0e-9,
    }


def relu_standard_normal_exact() -> dict[str, float]:
    """Exact scalar moments for max(N(0,1),0), used by identity control."""

    mean = 1.0 / SQRT_2PI
    raw2 = 0.5
    raw3 = math.sqrt(2.0 / math.pi)
    raw4 = 1.5
    variance = raw2 - mean * mean
    K3 = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
    central4 = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
    K4 = central4 - 3.0 * variance * variance
    return {
        "mean": mean,
        "variance": variance,
        "K3": K3,
        "K4": K4,
    }


def small_exact_self_test() -> dict[str, object]:
    """Dataset-free audit of masks, pooling, K21, split, and certificates."""

    if ARM_MASKS != tuple(f"{value:04b}" for value in range(16)):
        raise AssertionError("arm enumeration changed")
    # Pooling test with deliberately different counts: normalized averaging is wrong.
    first = RawTape.zeros(1, 2)
    second = RawTape.zeros(1, 2)
    weights = np.eye(2, dtype=np.float32)[None, :, :]
    accumulate_cpu_block(first, weights, np.array([[1.0, -2.0]], dtype=np.float32))
    accumulate_cpu_block(
        second,
        weights,
        np.array([[3.0, 4.0], [-1.0, 2.0], [0.5, -0.5]], dtype=np.float32),
    )
    merged = merge_raw_tapes([first, second])
    if merged.count != 4:
        raise AssertionError("raw pooling count failed")
    teacher = normalize_raw_tape(merged, weights)
    if not np.array_equal(teacher.pre_K21[0], np.zeros((2, 2))):
        raise AssertionError("Gaussian source K21 initialization failed")

    delta = np.array([[2.0, -0.75], [-0.75, 3.0]])
    host_cov = delta + np.eye(2)
    teacher_cov = np.eye(2)
    w = np.array([[1.0, 2.0], [-3.0, 0.5]])
    probe = np.eye(2)
    action = covariance_action_split(host_cov, teacher_cov, w, probe)
    literal_total = np.diag(w.T @ delta @ w)
    if not np.allclose(action.total, literal_total, rtol=1e-14, atol=1e-14):
        raise AssertionError("literal covariance action mismatch")

    a = np.array([1.0, 3.0])
    b = np.array([0.0, 5.0])
    error = bank_error(a, b)
    if not np.array_equal(error.full_eight_proxy, np.array([0.25, 1.0])):
        raise AssertionError("D^2/4 changed")
    if not np.array_equal(error.conservative, np.array([0.5, 2.0])):
        raise AssertionError("D^2/2 changed")
    certificate = extrapolate_certificate(16.0e-9, 4.0e-9, 1.0e-9)
    if certificate["rho"] != 0.25 or certificate["U22"] != 6.25e-11:
        raise AssertionError("extrapolation rule changed")
    exact = relu_standard_normal_exact()
    return {
        "status": "PASS_DATASET_FREE_SMALL_EXACT_CORE_TEST",
        "arm_masks": list(ARM_MASKS),
        "pooled_count": merged.count,
        "covariance_action_total": action.total.tolist(),
        "full_eight_proxy": error.full_eight_proxy.tolist(),
        "conservative": error.conservative.tolist(),
        "certificate": certificate,
        "relu_standard_normal": exact,
        "target_values_decoded": 0,
        "target_metrics_computed": 0,
        "fits": 0,
    }
