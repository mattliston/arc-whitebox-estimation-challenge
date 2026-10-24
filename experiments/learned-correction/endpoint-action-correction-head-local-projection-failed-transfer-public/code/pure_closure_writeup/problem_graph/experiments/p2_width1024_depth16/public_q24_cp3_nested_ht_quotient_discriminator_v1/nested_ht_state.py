#!/usr/bin/env python3
"""Source-only algebra for the quotient nested pair-vector HT carrier.

This module is deliberately independent of the public estimator runtime.  It
contains no target access, fitting, board call, or submission path.  The dense
NumPy routines are reference spellings for the frozen source discriminator and
for a later FlopScope implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NestedHTState:
    """One post-reset nested-HT state.

    `u` is `(n,r)`, `core` is `(r,r,k)`, and `v` is always `u[:,:k]`.
    The core is symmetric in its first two indices.
    """

    u: np.ndarray
    core: np.ndarray

    @property
    def r(self) -> int:
        return int(self.u.shape[1])

    @property
    def k(self) -> int:
        return int(self.core.shape[2])

    @property
    def v(self) -> np.ndarray:
        return self.u[:, : self.k]

    def validate(self, *, atol: float = 2.0e-5) -> None:
        n, r = self.u.shape
        if self.core.shape[:2] != (r, r):
            raise AssertionError((self.u.shape, self.core.shape))
        if not (0 < self.k <= r <= n):
            raise AssertionError((n, r, self.k))
        if not np.all(np.isfinite(self.u)) or not np.all(np.isfinite(self.core)):
            raise AssertionError("nonfinite nested-HT state")
        gram = self.u.astype(np.float64).T @ self.u.astype(np.float64)
        gram_error = float(np.max(np.abs(gram - np.eye(r))))
        symmetry_error = float(
            np.max(
                np.abs(
                    self.core.astype(np.float64)
                    - self.core.swapaxes(0, 1).astype(np.float64)
                )
            )
        )
        if gram_error > atol or symmetry_error > atol:
            raise AssertionError((gram_error, symmetry_error))


def canonicalize_columns(value: np.ndarray) -> np.ndarray:
    """Fix column signs without changing their span."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def row_pair(u: np.ndarray) -> np.ndarray:
    """Return rows `u_i outer u_i`, flattened in `(a,b)` order."""

    value = np.asarray(u)
    return np.einsum("ia,ib->iab", value, value, optimize=True).reshape(
        value.shape[0], value.shape[1] * value.shape[1]
    )


def row_mixed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return rows `u_i outer v_i`, flattened in `(a,s)` order."""

    first = np.asarray(u)
    second = np.asarray(v)
    return np.einsum("ia,is->ias", first, second, optimize=True).reshape(
        first.shape[0], first.shape[1] * second.shape[1]
    )


def base_core_from_cp(
    cp_lambda: np.ndarray,
    projected_p: np.ndarray,
    projected_q: np.ndarray,
    *,
    batch: int = 256,
) -> np.ndarray:
    """Build `sum_j lambda_j p_j outer p_j outer q_j`.

    `projected_p` is `(r,R)` and `projected_q` is `(k,R)`.  The returned
    `(r,r,k)` core is the unsymmetrized base core; the carrier's outer `sym`
    supplies the three physical tensor placements.
    """

    lam = np.asarray(cp_lambda, dtype=np.float64)
    p = np.asarray(projected_p, dtype=np.float64)
    q = np.asarray(projected_q, dtype=np.float64)
    r, rank = p.shape
    k, q_rank = q.shape
    if lam.shape != (rank,) or q_rank != rank:
        raise AssertionError((lam.shape, p.shape, q.shape))
    flat = np.zeros((r * r, k), dtype=np.float64)
    for start in range(0, rank, batch):
        stop = min(start + batch, rank)
        p_part = p[:, start:stop]
        pair = np.einsum("aj,bj->abj", p_part, p_part, optimize=True).reshape(
            r * r, stop - start
        )
        weighted_q = (q[:, start:stop] * lam[None, start:stop]).T
        flat += pair @ weighted_q
    core = flat.reshape(r, r, k)
    return 0.5 * (core + core.swapaxes(0, 1))


def lifted_base_core(
    repeated_plane: np.ndarray,
    u: np.ndarray,
    k: int,
    *,
    row_pair_u: np.ndarray | None = None,
) -> np.ndarray:
    """Project the exact coordinate lift `L(H)` into the base core.

    The diagonal of the CP coefficient plane is divided by three because
    `sym(e_i,e_i,e_i)=3 e_i^3` under the closure convention.
    """

    basis = np.asarray(u, dtype=np.float64)
    n, r = basis.shape
    if not (0 < k <= r):
        raise AssertionError((r, k))
    h = np.array(repeated_plane, dtype=np.float64, copy=True, order="C")
    if h.shape != (n, n):
        raise AssertionError((h.shape, basis.shape))
    diagonal = np.diag_indices(n)
    h[diagonal] /= 3.0
    v = basis[:, :k]
    pair = row_pair(basis) if row_pair_u is None else np.asarray(row_pair_u)
    flat = pair.astype(np.float64).T @ (h @ v)
    core = flat.reshape(r, r, k)
    return 0.5 * (core + core.swapaxes(0, 1))


def project_cp_quotient_core(
    cp_lambda: np.ndarray,
    cp_p: np.ndarray,
    cp_q: np.ndarray,
    repeated_plane: np.ndarray,
    u: np.ndarray,
    k: int,
    *,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quotient, raw, and lifted base cores in a fixed nested basis."""

    basis = np.asarray(u, dtype=np.float64)
    v = basis[:, :k]
    p = basis.T @ np.asarray(cp_p, dtype=np.float64)
    q = v.T @ np.asarray(cp_q, dtype=np.float64)
    raw = base_core_from_cp(cp_lambda, p, q, batch=batch)
    pair = row_pair(basis)
    h = np.array(repeated_plane, dtype=np.float64, copy=True, order="C")
    diagonal = np.diag_indices(h.shape[0])
    h[diagonal] /= 3.0
    lift = (pair.T @ (h @ v)).reshape(basis.shape[1], basis.shape[1], k)
    lift = 0.5 * (lift + lift.swapaxes(0, 1))
    quotient = raw - lift
    return quotient, raw, lift


def transport_core(core: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply `A` on the two pair modes and `B` on the action mode."""

    value = np.asarray(core, dtype=np.float64)
    pair_map = np.asarray(a, dtype=np.float64)
    action_map = np.asarray(b, dtype=np.float64)
    first = np.einsum("pa,abs->pbs", pair_map, value, optimize=True)
    second = np.einsum("qb,pbs->pqs", pair_map, first, optimize=True)
    result = np.einsum("ts,pqs->pqt", action_map, second, optimize=True)
    return 0.5 * (result + result.swapaxes(0, 1))


def read_k21(u: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Read the complete directed `K21[i,j]=T[i,i,j]` plane."""

    basis = np.asarray(u)
    value = np.asarray(core)
    n, r = basis.shape
    if value.shape[:2] != (r, r):
        raise AssertionError((basis.shape, value.shape))
    k = value.shape[2]
    v = basis[:, :k]
    d = row_pair(basis) @ value.reshape(r * r, k)
    e = row_mixed(basis, v) @ value.transpose(0, 2, 1).reshape(r * k, r)
    return np.asarray(d @ v.T + 2.0 * (e @ basis.T), dtype=np.float64)


def read_k3(u: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Read only the tensor diagonal."""

    basis = np.asarray(u)
    value = np.asarray(core)
    r = basis.shape[1]
    k = value.shape[2]
    d = row_pair(basis) @ value.reshape(r * r, k)
    return np.asarray(3.0 * np.sum(d * basis[:, :k], axis=1), dtype=np.float64)


def reset_star(k21_target: np.ndarray, u: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Return the float32 coordinate star that restores current `K21`."""

    return np.asarray(
        np.asarray(k21_target, dtype=np.float64) - read_k21(u, core),
        dtype=np.float32,
    )


def explicit_tensor(u: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Tiny-only explicit tensor constructor used by the algebra self-test."""

    basis = np.asarray(u, dtype=np.float64)
    value = np.asarray(core, dtype=np.float64)
    v = basis[:, : value.shape[2]]
    first = np.einsum("ia,jb,ks,abs->ijk", basis, basis, v, value, optimize=True)
    return first + first.transpose(0, 2, 1) + first.transpose(2, 1, 0)


def explicit_lift(repeated_plane: np.ndarray) -> np.ndarray:
    """Tiny-only exact coordinate lift `L(H)`."""

    h = np.asarray(repeated_plane, dtype=np.float64)
    n = h.shape[0]
    result = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, i, j] = h[i, j]
            result[i, j, i] = h[i, j]
            result[j, i, i] = h[i, j]
    return result


def self_test() -> None:
    rng = np.random.default_rng(2026082612832)
    n = 7
    rank = 11
    u = np.eye(n)
    p = rng.normal(size=(n, rank))
    q = rng.normal(size=(n, rank))
    lam = rng.normal(size=rank)
    base = base_core_from_cp(lam, p, q)
    tensor = explicit_tensor(u, base)
    direct = np.einsum("j,ij,kj,lj->ikl", lam, p, p, q, optimize=True)
    direct = direct + direct.transpose(0, 2, 1) + direct.transpose(2, 1, 0)
    if not np.allclose(tensor, direct, rtol=1.0e-11, atol=1.0e-11):
        raise AssertionError("base CP reconstruction")
    h = np.einsum("iij->ij", tensor)
    if not np.allclose(read_k21(u, base), h, rtol=1.0e-11, atol=1.0e-11):
        raise AssertionError("K21 read")
    lift_core = lifted_base_core(h, u, n)
    lift = explicit_tensor(u, lift_core)
    if not np.allclose(lift, explicit_lift(h), rtol=1.0e-11, atol=1.0e-11):
        raise AssertionError("coordinate lift")
    quotient = tensor - lift
    if float(np.max(np.abs(np.einsum("iij->ij", quotient)))) > 1.0e-10:
        raise AssertionError("quotient repeated plane")
    quotient_core, raw_core, projected_lift = project_cp_quotient_core(
        lam, p, q, h, u, n
    )
    if not np.allclose(raw_core, base, rtol=1.0e-11, atol=1.0e-11):
        raise AssertionError("raw core projection")
    if not np.allclose(projected_lift, lift_core, rtol=1.0e-11, atol=1.0e-11):
        raise AssertionError("lift core projection")
    if not np.allclose(
        explicit_tensor(u, quotient_core), quotient, rtol=1.0e-11, atol=1.0e-11
    ):
        raise AssertionError("quotient core projection")

    # Exercise the actual nested case r>k and independently verify that both
    # public projection spellings use the requested action width, not r.
    nested_r = 5
    nested_k = 2
    nested_u, _ = np.linalg.qr(rng.normal(size=(n, nested_r)), mode="reduced")
    nested_lift = lifted_base_core(h, nested_u, nested_k)
    if nested_lift.shape != (nested_r, nested_r, nested_k):
        raise AssertionError(nested_lift.shape)
    _, _, projected_nested_lift = project_cp_quotient_core(
        lam, p, q, h, nested_u, nested_k
    )
    if not np.allclose(
        nested_lift, projected_nested_lift, rtol=1.0e-11, atol=1.0e-11
    ):
        raise AssertionError("r>k lift projection")
    print("nested_ht_state self-test: PASS")


if __name__ == "__main__":
    self_test()
