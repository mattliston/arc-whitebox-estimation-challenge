"""Never-carried push routes: the K22 F-form (ij|ij) and kappa211 -> planes.

TAG=missing-routes-core  Two exact masses the deployed supply discards.

ARM 1 - F-form.  Pushing the K22 residual K (symmetric, zero diagonal) as a
symmetric 4-tensor (entries K_ij/2 at each of the 6 arrangements of two i's
and two j's; the /2 is FIXED by matching the deployed k4 diagonal
3*(W2^T K W2)_aa) yields exactly:
  k4 diag  3 (W2^T K W2)_aa                    - fully carried
  k31(a,b) 3 sum W_ia^2 K_ij W_ja W_jb        - fully carried
  k22(a,b) (W2^T K W2)_ab + 2 F(a,b),
           F(a,b) = sum_ij (W_ia W_ib) K_ij (W_ja W_jb)
so the ONLY missing content is 2F on the k22 plane (diagonal zeroed: the F
diagonal duplicates the carried route).  Brute-verified vs the tensor push.

ARM 2 - kappa211 planes.  The kappa211(i;j,k) source pattern (i,i,j,k)
(tensor entries kappa211/4 at each of the 12 arrangements; /4 FIXED by the
banked diagonal S(a) = 3 sum W2 W W kappa211) also carries, over distinct
i,j,k (j<->k symmetric):
  k31(a,b) = 1.5 * (R1 + R2)   R1: i^2@a, j@a, k@b   R2: i@(a,b), j@a, k@a
  k22(a,b) = 0.5 * (P1 + P1^T) + 2 * P2
                               P1: i^2@a, j@b, k@b   P2: i@(a,b), j@a, k@b
Every route is evaluated per FORMULA term through the same class machinery
as t34_supply_core (star t=0 / chain q=0 / triangle; zero-diagonal R powers
auto-exclude linked coincidences, one manual subtraction per zero exponent).
The i-mixed-column routes that do not factor into plain GEMMs (chain P2 and
the star P2 coincidence) use the general bilinear-pair pass
  G[a,b] = sum_ij K_ij L1_ia L2_ib R1_ja R2_jb
(one n^2 GEMM per output column - the F-form's own cost).  All coefficients
and routes are brute-verified against the explicit tensor push AND the
term-level triple loop at small n by verify_missing_routes.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public_k211_t34_v1"))
from t34_supply_core import (CANONICAL_GROUPS, T2_GROUPS, T34_GROUPS,
                             _functionals, _r_powers)

ALL_GROUPS = T2_GROUPS + T34_GROUPS
K31_ROUTE_COEF = 1.5
K22_P1_COEF = 0.5
K22_P2_COEF = 2.0


def group_terms(tag):
    """The raw (p, q, t, coef) FORMULA terms a canonical group collapses."""
    spec = CANONICAL_GROUPS[tag]
    if spec[0] == "A":
        _c, p, q, coef, _m = spec
        return [(p, q, 0, coef)] if p == q else [(p, q, 0, coef),
                                                 (q, p, 0, coef)]
    if spec[0] == "B":
        _c, p, t, coef, _m = spec
        return [(p, 0, t, coef), (0, p, t, coef)]
    return [(1, 1, 1, 1.0), (1, 1, 2, 0.5), (1, 2, 1, 0.5), (2, 1, 1, 0.5)]


def pair_pass(K, L1, L2, Ra, Rb, block=8):
    """G[a,b] = sum_ij K_ij L1_ia L2_ib Ra_ja Rb_jb (the F-form family).

    Blocked over b so the n^2-per-column GEMMs fuse into (n, na*block)
    GEMMs; exact, order-independent."""
    n, na = L1.shape
    G = np.empty((na, na))
    for start in range(0, na, block):
        stop = min(start + block, na)
        width = stop - start
        right = (Ra[:, :, None] * Rb[:, None, start:stop]).reshape(n, na * width)
        KR = K @ right
        left = (L1[:, :, None] * L2[:, None, start:stop]).reshape(n, na * width)
        G[:, start:stop] = (left * KR).sum(axis=0).reshape(na, width)
    return G


def f_form_k22_missing(W, K):
    """ARM 1: the never-carried k22 content 2F, diagonal zeroed."""
    add = 2.0 * pair_pass(K, W, W, W, W)
    np.fill_diagonal(add, 0.0)
    return add


def _routes_star(W, W2, Rp, A, B, p, q, coef, mult):
    Vi = B[p + q] - 2.0 * A[p] * A[q]
    X = A[p][:, None] * W
    Y = A[q][:, None] * W
    M1 = Rp[p] @ X
    M2 = Rp[q] @ Y
    D = Rp[p] * Rp[q]
    AAW = (A[p] * A[q])[:, None] * W
    AAW2 = AAW * W
    WB2 = W2 * Vi[:, None]
    WBm = W * Vi[:, None]
    s = mult * coef
    R1 = (WB2 * M1).T @ M2 - ((D.T @ WB2) * AAW).T @ W
    R2 = (WBm * M1 * M2).T @ W - (WBm * (D @ AAW2)).T @ W
    P1 = WB2.T @ (M1 * M2) - WB2.T @ (D @ AAW2)
    P2 = (WBm * M1).T @ (W * M2) - pair_pass(D, WBm, W, AAW, W)
    return s * R1, s * R2, s * P1, s * P2


def _routes_chain(W, W2, Rp, A, B, p, t, coef, mult):
    """Canonical (p,0,t) PLUS its (0,p,t) mirror, which planes cannot
    collapse by doubling: the mirror puts the tail leg on column a and the
    middle leg on column b, so R1_mirror is its own GEMM form, R2 and P1
    are leg-symmetric (mirror = canonical), and P2_mirror = P2^T."""
    assert mult == 2.0, "chains always carry an explicit mirror"
    X = A[p + t][:, None] * W            # the middle leg
    AkW = A[t][:, None] * W              # the tail leg
    N = Rp[t] @ AkW
    D = Rp[p] * Rp[t]
    DX = D @ X
    WB2 = W2 * B[p][:, None]
    WBm = W * B[p][:, None]
    I = Rp[p] @ (X * N)
    RpT_WB2 = Rp[p].T @ WB2
    R1 = (RpT_WB2 * X).T @ N - (WB2 * DX).T @ AkW
    R1m = (RpT_WB2 * N).T @ X - (WB2 * AkW).T @ DX
    R2 = (WBm * I).T @ W - (WBm * AkW * DX).T @ W
    P1 = WB2.T @ I - WB2.T @ (AkW * DX)
    P2 = pair_pass(Rp[p], WBm, W, X, N) - (WBm * DX).T @ (W * AkW)
    return (coef * (R1 + R1m), coef * (2.0 * R2), coef * (2.0 * P1),
            coef * (P2 + P2.T))


def _routes_triangles(W, W2, Rp, A, B):
    """(1,1,1) + (1,2,1)+(2,1,1) + (1,1,2), plane routes, fused per column.

    Per column a and triangle spec, T_i(a, cv) = [((Rp diag(u[:,a]) Rt)
    * Rq) @ v[:, cv]]_i; routes read T slices:
      R1[a,:] = (W2[:,a] Bi)^T t_slice        R2[a,:] = ((W[:,a] Bi t_aa) W)
      P1 = (W2 Bi)^T diagT                    P2[a,:] = (W[:,a] Bi)^T (W t)
    (1,1,1) and (1,2,1) share the G_a chain; mirror (2,1,1) enters via the
    j<->k relabeling: for triangles the mirrored term equals the canonical
    one with u/v swapped, which the ordered (j,k) sum already covers when
    u and v are set per term, so (1,2,1) carries its own u=v-swapped pass
    doubled by mult=2 exactly like the diagonal implementation.
    """
    n, na = W.shape
    # (Rp(i,j), Rt(j,k), Rq(i,k), u=A[p+t]W, v=A[q+t]W, coef, Bi).  The
    # (1,2,1)/(2,1,1) mirror pair CANNOT be collapsed for planes (the
    # routes tell the j-leg from the k-leg) - each carries its own spec.
    specs = (
        (Rp[1], Rp[1], Rp[1], A[2][:, None] * W, A[2][:, None] * W, 1.0, B[2]),   # (1,1,1)
        (Rp[1], Rp[1], Rp[2], A[2][:, None] * W, A[3][:, None] * W, 0.5, B[3]),   # (1,2,1)
        (Rp[2], Rp[1], Rp[1], A[3][:, None] * W, A[2][:, None] * W, 0.5, B[3]),   # (2,1,1)
        (Rp[1], Rp[2], Rp[1], A[3][:, None] * W, A[3][:, None] * W, 0.5, B[2]),   # (1,1,2)
    )
    R1 = np.zeros((na, na))
    R2 = np.zeros((na, na))
    P1 = np.zeros((na, na))
    P2 = np.zeros((na, na))
    for Rpp, Rtt, Rqq, U, V, s, Bi in specs:
        diagT = np.empty((n, na))
        WB2 = W2 * Bi[:, None]
        WBm = W * Bi[:, None]
        r1 = np.empty((na, na))
        r2 = np.empty((na, na))
        p2 = np.empty((na, na))
        for a in range(na):
            G = (Rpp * U[:, a][None, :]) @ Rtt
            t_slice = (G * Rqq) @ V
            diagT[:, a] = t_slice[:, a]
            r1[a] = WB2[:, a] @ t_slice
            r2[a] = (WBm[:, a] * t_slice[:, a]) @ W
            p2[a] = WBm[:, a] @ (W * t_slice)
        R1 += s * r1
        R2 += s * r2
        P1 += s * (WB2.T @ diagT)
        P2 += s * p2
    return R1, R2, P1, P2


def kappa211_plane_supplies(W, rho, c1, c2, groups=None,
                            include_triangles=True):
    """ARM 2: (k31_add, k22_add) from kappa211 orders <=4, diagonals zeroed."""
    if groups is None:
        groups = ALL_GROUPS
    A, B = _functionals(np.asarray(c1, dtype=np.float64),
                        np.asarray(c2, dtype=np.float64))
    Rp = _r_powers(np.asarray(rho, dtype=np.float64))
    W = np.asarray(W, dtype=np.float64)
    W2 = W * W
    na = W.shape[1]
    R1 = np.zeros((na, na))
    R2 = np.zeros((na, na))
    P1 = np.zeros((na, na))
    P2 = np.zeros((na, na))
    for tag in groups:
        spec = CANONICAL_GROUPS[tag]
        if spec[0] == "A":
            # Asymmetric stars carry an explicit (q,p) mirror pass; the
            # generic star routine covers both orientations.
            parts_list = [_routes_star(W, W2, Rp, A, B, spec[1], spec[2],
                                       spec[3], 1.0)]
            if spec[1] != spec[2]:
                parts_list.append(_routes_star(W, W2, Rp, A, B, spec[2],
                                               spec[1], spec[3], 1.0))
        elif spec[0] == "B":
            parts_list = [_routes_chain(W, W2, Rp, A, B, spec[1], spec[2],
                                        spec[3], spec[4])]
        elif include_triangles:
            parts_list = [_routes_triangles(W, W2, Rp, A, B)]
        else:
            continue
        for parts in parts_list:
            R1 += parts[0]
            R2 += parts[1]
            P1 += parts[2]
            P2 += parts[3]
    k31 = K31_ROUTE_COEF * (R1 + R2)
    k22 = K22_P1_COEF * (P1 + P1.T) + K22_P2_COEF * P2
    np.fill_diagonal(k31, 0.0)
    np.fill_diagonal(k22, 0.0)
    return k31, k22


# --------------------------------------------------------------------------
# Brute oracles (small n): explicit symmetric-tensor pushes.
def push_tensor(T4, W):
    return np.einsum("pqrs,pa,qb,rc,sd->abcd", T4, W, W, W, W)


def tensor_from_K(K):
    """Entries K_ij/2 at the 6 two-i-two-j arrangements, ordered (i,j)."""
    n = K.shape[0]
    T = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            if i == j or K[i, j] == 0.0:
                continue
            half = 0.5 * K[i, j]
            for slots in ((i, i, j, j), (i, j, i, j), (i, j, j, i),
                          (j, i, i, j), (j, i, j, i), (j, j, i, i)):
                T[slots] += half
    return T


def tensor_from_k211(values, n):
    """values[(i,j,k)] -> entries value/4 at the 12 iijk arrangements."""
    T = np.zeros((n, n, n, n))
    import itertools
    for (i, j, k), value in values.items():
        quarter = 0.25 * value
        for slots in set(itertools.permutations((i, i, j, k))):
            T[slots] += quarter
    return T


def k211_values(rho, c1, c2, terms, n):
    """Triple-loop kappa211(i;j,k) over distinct triples for given terms."""
    A, B = _functionals(np.asarray(c1, dtype=np.float64),
                        np.asarray(c2, dtype=np.float64))
    out = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                value = 0.0
                for p, q, t, coef in terms:
                    i_func = B[p + q][i]
                    if t == 0:
                        i_func = i_func - 2.0 * A[p][i] * A[q][i]
                    value += (coef * rho[i, j] ** p * rho[i, k] ** q
                              * rho[j, k] ** t
                              * i_func * A[p + t][j] * A[q + t][k])
                if value != 0.0:
                    out[(i, j, k)] = value
    return out
