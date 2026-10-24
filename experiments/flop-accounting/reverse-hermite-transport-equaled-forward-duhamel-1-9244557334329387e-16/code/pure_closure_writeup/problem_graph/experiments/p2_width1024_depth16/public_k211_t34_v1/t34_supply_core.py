"""Exact order-3 + order-4 kappa211 supply into the next marginal kappa4.

TAG=k211-t34-core  Extends public_k211_supply_v1/supply_core.py (the T2
truncation) with every order-3 (7) and order-4 (12) term of
public_trivariate_derivation_v1/FORMULA.json:

  kappa211(i;j,k) term = r_ij^p r_ik^q r_jk^t / (p! q! t!)
                         * (B[p+q] - 2 A[p] A[q] [t=0]) * A[p+t]_j * A[q+t]_k
  S(a) = 3 * sum over DISTINCT i,j,k of W_ia^2 W_ja W_ka kappa211(i;j,k)

with A[q] = c1[q] (A[0]=0) and B[q] = c2[q] - 2 m c1[q], m = c1[0].

Contraction classes (R = rho with zero diagonal, so every linked pair's
coincidence is auto-excluded; one manual subtraction per zero exponent):
  class A (t=0):  star at i.   sum_i W2 Vi [ (R^p@(AjW)) * (R^q@(AkW))
                                             - (R^p*R^q)@(AjAk W^2) ]
  class B (q=0):  chain i-j-k. sum_i W2 B[p] [ R^p@((AjW)*(R^t@(AkW)))
                                             - (AkW)*((R^p*R^t)@(AjW)) ]
  class C (all>0): triangle.   per output column a:
                   T[:,a] = ((R^p diag(Aj*W[:,a]) R^t) * R^q) @ (Ak*W[:,a])
Mirror pairs (p,q,t)<->(q,p,t) are collapsed (computed once, doubled);
(1,1,1), (2,2,0), (1,1,2) are self-mirrors.  Brute-verified term-by-term
against the raw 22-entry FORMULA list at small n.
"""
import numpy as np

# (p, q, t, coefficient) straight from FORMULA.json, orders 2..4, all 22.
FORMULA_TERMS = (
    (0, 1, 1, 1.0), (0, 1, 2, 0.5), (0, 1, 3, 1.0 / 6.0),
    (0, 2, 1, 0.5), (0, 2, 2, 0.25), (0, 3, 1, 1.0 / 6.0),
    (1, 0, 1, 1.0), (1, 0, 2, 0.5), (1, 0, 3, 1.0 / 6.0),
    (1, 1, 0, 1.0), (1, 1, 1, 1.0), (1, 1, 2, 0.5),
    (1, 2, 0, 0.5), (1, 2, 1, 0.5), (1, 3, 0, 1.0 / 6.0),
    (2, 0, 1, 0.5), (2, 0, 2, 0.25), (2, 1, 0, 0.5),
    (2, 1, 1, 0.5), (2, 2, 0, 0.25), (3, 0, 1, 1.0 / 6.0),
    (3, 1, 0, 1.0 / 6.0),
)


def _functionals(c1, c2):
    m = c1[0]
    A = [np.zeros_like(m), c1[1], c1[2], c1[3], c1[4]]
    B = [None] + [c2[qq] - 2.0 * m * c1[qq] for qq in range(1, 5)]
    return A, B


def _r_powers(rho):
    R = rho - np.eye(rho.shape[0])
    return {1: R, 2: R * R, 3: R * R * R}


def _class_a(W, W2, Rp, A, B, p, q, coef, mult):
    Vi = B[p + q] - 2.0 * A[p] * A[q]
    M1 = Rp[p] @ (A[p][:, None] * W)
    M2 = Rp[q] @ (A[q][:, None] * W)
    corr = (Rp[p] * Rp[q]) @ ((A[p] * A[q])[:, None] * W2)
    return (mult * coef) * ((W2 * Vi[:, None]) * (M1 * M2 - corr)).sum(axis=0)


def _class_b(W, W2, Rp, A, B, p, t, coef, mult):
    Aj, Ak = A[p + t], A[t]
    inner = Rp[p] @ ((Aj[:, None] * W) * (Rp[t] @ (Ak[:, None] * W)))
    corr = (Ak[:, None] * W) * ((Rp[p] * Rp[t]) @ (Aj[:, None] * W))
    return (mult * coef) * ((W2 * B[p][:, None]) * (inner - corr)).sum(axis=0)


def _triangles(W, W2, Rp, A, B):
    """(1,1,1) + 2*(1,2,1) + (1,1,2)/2, sharing the per-column GEMM."""
    n, na = W.shape
    R1, R2 = Rp[1], Rp[2]
    U2 = A[2][:, None] * W        # A[p+t] for (1,1,1) and (1,2,1)
    U3 = A[3][:, None] * W        # for (1,1,2)
    S = np.zeros(na)
    for a in range(na):
        G = (R1 * U2[:, a][None, :]) @ R1
        t111 = (G * R1) @ U2[:, a]
        t121 = (G * R2) @ U3[:, a]
        G2 = (R1 * U3[:, a][None, :]) @ R2
        t112 = (G2 * R1) @ U3[:, a]
        S[a] = (W2[:, a] * (
            B[2] * (1.0 * t111 + 0.5 * t112) + B[3] * (2.0 * 0.5 * t121)
        )).sum()
    return S


# Canonical mirror-collapsed groups: tag -> (class, p, q_or_t, coef, mult).
# Star groups carry (p, q); chain groups carry (p, t) and cover the (q=0)
# canonical plus its (p=0) mirror via mult=2.  "TRI" is the fused triangle
# routine (1,1,1) + 2*(1,2,1) + (1,1,2)/2.
CANONICAL_GROUPS = {
    "A110": ("A", 1, 1, 1.0, 1.0),          # order 2 (1,1,0)
    "B101": ("B", 1, 1, 1.0, 2.0),          # order 2 (1,0,1)+(0,1,1)
    "A120": ("A", 1, 2, 0.5, 2.0),          # (1,2,0)+(2,1,0)
    "B102": ("B", 1, 2, 0.5, 2.0),          # (1,0,2)+(0,1,2)
    "B201": ("B", 2, 1, 0.5, 2.0),          # (2,0,1)+(0,2,1)
    "A130": ("A", 1, 3, 1.0 / 6.0, 2.0),    # (1,3,0)+(3,1,0)
    "A220": ("A", 2, 2, 0.25, 1.0),         # (2,2,0)
    "B103": ("B", 1, 3, 1.0 / 6.0, 2.0),    # (1,0,3)+(0,1,3)
    "B202": ("B", 2, 2, 0.25, 2.0),         # (2,0,2)+(0,2,2)
    "B301": ("B", 3, 1, 1.0 / 6.0, 2.0),    # (3,0,1)+(0,3,1)
    "TRI": ("T",),                          # (1,1,1),(1,2,1)+(2,1,1),(1,1,2)
}
T2_GROUPS = ("A110", "B101")
T34_CHAINSTAR_GROUPS = ("A120", "B102", "B201", "A130", "A220", "B103",
                        "B202", "B301")
T34_GROUPS = T34_CHAINSTAR_GROUPS + ("TRI",)


def kappa211_t34_supply(W, rho, c1, c2, include_t2=False, groups=None):
    """kappa211 supply S(a) over the requested canonical groups.

    Default (groups=None): the order-3+4 terms; include_t2 adds order 2.
    groups: explicit iterable of CANONICAL_GROUPS tags, overriding both.
    W (n, na) layer weights; rho (n, n) correlation; c1, c2 (5, n): the
    c-table channels q=0..4 (c1[q] = E[x He_q], c2[q] = E[x^2 He_q]).
    """
    if groups is None:
        groups = (T2_GROUPS + T34_GROUPS) if include_t2 else T34_GROUPS
    A, B = _functionals(np.asarray(c1, dtype=np.float64),
                        np.asarray(c2, dtype=np.float64))
    Rp = _r_powers(np.asarray(rho, dtype=np.float64))
    W = np.asarray(W, dtype=np.float64)
    W2 = W * W
    S = np.zeros(W.shape[1])
    for tag in groups:
        spec = CANONICAL_GROUPS[tag]
        if spec[0] == "A":
            S += _class_a(W, W2, Rp, A, B, spec[1], spec[2], spec[3], spec[4])
        elif spec[0] == "B":
            S += _class_b(W, W2, Rp, A, B, spec[1], spec[2], spec[3], spec[4])
        else:
            S += _triangles(W, W2, Rp, A, B)
    return 3.0 * S


def kappa211_supply_brute(W, rho, c1, c2, orders=(3, 4)):
    """Triple loop over the raw FORMULA term list; the verification oracle."""
    n, na = W.shape
    A, B = _functionals(np.asarray(c1, dtype=np.float64),
                        np.asarray(c2, dtype=np.float64))
    terms = [term for term in FORMULA_TERMS
             if (term[0] + term[1] + term[2]) in orders]
    S = np.zeros(na)
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
                    value += (coef
                              * rho[i, j] ** p * rho[i, k] ** q
                              * rho[j, k] ** t
                              * i_func * A[p + t][j] * A[q + t][k])
                S += 3.0 * (W[i] ** 2) * W[j] * W[k] * value
    return S
