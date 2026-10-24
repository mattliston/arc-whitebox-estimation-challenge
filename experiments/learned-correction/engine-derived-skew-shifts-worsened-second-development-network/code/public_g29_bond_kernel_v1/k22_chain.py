#!/usr/bin/env python3
"""TAG g29-k22-chain: the engine's fourth-order plane chain (fourth_emission_1024 + the plane-supported push) as a
pure-numpy module the card can run in-card (T-89).  Copied from public_g19_attribution_v1/run_tensor_recursion_v4.py
(fourth_emission_1024, fourth_push_1024) and run_attribution.py (edg2_relu_mean); the 1024^4 pair term of the K22 push
is dropped (it changes the engine's K22 planes by 0.01%: tensor_planes_rank0035_dep_nobigpair vs _g29ref, corr 0.9999,
ratio 1.000).  Per layer: emission(m, v, k3, k4, C, K21, K22, K31) -> post (k4, K22, K31); push(post, W) -> next pre planes.
Cost per layer: emission = Mehler series (elementwise, 17 terms x 4 maps), push = 8 GEMM-class einsums."""
from math import erf, factorial
import numpy as np

_erf = np.vectorize(erf)


def edg2_relu_mean(m, v, k3, k4):
    v = np.maximum(v, 1e-30)
    s = np.sqrt(v)
    a = m / s
    pdf = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + _erf(a / np.sqrt(2)))
    base = s * pdf + m * cdf
    ma = -a
    he1 = ma
    he2 = ma * ma - 1.0
    he4 = ma ** 4 - 6.0 * ma * ma + 3.0
    c1_3 = s * he1 * pdf
    c1_4 = s * he2 * pdf
    c1_6 = s * he4 * pdf
    g3 = k3 / s ** 3
    g4 = k4 / s ** 4
    return base + (g3 / 6.0) * c1_3 + (g4 / 24.0) * c1_4 + (g3 * g3 / 72.0) * c1_6


def columns(m, v, qmax=30):
    v = np.maximum(v, 1e-30)
    sd = np.sqrt(v)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qmax + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c1 = [sd * pdf + m * cdf, sd * cdf]
    for q in range(2, qmax):
        c1.append(sd * He[q - 2] * pdf)
    raw1 = c1[0]
    raw2 = (m * m + v) * cdf + m * sd * pdf
    c2 = [raw2, 2 * sd * raw1, 2 * v * cdf]
    for q in range(3, qmax):
        c2.append(2 * v * He[q - 3] * pdf)
    return np.stack(c1), np.stack(c2), sd, pdf, cdf


def fourth_emission_1024(m, v, k3, k4, C, K21c, K22c, K31c, r3fix=False):
    """Post-ReLU fourth-order planes (k4diag, K22, K31), connected, via M22 (c2xc2) and M31 (c3xc1) Mehler maps."""
    c1, c2, sd, pdf0, cdf0 = columns(m, v)
    v_ = np.maximum(v, 1e-30)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    qn = c1.shape[0]
    raw1 = c1[0]
    raw2 = (m * m + v_) * cdf + m * sd * pdf
    raw3 = (m ** 3 + 3 * m * v_) * cdf + (m * m + 2 * v_) * sd * pdf
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qn + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c3 = [raw3, 3 * sd * raw2, 6 * v_ * raw1, 6 * v_ * sd * cdf]
    for q in range(4, qn):
        c3.append(6 * v_ * sd * He[q - 4] * pdf)
    c3 = np.stack(c3)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, b, tmax=16):
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / float(factorial(t)) * np.outer(colsA[a + t] / sd ** a, colsB[b + t] / sd ** b)
        return acc

    def dM(colsA, colsB):
        return ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21c / 2.0) * series(colsA, colsB, 2, 1)
                + (K21c.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4)
                + (K22c / 4.0) * series(colsA, colsB, 2, 2)
                + (K31c / 6.0) * series(colsA, colsB, 3, 1)
                + (K31c.T / 6.0) * series(colsA, colsB, 1, 3))

    M11 = series(c1, c1, 0, 0) + dM(c1, c1)
    M21 = series(c2, c1, 0, 0) + dM(c2, c1)
    M22 = series(c2, c2, 0, 0) + dM(c2, c2)
    M31 = series(c3, c1, 0, 0) + dM(c3, c1)
    r1 = edg2_relu_mean(m, v_, k3, k4)
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    he1, he3 = ma, ma ** 3 - 3 * ma
    r2 = (raw2 + (g3v / 6) * (2 * v_ * pdf) + (g4v / 24) * (2 * v_ * he1 * pdf)
          + (g3v * g3v / 72) * (2 * v_ * he3 * pdf))
    r3v = raw3
    if r3fix:
        r3v = (c3[0] + (k3 / 6.0) / sd ** 3 * c3[3] + (k4 / 24.0) / sd ** 4 * c3[4]
               + (k3 * k3 / 72.0) / sd ** 6 * c3[6])
    mu11 = M11 - np.outer(r1, r1)
    mu22 = (M22 - 2 * r1[None, :] * M21 - 2 * r1[:, None] * M21.T
            + np.outer(r1 ** 2, r2) + np.outer(r2, r1 ** 2)
            + 4 * np.outer(r1, r1) * M11 - 3 * np.outer(r1 ** 2, r1 ** 2))
    mu31 = (M31 - r1[None, :] * r3v[:, None] - 3 * r1[:, None] * M21
            + 3 * np.outer(r1, r1) * r2[:, None]
            + 3 * (r1 ** 2)[:, None] * M11 - 3 * np.outer(r1 ** 3, r1))
    vpost = np.maximum(r2 - r1 * r1, 1e-30)
    K22post = mu22 - np.outer(vpost, vpost) - 2 * mu11 ** 2
    K31post = mu31 - 3 * vpost[:, None] * mu11
    k4post = np.diag(K22post).copy()
    np.fill_diagonal(K22post, 0.0)
    np.fill_diagonal(K31post, 0.0)
    return k4post, K22post, K31post


def fourth_push_np(k4post, K22post, K31post, Wl):
    """Plane-supported push of the post fourth-order planes through Wl (in x out); the 1024^4 pair term is dropped."""
    W2 = Wl * Wl
    W3 = W2 * Wl
    k4n = (np.einsum("ia,ia,i->a", W2, W2, k4post, optimize=True)
           + 3.0 * np.einsum("ia,ja,ij->a", W2, W2, K22post, optimize=True)
           + 4.0 * np.einsum("ia,ja,ij->a", W3, Wl, K31post, optimize=True))
    K22n = (np.einsum("ia,ib,i->ab", W2, W2, k4post, optimize=True)
            + W2.T @ K22post @ W2
            + 2.0 * np.einsum("ia,ib,jb,ij->ab", W2, Wl, Wl, K31post, optimize=True)
            + 2.0 * np.einsum("ia,ja,ib,ij->ab", Wl, Wl, W2, K31post, optimize=True))
    K31n = (np.einsum("ia,ib,i->ab", W3, Wl, k4post, optimize=True)
            + W3.T @ K31post @ Wl
            + 3.0 * np.einsum("ia,ja,ib,ij->ab", W2, Wl, Wl, K31post, optimize=True)
            + 3.0 * np.einsum("ia,ja,jb,ij->ab", W2, Wl, Wl, K22post, optimize=True))
    np.fill_diagonal(K22n, 0.0)
    np.fill_diagonal(K31n, 0.0)
    return k4n, K22n, K31n
