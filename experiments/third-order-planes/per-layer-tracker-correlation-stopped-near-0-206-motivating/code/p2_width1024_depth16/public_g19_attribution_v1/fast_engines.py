#!/usr/bin/env python3
"""Batched engine kernels (full-machine policy: use it, don't OOM).

- batch_scalars: the certified CF-only per-unit scalars
  (E[phi], Var(phi), P(z>0), p(0), p'(0), p''(0)) vectorized across
  units with t-block accumulation. Same t-grid as the certified
  probe (4k head + 24k tail to t=300), trapezoid weights.
- mps_materialize: batched per-unit M_c = sum_pairs D1^T diag(w) D2
  on MPS (f32), returned on CPU as f64.
Memory budget: unit-chunks x t-blocks sized for ~4GB transients on a
48GB box.
"""
import numpy as np

_TS = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                      np.linspace(1.0, 300.0, 24000)])
_WTS = np.gradient(_TS)
# short grid: CF is dead beyond t~20 for O(1)-variance units; the
# only slow integrand is (1-Re cf)/t^2 -> analytic tail 1/T1.
_T1 = 25.0
_TS_S = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                        np.linspace(1.0, _T1, 6000)])
_WTS_S = np.gradient(_TS_S)


def compress_modes(lam, at, K=128):
    """Top-K modes exact + analytic tail moments (certified below).
    Returns lam_top (K), at_top (K), tail = (S1, S2, S3, A0, A1, A2)
    with Sk = sum lam^k, A_j = sum at^2 lam^j over the tail."""
    imp = at ** 2 + lam ** 2
    idx = np.argsort(imp)[::-1]
    top, rest = idx[:K], idx[K:]
    lt, at_ = lam[rest], at[rest]
    tail = (lt.sum(), (lt ** 2).sum(), (lt ** 3).sum(),
            (at_ ** 2).sum(), (at_ ** 2 * lt).sum(),
            (at_ ** 2 * lt ** 2).sum())
    return lam[top], at[top], tail


def batch_scalars_trunc(m_vec, lam_top, at_top, tails, tblock=7000):
    """batch_scalars on top-K modes + analytic tail terms.
    log CF tail: sum log(1-it l) ~= -it S1 + t^2 S2/2 + i t^3 S3/3
    a-part tail: sum a^2/(1-it l) ~= A0 + it A1 - t^2 A2
    d1/d2 tails follow by differentiating in it."""
    B, K = lam_top.shape
    S1, S2, S3, A0, A1, A2 = [np.array([tl[j] for tl in tails])
                              for j in range(6)]
    Ez2 = (m_vec ** 2 + (at_top ** 2).sum(1) + A0
           + 0.5 * ((lam_top ** 2).sum(1) + S2))
    accs = {k: np.zeros(B) for k in
            ("eabs", "imcf2t", "imcft", "recf", "tim", "t2re")}
    for s in range(0, len(_TS_S), tblock):
        ts = _TS_S[s:s + tblock]
        w = _WTS_S[s:s + tblock]
        den = 1.0 - 1j * ts[None, :, None] * lam_top[:, None, :]
        a2 = (at_top ** 2)[:, None, :]
        sum_a_den = (a2 / den).sum(2) + (A0[:, None]
                                         + 1j * ts[None, :] * A1[:, None]
                                         - ts[None, :] ** 2 * A2[:, None])
        sum_log = (np.log(den).sum(2)
                   + (-1j * ts[None, :] * S1[:, None]
                      + ts[None, :] ** 2 * S2[:, None] / 2
                      + 1j * ts[None, :] ** 3 * S3[:, None] / 3))
        lamS = lam_top.sum(1) + S1
        lcf = (1j * np.outer(m_vec, ts)
               - 1j * lamS[:, None] * ts[None, :] / 2
               - 0.5 * ts[None, :] ** 2 * sum_a_den
               - 0.5 * sum_log)
        cf = np.exp(lcf)
        sum_a_den2 = ((a2 * lam_top[:, None, :] / den ** 2).sum(2)
                      + (A1[:, None] + 2j * ts[None, :] * A2[:, None]))
        sum_a_den3 = ((a2 * lam_top[:, None, :] ** 2 / den ** 3).sum(2)
                      + A2[:, None])
        sum_l_den = ((lam_top[:, None, :] / den).sum(2)
                     + (S1[:, None] + 1j * ts[None, :] * S2[:, None]
                        - ts[None, :] ** 2 * S3[:, None]))
        sum_l2_den2 = ((lam_top[:, None, :] ** 2 / den ** 2).sum(2)
                       + (S2[:, None] + 2j * ts[None, :] * S3[:, None]))
        d1 = (1j * m_vec[:, None] - 0.5j * lamS[:, None]
              - ts[None, :] * sum_a_den
              - 0.5j * ts[None, :] ** 2 * sum_a_den2
              + 0.5j * sum_l_den)
        d2 = (-sum_a_den - 2j * ts[None, :] * sum_a_den2
              + ts[None, :] ** 2 * sum_a_den3
              - 0.5 * sum_l2_den2)
        cf2 = cf * (d1 * d1 + d2)
        accs["eabs"] += ((1 - cf.real) / ts[None, :] ** 2 * w).sum(1)
        accs["imcf2t"] += (cf2.imag / ts[None, :] * w).sum(1)
        accs["imcft"] += (cf.imag / ts[None, :] * w).sum(1)
        accs["recf"] += (cf.real * w).sum(1)
        accs["tim"] += (ts[None, :] * cf.imag * w).sum(1)
        accs["t2re"] += (ts[None, :] ** 2 * cf.real * w).sum(1)
    e1 = 0.5 * (m_vec + (2 / np.pi) * (accs["eabs"] + 1.0 / _T1))
    e2 = 0.5 * (Ez2 - (2 / np.pi) * accs["imcf2t"])
    pz = 0.5 + accs["imcft"] / np.pi
    p0 = accs["recf"] / np.pi
    p1 = accs["tim"] / np.pi
    p2 = -accs["t2re"] / np.pi
    return e1, e2 - e1 * e1, pz, p0, p1, p2


def batch_scalars(m_vec, lam_mat, at_mat, tblock=7000):
    """Vectorized certified scalars.

    m_vec: (B,); lam_mat, at_mat: (B, R). Returns six (B,) arrays:
    e1, var, pz, p0, p1, p2."""
    B, R = lam_mat.shape
    Ez2 = m_vec ** 2 + (at_mat ** 2).sum(1) + 0.5 * (lam_mat ** 2).sum(1)
    accs = {k: np.zeros(B) for k in
            ("eabs", "imcf2t", "imcft", "recf", "tim", "t2re")}
    for s in range(0, len(_TS), tblock):
        ts = _TS[s:s + tblock]
        w = _WTS[s:s + tblock]
        den = 1.0 - 1j * ts[None, :, None] * lam_mat[:, None, :]
        a2 = (at_mat ** 2)[:, None, :]
        lcf = (1j * np.outer(m_vec, ts)
               - 1j * lam_mat.sum(1)[:, None] * ts[None, :] / 2
               - 0.5 * ts[None, :] ** 2 * (a2 / den).sum(2)
               - 0.5 * np.log(den).sum(2))
        cf = np.exp(lcf)
        d1 = (1j * m_vec[:, None]
              - 0.5j * lam_mat.sum(1)[:, None]
              - ts[None, :] * (a2 / den).sum(2)
              - 0.5j * ts[None, :] ** 2
              * (a2 * lam_mat[:, None, :] / den ** 2).sum(2)
              + 0.5j * (lam_mat[:, None, :] / den).sum(2))
        d2 = (-(a2 / den).sum(2)
              - 2j * ts[None, :]
              * (a2 * lam_mat[:, None, :] / den ** 2).sum(2)
              + ts[None, :] ** 2
              * (a2 * lam_mat[:, None, :] ** 2 / den ** 3).sum(2)
              - 0.5 * (lam_mat[:, None, :] ** 2 / den ** 2).sum(2))
        cf2 = cf * (d1 * d1 + d2)
        accs["eabs"] += ((1 - cf.real) / ts[None, :] ** 2 * w).sum(1)
        accs["imcf2t"] += (cf2.imag / ts[None, :] * w).sum(1)
        accs["imcft"] += (cf.imag / ts[None, :] * w).sum(1)
        accs["recf"] += (cf.real * w).sum(1)
        accs["tim"] += (ts[None, :] * cf.imag * w).sum(1)
        accs["t2re"] += (ts[None, :] ** 2 * cf.real * w).sum(1)
    e1 = 0.5 * (m_vec + (2 / np.pi) * (accs["eabs"] + 1.0 / 300.0))
    e2 = 0.5 * (Ez2 - (2 / np.pi) * accs["imcf2t"])
    pz = 0.5 + accs["imcft"] / np.pi
    p0 = accs["recf"] / np.pi
    p1 = accs["tim"] / np.pi
    p2 = -accs["t2re"] / np.pi
    return e1, e2 - e1 * e1, pz, p0, p1, p2


def mps_materialize(dicts, pairs, chunk=None):
    """Yield (c0, M_chunk f64 CPU) for all units, batched on MPS."""
    import torch
    dev = torch.device("mps")
    n = pairs[0][2].shape[0]
    D = [torch.tensor(d, dtype=torch.float32, device=dev)
         for d in dicts]
    Ws = [(d1, d2, torch.tensor(w, dtype=torch.float32, device=dev))
          for (d1, d2, w) in pairs]
    if chunk is None:
        chunk = 64
    for c0 in range(0, n, chunk):
        ce = min(c0 + chunk, n)
        M = torch.zeros((ce - c0, D[0].shape[1], D[0].shape[1]),
                        dtype=torch.float32, device=dev)
        for (d1, d2, w) in Ws:
            scaled = D[d1].T[None, :, :] * w[c0:ce, None, :]
            term = torch.matmul(scaled, D[d2])
            if d1 == d2:
                M += term
            else:
                M += 0.5 * (term + term.transpose(1, 2))
        yield c0, M.to("cpu").numpy().astype(np.float64)
