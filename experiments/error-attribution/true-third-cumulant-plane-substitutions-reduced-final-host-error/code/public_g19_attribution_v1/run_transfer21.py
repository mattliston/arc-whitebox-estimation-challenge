#!/usr/bin/env python3
"""TAG g19d-r21-transfer-v1: apply the synthetic-fitted universal remainder
corrections (M21 from R21_CHAR betas; M11 refit here with the same basis) at
the REAL closure emission sites (rank 35, layers with true post-planes) and
score the predicted emission correction dK21_hat against the true emission
error (true post-K21 - closure k21_target). Decision: corr >= 0.6 -> wire
into the tangent chain. Usage: run_transfer21.py [rank]"""
import json
import sys
from math import erf, factorial

import numpy as np

_erf = np.vectorize(erf)
TMAX = 14
COLS = ("ai,aj,rho,g3i,g3j,g4i,g4j,ss,R,truth,mui,vi,k3i,k4i,muj,vj,k3j,"
        "k4j,cov,k21,k12,k22,k31,k13,R21,truth21").split(",")
IX = {c: k for k, c in enumerate(COLS)}
NAMES = ["f03", "f04", "f05", "f06", "f12", "f13", "f21", "f22", "f30",
         "f31", "f33", "f40", "f50", "f60"]


def cols_scalar(m, v, qmax=22):
    v = max(v, 1e-30)
    sd = np.sqrt(v)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(al / np.sqrt(2)))
    ma = -al
    He = [1.0, ma]
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
    return np.array(c1), np.array(c2), sd


def features_point(row, outer):
    mui, vi = row[IX["mui"]], row[IX["vi"]]
    muj, vj = row[IX["muj"]], row[IX["vj"]]
    k3i, k4i = row[IX["k3i"]], row[IX["k4i"]]
    k3j, k4j = row[IX["k3j"]], row[IX["k4j"]]
    cov = row[IX["cov"]]
    k21, k12 = row[IX["k21"]], row[IX["k12"]]
    k22, k31, k13 = row[IX["k22"]], row[IX["k31"]], row[IX["k13"]]
    c1i, c2i, sdi = cols_scalar(mui, vi)
    c1j, c1j2, sdj = cols_scalar(muj, vj)
    A = c2i if outer == 2 else c1i
    rho = np.clip(cov / (sdi * sdj), -0.9999, 0.9999)

    def S(a, b):
        acc = 0.0
        rp = 1.0
        for t in range(TMAX + 1):
            if t > 0:
                rp *= rho
            acc += rp / factorial(t) * (A[a + t] / sdi ** a) \
                * (c1j[b + t] / sdj ** b)
        return acc

    k5i = 5 * k3i * k4i / vi - (40.0 / 9) * k3i ** 3 / vi ** 2
    k5j = 5 * k3j * k4j / vj - (40.0 / 9) * k3j ** 3 / vj ** 2
    return {
        "f30": (k3i / 6) * S(3, 0), "f03": (k3j / 6) * S(0, 3),
        "f21": (k21 / 2) * S(2, 1), "f12": (k12 / 2) * S(1, 2),
        "f40": (k4i / 24) * S(4, 0), "f04": (k4j / 24) * S(0, 4),
        "f22": (k22 / 4) * S(2, 2),
        "f31": (k31 / 6) * S(3, 1), "f13": (k13 / 6) * S(1, 3),
        "f50": (k5i / 120) * S(5, 0), "f05": (k5j / 120) * S(0, 5),
        "f60": (k3i * k3i / 72) * S(6, 0), "f06": (k3j * k3j / 72) * S(0, 6),
        "f33": (k3i * k3j / 36) * S(3, 3),
    }


def fit_betas(target_col, outer):
    Xs, ys = [], []
    for s in (31, 32, 33, 34):
        A = np.load(f"REMAINDER_CHAR_s{s}_x1.npz")["data"]
        for r in A:
            f = features_point(r, outer)
            Xs.append([f[n] for n in NAMES])
            ys.append(r[IX[target_col]])
    X, y = np.array(Xs), np.array(ys)
    G = X.T @ X + 1e-10 * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    beta = np.linalg.solve(G, X.T @ y)
    pred = X @ beta
    r2 = 1 - float(np.sum((y - pred) ** 2) / np.sum(y ** 2))
    return {n: float(b) for n, b in zip(NAMES, beta)}, r2


def cols_vec(m, v, qmax=22):
    v = np.maximum(v, 1e-30)
    sd = np.sqrt(v)
    a = m / sd
    pdf = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(a / np.sqrt(2)))
    ma = -a
    He = [np.ones_like(a), ma]
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
    return np.stack(c1), np.stack(c2), sd


def delta_map(beta, outer, m, v, k3, k4, C, K21c, K22c, K31c):
    c1, c2, sd = cols_vec(m, v)
    A = c2 if outer == 2 else c1
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def S(a, b):
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(TMAX + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / factorial(t) * np.outer(A[a + t] / sd ** a,
                                                c1[b + t] / sd ** b)
        return acc

    k5 = 5 * k3 * k4 / v - (40.0 / 9) * k3 ** 3 / v ** 2
    F = {
        "f30": (k3[:, None] / 6) * S(3, 0), "f03": (k3[None, :] / 6) * S(0, 3),
        "f21": (K21c / 2) * S(2, 1), "f12": (K21c.T / 2) * S(1, 2),
        "f40": (k4[:, None] / 24) * S(4, 0),
        "f04": (k4[None, :] / 24) * S(0, 4),
        "f22": (K22c / 4) * S(2, 2),
        "f31": (K31c / 6) * S(3, 1), "f13": (K31c.T / 6) * S(1, 3),
        "f50": (k5[:, None] / 120) * S(5, 0),
        "f05": (k5[None, :] / 120) * S(0, 5),
        "f60": ((k3 * k3)[:, None] / 72) * S(6, 0),
        "f06": ((k3 * k3)[None, :] / 72) * S(0, 6),
        "f33": (np.outer(k3, k3) / 36) * S(3, 3),
    }
    out = np.zeros_like(rho)
    for n in NAMES:
        out += beta[n] * F[n]
    return out


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    beta21, r2_21 = fit_betas("R21", 2)
    beta11, r2_11 = fit_betas("R", 1)
    print(json.dumps({"fit_R2": {"M21": round(r2_21, 4),
                                 "M11": round(r2_11, 4)}}))
    d = np.load(f"states_rank{rank:04d}.npz")
    z = np.load(f"host_cov_rank{rank:04d}_base.npz")
    Pp = np.load(f"mps_planes_rank{rank:04d}_post.npz")
    kt = np.load(f"k21_targets_rank{rank:04d}.npz")["kt"].astype(np.float64)
    sys.path.insert(0, ".")
    from run_attribution import edg2_relu_mean
    out = {"rank": rank}
    for l in (5, 8, 12, 13):
        key = f"K21_{l}"
        if key not in Pp:
            continue
        i = l - 1
        m, v = d["host_m"][i], d["host_v"][i]
        k3, k4 = d["host_k3"][i], d["host_k4"][i]
        C = z["cov"][i].astype(np.float64)
        K21c = z["k21"][i].astype(np.float64)
        K22c = z["k22"][i].astype(np.float64)
        K31c = z["k31"][i].astype(np.float64)
        dM21 = delta_map(beta21, 2, m, v, k3, k4, C, K21c, K22c, K31c)
        dM11 = delta_map(beta11, 1, m, v, k3, k4, C, K21c, K22c, K31c)
        r1 = edg2_relu_mean(m, v, k3, k4)
        dK21 = dM21 - 2 * r1[:, None] * dM11
        Tt = Pp[key].astype(np.float64)
        base = kt[l]
        dtrue = Tt - base
        scale = float(np.sqrt(np.mean(Tt ** 2)))
        rel_before = float(np.sqrt(np.mean(dtrue ** 2))) / scale
        res = dtrue - dK21
        rel_after = float(np.sqrt(np.mean(res ** 2))) / scale
        corr = float(np.sum(dK21 * dtrue)
                     / np.sqrt(np.sum(dK21 ** 2) * np.sum(dtrue ** 2)))
        b = float(np.sum(dK21 * dtrue) / np.sum(dK21 ** 2))
        out[f"L{l}"] = {"rel_before": round(rel_before, 4),
                        "rel_after": round(rel_after, 4),
                        "corr": round(corr, 4), "beta": round(b, 4)}
        print(json.dumps({f"L{l}": out[f"L{l}"]}))
    open(f"TRANSFER21_rank{rank:04d}.json", "w").write(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
