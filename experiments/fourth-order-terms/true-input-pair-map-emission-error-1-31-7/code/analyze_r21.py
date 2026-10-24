#!/usr/bin/env python3
"""TAG g19d-r21-char-v1: characterize the M21 (pair (2,1)-moment) remainder
R21 = truth21 - analytic21 on the synthetic panels, mirroring the M11
characterization that found the universal 0.61x Lambda4-block. Features =
channel values the analytic already computes (recalibration hypothesis) plus
candidate NEW channels (k5-closure, quadratic-k3). Fit on seeds 31-33,
holdout 34. Report R^2 and coefficients.
Usage: analyze_r21.py"""
import json
from math import factorial, erf

import numpy as np

_erf = np.vectorize(erf)
COLS = ("ai,aj,rho,g3i,g3j,g4i,g4j,ss,R,truth,mui,vi,k3i,k4i,muj,vj,k3j,"
        "k4j,cov,k21,k12,k22,k31,k13,R21,truth21").split(",")
IX = {c: k for k, c in enumerate(COLS)}


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


def features_point(row):
    mui, vi = row[IX["mui"]], row[IX["vi"]]
    muj, vj = row[IX["muj"]], row[IX["vj"]]
    k3i, k4i = row[IX["k3i"]], row[IX["k4i"]]
    k3j, k4j = row[IX["k3j"]], row[IX["k4j"]]
    cov = row[IX["cov"]]
    k21, k12 = row[IX["k21"]], row[IX["k12"]]
    k22, k31, k13 = row[IX["k22"]], row[IX["k31"]], row[IX["k13"]]
    c1i, c2i, sdi = cols_scalar(mui, vi)
    c1j, c2j, sdj = cols_scalar(muj, vj)
    rho = np.clip(cov / (sdi * sdj), -0.9999, 0.9999)

    def S(a, b, tmax=14):
        acc = 0.0
        rp = 1.0
        for t in range(tmax + 1):
            if t > 0:
                rp *= rho
            acc += rp / factorial(t) * (c2i[a + t] / sdi ** a) \
                * (c1j[b + t] / sdj ** b)
        return acc

    k5i = 5 * k3i * k4i / vi - (40.0 / 9) * k3i ** 3 / vi ** 2
    k5j = 5 * k3j * k4j / vj - (40.0 / 9) * k3j ** 3 / vj ** 2
    feats = {
        "f30": (k3i / 6) * S(3, 0), "f03": (k3j / 6) * S(0, 3),
        "f21": (k21 / 2) * S(2, 1), "f12": (k12 / 2) * S(1, 2),
        "f40": (k4i / 24) * S(4, 0), "f04": (k4j / 24) * S(0, 4),
        "f22": (k22 / 4) * S(2, 2),
        "f31": (k31 / 6) * S(3, 1), "f13": (k13 / 6) * S(1, 3),
        "f50": (k5i / 120) * S(5, 0), "f05": (k5j / 120) * S(0, 5),
        "f41": (k3i * k3j / 12) * S(4, 1) if False else (k4i * k3j / 24 / 6 * 0
                                                          + k3i * k4j * 0),
        "f60": (k3i * k3i / 72) * S(6, 0), "f06": (k3j * k3j / 72) * S(0, 6),
        "f33": (k3i * k3j / 36) * S(3, 3),
        "f42": (k3i * k3i / 72) * S(4, 2) * 0 + (k4i * k4j / 576) * S(4, 4),
    }
    del feats["f41"], feats["f42"]
    return feats


def main():
    rows = {}
    for s in (31, 32, 33, 34):
        d = np.load(f"REMAINDER_CHAR_s{s}_x1.npz")
        rows[s] = d["data"]
    names = None
    Xs, ys = {}, {}
    for s, A in rows.items():
        F = []
        for r in A:
            f = features_point(r)
            if names is None:
                names = sorted(f)
            F.append([f[n] for n in names])
        Xs[s] = np.array(F)
        ys[s] = A[:, IX["R21"]]

    def fit(seeds):
        X = np.concatenate([Xs[s] for s in seeds])
        y = np.concatenate([ys[s] for s in seeds])
        G = X.T @ X + 1e-10 * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
        return np.linalg.solve(G, X.T @ y)

    beta = fit([31, 32, 33])
    out = {"names": names,
           "beta": {n: round(float(b), 4) for n, b in zip(names, beta)}}
    for s in (31, 32, 33, 34):
        pred = Xs[s] @ beta
        ss = float(np.sum(ys[s] ** 2))
        out[f"s{s}"] = {
            "R2": round(1 - float(np.sum((ys[s] - pred) ** 2)) / ss, 4),
            "corr": round(float(np.sum(pred * ys[s])
                          / np.sqrt(np.sum(pred ** 2) * ss)), 4),
            "R21_rms": float(np.sqrt(np.mean(ys[s] ** 2))),
            "rel_vs_truth21": float(np.sqrt(np.mean(ys[s] ** 2)
                                    / np.mean(rows[s][:, IX["truth21"]] ** 2))),
        }
    # ladder: 4th-order recal only vs full
    sub = [n for n in names if n in ("f40", "f04", "f22", "f31", "f13")]
    idx = [names.index(n) for n in sub]

    def fit_sub(seeds):
        X = np.concatenate([Xs[s] for s in seeds])[:, idx]
        y = np.concatenate([ys[s] for s in seeds])
        G = X.T @ X + 1e-10 * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
        return np.linalg.solve(G, X.T @ y)

    bs = fit_sub([31, 32, 33])
    pred4 = Xs[34][:, idx] @ bs
    out["holdout_L4only"] = {
        "R2": round(1 - float(np.sum((ys[34] - pred4) ** 2)
                    / np.sum(ys[34] ** 2)), 4),
        "beta": {n: round(float(b), 4) for n, b in zip(sub, bs)}}
    open("R21_CHAR_RESULT.json", "w").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
