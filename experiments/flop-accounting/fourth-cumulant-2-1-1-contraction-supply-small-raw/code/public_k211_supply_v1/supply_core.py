"""Exact T2-truncated kappa211 supply into the next marginal kappa4.

S(a) = 3*(T1 + 2*T2), brute-verified:
T1 = sum_i W2_ia V2_i [ (sum_j R_ij U1_j W_ja)^2 - sum_j R_ij^2 U1_j^2 W_ja^2 ]
T2 = sum_i W2_ia B1_i [ sum_j R_ij U2_j W_ja M1_ja - U1_i W_ia sum_j R_ij^2 U2_j W_ja ]
with M1 = R @ (U1*W), R = rho with zero diagonal,
U1 = c1[1], U2 = c1[2], B1 = c2[1]-2m c1[1], V2 = c2[2]-2m c1[2]-2 c1[1]^2.
"""
import numpy as np


def kappa211_supply(W, rho, c1_0, c1_1, c1_2, c2_1, c2_2):
    m = c1_0
    U1, U2 = c1_1, c1_2
    B1 = c2_1 - 2.0 * m * c1_1
    V2 = c2_2 - 2.0 * m * c1_2 - 2.0 * c1_1 ** 2
    W2 = W * W
    R = rho - np.eye(rho.shape[0])
    RR = R * R
    M1 = R @ (U1[:, None] * W)
    T1 = ((W2 * V2[:, None]) * (M1 * M1 - RR @ ((U1 * U1)[:, None] * W2))).sum(axis=0)
    inner2 = R @ ((U2[:, None] * W) * M1)
    corr = (U1[:, None] * W) * (RR @ (U2[:, None] * W))
    T2 = ((W2 * B1[:, None]) * (inner2 - corr)).sum(axis=0)
    return 3.0 * (T1 + 2.0 * T2)


def kappa211_supply_brute(W, rho, c1_0, c1_1, c1_2, c2_1, c2_2):
    n, na = W.shape
    m = c1_0
    U1, U2 = c1_1, c1_2
    B1 = c2_1 - 2.0 * m * c1_1
    V2 = c2_2 - 2.0 * m * c1_2 - 2.0 * c1_1 ** 2
    S = np.zeros(na)
    for i in range(n):
        for j in range(n):
            if j == i: continue
            for k in range(n):
                if k == i or k == j: continue
                k211 = (V2[i] * rho[i, j] * rho[i, k] * U1[j] * U1[k]
                        + B1[i] * U2[j] * U1[k] * rho[i, j] * rho[j, k]
                        + B1[i] * U1[j] * U2[k] * rho[i, k] * rho[j, k])
                S += 3.0 * (W[i] ** 2) * W[j] * W[k] * k211
    return S
