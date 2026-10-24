#!/usr/bin/env python3
"""Does the exact L5-class pair correction predict the measured remainder?
L5 M11 terms: channels a+b=7 (L3*L4 conv) and a+b=9 (L3^3/6 conv), weights
from the deriver's exact tables using the pair's standardized cumulants.
Prediction should carry beta ~ 1 if the covariance path lacks exactly this."""
import json, sys
from math import factorial, erf
import numpy as np

_erf = np.vectorize(erf)

def c1_cols(m, v, qmax=20):
    v = max(v, 1e-30); sd = np.sqrt(v); al = m/sd
    pdf = np.exp(-0.5*al*al)/np.sqrt(2*np.pi)
    cdf = 0.5*(1+erf(al/np.sqrt(2)))
    ma = -al
    He = [1.0, ma]
    for k in range(2, qmax):
        He.append(ma*He[-1] - (k-1)*He[-2])
    cols = [sd*pdf + m*cdf, sd*cdf]
    for q in range(2, qmax):
        cols.append(sd*He[q-2]*pdf)
    return np.array(cols), sd
from math import erf

def l5_m11(mui, vi, k3i, k4i, muj, vj, k3j, k4j, cov, k21, k12, k22, k31, k13):
    sdi, sdj = np.sqrt(vi), np.sqrt(vj)
    rho = np.clip(cov/(sdi*sdj), -0.9999, 0.9999)
    ci, _ = c1_cols(mui, vi); cj, _ = c1_cols(muj, vj)
    # standardized cumulant blocks (deriver's convention)
    k30, k03 = k3i/sdi**3, k3j/sdj**3
    k21s, k12s = k21/(sdi**2*sdj), k12/(sdi*sdj**2)
    k40, k04 = k4i/sdi**4, k4j/sdj**4
    k31s, k13s = k31/(sdi**3*sdj), k13/(sdi*sdj**3)
    k22s = k22/(sdi**2*sdj**2)
    w3 = {0: k03/6, 1: k12s/2, 2: k21s/2, 3: k30/6}          # a-index
    w4 = {0: k04/24, 1: k13s/6, 2: k22s/4, 3: k31s/6, 4: k40/24}
    # L3*L4 channels a+b=7: weight[a] = sum_{p+q=a} w3[p]*w4[q]
    w34 = {}
    for a in range(8):
        w34[a] = sum(w3.get(p, 0)*w4.get(a-p, 0) for p in range(4))
    # L3^3/6 channels a+b=9
    w33 = {}
    for a in range(7):
        w33[a] = sum(w3.get(p, 0)*w3.get(a-p, 0) for p in range(4))
    w333 = {}
    for a in range(10):
        w333[a] = sum(w33.get(p, 0)*w3.get(a-p, 0) for p in range(7)) / 6.0
    def Gk(a, b, tmax=10):
        acc = 0.0; rp = 1.0
        for t in range(tmax+1):
            if t > 0: rp *= rho
            acc += rp/factorial(t) * (ci[1+0+a+t-1+1-1] if False else (ci[a+t]/sdi**a)) * (cj[b+t]/sdj**b)
        return acc
    dM11 = 0.0
    for a in range(8):
        b = 7 - a
        if abs(w34[a]) > 0:
            dM11 += w34[a]*Gk(a, b)
    for a in range(10):
        b = 9 - a
        if abs(w333[a]) > 0:
            dM11 += w333[a]*Gk(a, b)
    # univariate L5 mean shifts (diagonal cumulants only)
    g3i_, g4i_ = k30, k40
    g3j_, g4j_ = k03, k04
    dr1i = (g3i_*g4i_/144.0)*(ci[7]) + (g3i_**3/1296.0)*(ci[9])
    dr1j = (g3j_*g4j_/144.0)*(cj[7]) + (g3j_**3/1296.0)*(cj[9])
    # order-2 means (base + edgeworth) for the products
    r1i = ci[0] + (g3i_/6)*ci[3] + (g4i_/24)*ci[4] + (g3i_**2/72)*ci[6]
    r1j = cj[0] + (g3j_/6)*cj[3] + (g4j_/24)*cj[4] + (g3j_**2/72)*cj[6]
    return dM11 - r1i*dr1j - r1j*dr1i

parts = []
for s in (11, 12, 13, 14):
    parts.append(np.load(f"REMAINDER_CHAR_s{s}.npz")["data"])
A = np.concatenate(parts)
R = A[:, 8]
pred = np.array([l5_m11(*row[10:24]) for row in A])
b = float(np.sum(pred*R)/np.sum(pred*pred))
corr = float(np.sum(pred*R)/np.sqrt(np.sum(pred**2)*np.sum(R**2)))
r2_b1 = 1-float(np.sum((R-pred)**2)/np.sum(R**2))
r2_fit = 1-float(np.sum((R-b*pred)**2)/np.sum(R**2))
print(json.dumps({"N": len(A), "corr": round(corr,4),
                  "beta": round(b,4), "R2_at_beta1": round(r2_b1,4),
                  "R2_at_fitted_beta": round(r2_fit,4),
                  "R_rms": float(np.sqrt(np.mean(R**2))),
                  "pred_rms": float(np.sqrt(np.mean(pred**2)))}))
