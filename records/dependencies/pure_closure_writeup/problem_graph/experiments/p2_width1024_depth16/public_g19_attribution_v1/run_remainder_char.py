#!/usr/bin/env python3
"""TAG g19b-remainder-char-v1: characterize the beyond-Edgeworth remainder
R = E[relu(h_i)relu(h_j)] - (OwensGauss + Edgeworth channels) on synthetic
pairs with EXACTLY known cumulants.
Pair law: h_u = m_u + s_u (g + a_u He2(g) + b_u He3(g)), corr(g_i,g_j) = r.
Exact moments/cumulants: 1D GH (polynomial-exact) + He-product identities.
Truth: root-split nested Gauss-Legendre (~1e-9).
Analytic: Owens-T Gaussian pair moment + channels (30,03,21,12,40,04,22).
Output: REMAINDER_CHAR.npz + fit report."""
import json, sys
from math import factorial, erf
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
sys.path.insert(0, str(Path('.').resolve().parent.parent / "x3_r0c1_kappa3_ceiling"))
from a0_gaussian import relu_gaussian_pair_second_moment

_erf = np.vectorize(erf)
GH_X, GH_W = hermegauss(80)
GH_W = GH_W / GH_W.sum()
GL_X, GL_W = leggauss(48)

def poly_coeffs(m, s, a, b):
    # h(g) = m + s(g + a(g^2-1) + b(g^3-3g)) as polynomial in g
    return np.array([m - s*a, s*(1 - 3*b), s*a, s*b])  # c0..c3

def h_moments(m, s, a, b, kmax=4):
    c = poly_coeffs(m, s, a, b)
    g = GH_X
    h = c[0] + c[1]*g + c[2]*g*g + c[3]*g**3
    return [float(np.sum(GH_W * h**k)) for k in range(1, kmax+1)]

def cumulants(mo):
    m1, m2, m3, m4 = mo
    mu = m1
    c2 = m2 - m1*m1
    c3 = m3 - 3*m1*m2 + 2*m1**3
    c4 = m4 - 4*m1*m3 - 3*m2*m2 + 12*m1*m1*m2 - 6*m1**4
    return mu, c2, c3, c4

def joint_stats(mi, si, ai, bi, mj, sj, aj, bj, r):
    # He expansion: h_u - E = s(He1 + a He2 + b He3); E[He_p(gi)He_q(gj)] = p! r^p delta_pq
    ci = np.array([1.0, ai, bi]); cj = np.array([1.0, aj, bj])
    cov = si*sj*sum(ci[p]*cj[p]*factorial(p+1)*r**(p+1) for p in range(3))
    # k21 = cum(hi,hi,hj): E[(dhi)^2 dhj] with dh = s(sum_p c_p He_{p+1})
    # E[He_a He_b (gi) He_c (gj)] via linearization He_a He_b = sum ...; do numerically 2D GH:
    return cov

G2_X, G2_W = hermegauss(48)
def joint_cum_num(mi, si, ai, bi, mj, sj, aj, bj, r):
    u = G2_X; wu = G2_W/G2_W.sum()
    U, V = np.meshgrid(u, u, indexing='ij')
    Wt = np.outer(wu, wu)
    gi = U; gj = r*U + np.sqrt(1-r*r)*V
    hi = mi + si*(gi + ai*(gi*gi-1) + bi*(gi**3-3*gi))
    hj = mj + sj*(gj + aj*(gj*gj-1) + bj*(gj**3-3*gj))
    E = lambda X: float(np.sum(Wt*X))
    m1i, m1j = E(hi), E(hj)
    di, dj = hi-m1i, hj-m1j
    cov = E(di*dj)
    k21 = E(di*di*dj)
    k12 = E(di*dj*dj)
    k22 = E(di*di*dj*dj) - E(di*di)*E(dj*dj) - 2*cov*cov
    vi_, vj_ = E(di*di), E(dj*dj)
    k31 = E(di*di*di*dj) - 3*vi_*cov
    k13 = E(di*dj*dj*dj) - 3*vj_*cov
    return cov, k21, k12, k22, k31, k13

def true_pair_moment21(mi, si, ai, bi, mj, sj, aj, bj, r):
    ci = poly_coeffs(mi, si, ai, bi); cj = poly_coeffs(mj, sj, aj, bj)
    def relu_poly(c, g):
        return np.maximum(c[0] + c[1]*g + c[2]*g*g + c[3]*g**3, 0.0)
    def segments(c, lo=-8.0, hi=8.0):
        roots = np.roots(c[::-1])
        pts = sorted([lo] + [float(x.real) for x in roots
                     if abs(x.imag) < 1e-10 and lo < x.real < hi] + [hi])
        return list(zip(pts[:-1], pts[1:]))
    total = 0.0
    for (a1, b1) in segments(ci):
        xg = 0.5*(a1+b1) + 0.5*(b1-a1)*GL_X
        wg = 0.5*(b1-a1)*GL_W
        phi_i = np.exp(-0.5*xg*xg)/np.sqrt(2*np.pi)
        hi_v = relu_poly(ci, xg)**2
        mask = hi_v > 0
        if not mask.any():
            continue
        inner = np.zeros_like(xg)
        mu_j = r*xg; s_j = np.sqrt(1-r*r)
        for (a2, b2) in segments(cj):
            yg = 0.5*(a2+b2) + 0.5*(b2-a2)*GL_X
            wy = 0.5*(b2-a2)*GL_W
            hj_v = relu_poly(cj, yg)
            for k in range(len(xg)):
                if not mask[k]:
                    continue
                dens = np.exp(-0.5*((yg-mu_j[k])/s_j)**2)/(s_j*np.sqrt(2*np.pi))
                inner[k] += float(np.sum(wy*dens*hj_v))
        total += float(np.sum(wg*phi_i*hi_v*inner))
    return total


def true_pair_moment(mi, si, ai, bi, mj, sj, aj, bj, r):
    ci = poly_coeffs(mi, si, ai, bi); cj = poly_coeffs(mj, sj, aj, bj)
    def relu_poly(c, g):
        return np.maximum(c[0] + c[1]*g + c[2]*g*g + c[3]*g**3, 0.0)
    # outer over gi with segment splits at real roots of P_i
    def segments(c, lo=-8.0, hi=8.0):
        roots = np.roots(c[::-1])
        pts = sorted([lo] + [float(x.real) for x in roots
                     if abs(x.imag) < 1e-10 and lo < x.real < hi] + [hi])
        return list(zip(pts[:-1], pts[1:]))
    total = 0.0
    for (a1, b1) in segments(ci):
        xg = 0.5*(a1+b1) + 0.5*(b1-a1)*GL_X
        wg = 0.5*(b1-a1)*GL_W
        phi_i = np.exp(-0.5*xg*xg)/np.sqrt(2*np.pi)
        hi_v = relu_poly(ci, xg)
        mask = hi_v > 0
        if not mask.any():
            continue
        inner = np.zeros_like(xg)
        mu_j = r*xg; s_j = np.sqrt(1-r*r)
        for (a2, b2) in segments(cj):
            # integrate over gj in [a2,b2] against N(mu_j, s_j^2)
            yg = 0.5*(a2+b2) + 0.5*(b2-a2)*GL_X
            wy = 0.5*(b2-a2)*GL_W
            hj_v = relu_poly(cj, yg)
            for k in range(len(xg)):
                if not mask[k]:
                    continue
                dens = np.exp(-0.5*((yg-mu_j[k])/s_j)**2)/(s_j*np.sqrt(2*np.pi))
                inner[k] += float(np.sum(wy*dens*hj_v))
        total += float(np.sum(wg*phi_i*hi_v*inner))
    return total

def c1_cols(m, v, qmax=14):
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

def c2_cols_scalar(m, v, qmax=20):
    import math
    v = max(v, 1e-30); sd = np.sqrt(v); al = m/sd
    pdf = np.exp(-0.5*al*al)/np.sqrt(2*np.pi)
    cdf = 0.5*(1+erf(al/np.sqrt(2)))
    ma = -al
    He = [1.0, ma]
    for k in range(2, qmax+2):
        He.append(ma*He[-1] - (k-1)*He[-2])
    c1 = [sd*pdf + m*cdf, sd*cdf]
    for q in range(2, qmax):
        c1.append(sd*He[q-2]*pdf)
    raw1 = c1[0]
    raw2 = (m*m+v)*cdf + m*sd*pdf
    c2 = [raw2, 2*sd*raw1, 2*v*cdf]
    for q in range(3, qmax):
        c2.append(2*v*He[q-3]*pdf)
    return np.array(c1), np.array(c2), sd


def analytic_pair21(mi, vi, k3i, k4i, mj, vj, k3j, k4j, cov, k21, k12, k22, k31, k13):
    from math import factorial
    c1i, c2i, sdi = c2_cols_scalar(mi, vi)
    c1j, c2j, sdj = c2_cols_scalar(mj, vj)
    rho = np.clip(cov/(sdi*sdj), -0.9999, 0.9999)
    def series(colsA, colsB, a, b, tmax=14):
        acc = 0.0; rp = 1.0
        for t in range(tmax+1):
            if t > 0: rp *= rho
            acc += rp/factorial(t)*(colsA[a+t]/sdi**a)*(colsB[b+t]/sdj**b)
        return acc
    def dM(colsA, colsB):
        return ((k3i/6/1)*series(colsA, colsB, 3, 0) + (k3j/6)*series(colsA, colsB, 0, 3)
                + (k21/2)*series(colsA, colsB, 2, 1) + (k12/2)*series(colsA, colsB, 1, 2)
                + (k4i/24)*series(colsA, colsB, 4, 0) + (k4j/24)*series(colsA, colsB, 0, 4)
                + (k22/4)*series(colsA, colsB, 2, 2)
                + (k31/6)*series(colsA, colsB, 3, 1) + (k13/6)*series(colsA, colsB, 1, 3))
    return series(c2i, c1j, 0, 0) + dM(c2i, c1j)


def analytic_pair(mi, vi, k3i, k4i, mj, vj, k3j, k4j, cov, k21, k12, k22):
    sdi, sdj = np.sqrt(vi), np.sqrt(vj)
    rho = np.clip(cov/(sdi*sdj), -0.9999, 0.9999)
    M = relu_gaussian_pair_second_moment(np.array([mi, mj]),
                                         np.array([sdi, sdj]),
                                         np.array([[1, rho], [rho, 1]]))[0, 1]
    ci_, _ = c1_cols(mi, vi); cj_, _ = c1_cols(mj, vj)
    def Gk(a, b, tmax=8):
        acc = 0.0; rp = 1.0
        for t in range(tmax+1):
            if t > 0: rp *= rho
            acc += rp/factorial(t) * (ci_[a+t]/sdi**a) * (cj_[b+t]/sdj**b)
        return acc
    dM = (k3i/6.0)*Gk(3,0) + (k3j/6.0)*Gk(0,3) \
         + (k21/2.0)*Gk(2,1) + (k12/2.0)*Gk(1,2) \
         + (k4i/24.0)*Gk(4,0) + (k4j/24.0)*Gk(0,4) + (k22/4.0)*Gk(2,2)
    return M + dM

def main():
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = np.random.default_rng(seed)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rows = []
    for n in range(N):
        alpha_i, alpha_j = rng.uniform(-1.0, 2.5, 2)
        s_i, s_j = rng.uniform(0.3, 1.2, 2)
        scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        a_i, a_j = scale * rng.uniform(0.004, 0.055, 2)
        b_i, b_j = scale * rng.uniform(0.0002, 0.0035, 2)
        r = rng.uniform(-0.5, 0.85)
        mi, mj = alpha_i*s_i, alpha_j*s_j
        moi = h_moments(mi, s_i, a_i, b_i)
        moj = h_moments(mj, s_j, a_j, b_j)
        mui, vi, k3i, k4i = cumulants(moi)
        muj, vj, k3j, k4j = cumulants(moj)
        cov, k21, k12, k22, k31, k13 = joint_cum_num(mi, s_i, a_i, b_i,
                                           mj, s_j, a_j, b_j, r)
        truth = true_pair_moment(mi, s_i, a_i, b_i, mj, s_j, a_j, b_j, r)
        truth21 = true_pair_moment21(mi, s_i, a_i, b_i, mj, s_j, a_j, b_j, r)
        ana21 = analytic_pair21(mui, vi, k3i, k4i, muj, vj, k3j, k4j,
                                cov, k21, k12, k22, k31, k13)
        R21 = truth21 - ana21
        r1i = float(np.sum((GH_W)*np.maximum(
            poly_coeffs(mi, s_i, a_i, b_i) @ np.stack(
                [np.ones_like(GH_X), GH_X, GH_X**2, GH_X**3]), 0)))
        ana = analytic_pair(mui, vi, k3i, k4i, muj, vj, k3j, k4j,
                            cov, k21, k12, k22)
        R = truth - ana
        g3i, g4i = k3i/vi**1.5, k4i/vi**2
        g3j, g4j = k3j/vj**1.5, k4j/vj**2
        rho = cov/np.sqrt(vi*vj)
        rows.append([mui/np.sqrt(vi), muj/np.sqrt(vj), rho,
                     g3i, g3j, g4i, g4j, np.sqrt(vi*vj), R, truth,
                     mui, vi, k3i, k4i, muj, vj, k3j, k4j,
                     cov, k21, k12, k22, k31, k13, R21, truth21])
        if n % 100 == 0:
            print(json.dumps({"n": n}), flush=True)
    A = np.array(rows)
    np.savez(f"REMAINDER_CHAR_s{seed}_x{sys.argv[3] if len(sys.argv)>3 else 1}.npz", data=A,
             cols="ai,aj,rho,g3i,g3j,g4i,g4j,ss,R,truth,mui,vi,k3i,k4i,muj,vj,k3j,k4j,cov,k21,k12,k22,k31,k13,R21,truth21")
    R = A[:, 8]
    print(json.dumps({
        "N": len(A),
        "R_rms": float(np.sqrt(np.mean(R**2))),
        "R_rel_rms_vs_truth": float(np.sqrt(np.mean((R/np.maximum(np.abs(A[:,9]),1e-6))**2))),
        "R_over_ss_rms": float(np.sqrt(np.mean((R/A[:,7])**2)))}))

if __name__ == "__main__":
    main()
