"""Offline check: how much of the Gaussian post-ReLU covariance lies beyond Mehler order 6 (the card's gauss_order),
and what relative error that tail induces in the NEXT layer's per-unit pre-activation variance.
Gaussian-closure propagation (mean + covariance, no higher cumulants), numpy float64, one generated network."""
import sys, math, numpy as np
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh')
import loader
from math import erf

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7100136
R_HI = 60          # reference order
R_CARD = 6         # card's gauss_order
W = loader.weights(seed).astype(np.float64)
n = 1024

def Phi(x): return 0.5 * (1 + np.vectorize(erf)(x / math.sqrt(2)))
def phi(x): return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def coeffs(a, R):
    """c_r(a) for relu(a+x), r=0..R: c0 = a Phi + phi, c1 = Phi, c_r = (-1)^r He_{r-2}(a) phi(a)."""
    c = np.zeros((R + 1, a.size))
    c[0] = a * Phi(a) + phi(a); c[1] = Phi(a)
    He_prev, He = np.zeros_like(a), np.ones_like(a)   # He_{-1}=0, He_0=1
    for r in range(2, R + 1):
        k = r - 2
        if k >= 1:
            He_prev, He = He, a * He - (k - 1) * He_prev   # He_k = a He_{k-1} - (k-1) He_{k-2}
        c[r] = ((-1) ** r) * He * phi(a)
    return c

def check_coeffs():
    # sanity: sum_r c_r^2 / r! = Var(relu(a+x)) + c0^2 (= E relu^2)
    a = np.array([-1.0, 0.0, 0.7])
    c = coeffs(a, 80)
    s = sum(c[r] ** 2 / math.factorial(r) for r in range(81))
    Er2 = (a * a + 1) * Phi(a) + a * phi(a)
    print('coeff check E relu^2:', s, Er2)

check_coeffs(); sys.stdout.flush()

# layer 0 input: x ~ N(0, I) -> z0 = x @ W[0]
mu = np.zeros(n); C = W[0].T @ W[0]
for l in range(16):
    v = np.diag(C).copy(); s = np.sqrt(v); a = mu / s
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0.0)
    c = coeffs(a, R_HI)
    # Mehler sums
    rr = np.ones_like(rho); full = np.zeros_like(rho); card = np.zeros_like(rho)
    for r in range(1, R_HI + 1):
        rr = rr * rho / r
        term = rr * np.outer(c[r], c[r])
        full += term
        if r <= R_CARD: card += term
    tail = (full - card) * np.outer(s, s)
    Cy = full * np.outer(s, s)
    m = s * c[0]
    Ey2 = v * ((a * a + 1) * Phi(a) + a * phi(a))
    np.fill_diagonal(Cy, Ey2 - m * m); np.fill_diagonal(tail, 0.0)
    offd = np.abs(rho[np.triu_indices(n, 1)])
    reff = np.trace(Cy) ** 2 / (Cy * Cy).sum()
    msg = (f'L{l:2d} |rho| mean {offd.mean():.3f} p99 {np.quantile(offd,0.99):.3f} frac>0.7 {np.mean(offd>0.7):.4f} '
           f'frac>0.9 {np.mean(offd>0.9):.5f} | post-cov eff.rank {reff:6.1f} | tail/|Cy offdiag| rms '
           f'{np.sqrt((tail**2).sum()/((Cy-np.diag(np.diag(Cy)))**2).sum()):.2e}')
    if l < 15:
        Wn = W[l + 1]
        vn = np.sum(Wn * (Cy @ Wn), axis=0)       # next pre-activation variance
        dvn = np.sum(Wn * (tail @ Wn), axis=0)    # part carried by the order>6 tail
        rel = dvn / vn
        msg += f' | next v: tail share mean {rel.mean():+.2e} rms {np.sqrt((rel**2).mean()):.2e}'
        mu = m @ Wn; C = Wn.T @ Cy @ Wn
    print(msg, flush=True)
