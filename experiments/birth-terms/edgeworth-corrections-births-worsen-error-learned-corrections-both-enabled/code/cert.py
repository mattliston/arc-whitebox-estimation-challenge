#!/usr/bin/env python3
"""TAG fable-physicist (EW-0): certificate for the first-order Edgeworth birth coefficients before use.
x = mu + sigma*z, z a standardized gamma (shape k: skew t3 = 2/sqrt(k), excess kurtosis t4 = 6/k), exact scipy CDF/PDF.
Checks, with a = mu/sigma, phi = pdf(a), d = phi/sigma:
  E[f']  = P(x>0) ~ Phi(a) + phi*(t3/6*(a^2-1) + t4/24*(3a-a^3))
  E[f''] = p(0)   ~ d*(1 + t3/6*(3a-a^3) + t4/24*(a^4-6a^2+3))
  E[f]   = E[relu] ~ mg + k3*e/6 + k4*f/24   (the card's own host formula; e = -a*phi/v, f = (a^2-1)*phi/(v*sigma))
Pass = the Edgeworth error is several times smaller than the Gaussian error at small skew, and shrinks ~O(t3^2) when k grows."""
import numpy as np
from scipy import stats, integrate
sigma = 1.3
print('   k     a |  P(x>0): gauss err  edgew err |   p(0): gauss err  edgew err |  E[relu]: gauss err  edgew err')
for k in (100., 400., 1600.):
    t3, t4 = 2 / np.sqrt(k), 6 / k
    Z = stats.gamma(k, loc=-np.sqrt(k), scale=1 / np.sqrt(k))     # mean 0, var 1, skew t3
    for a in (-1.5, -0.5, 0.0, 0.7, 1.5):
        mu = a * sigma; v = sigma ** 2
        phi = stats.norm.pdf(a); Phi = stats.norm.cdf(a); d = phi / sigma
        P_true = Z.sf(-a)                                          # P(mu + sigma z > 0) = P(z > -a)
        p0_true = Z.pdf(-a) / sigma
        m_true = integrate.quad(lambda z: (mu + sigma * z) * Z.pdf(z), -a, np.inf, limit=200)[0]
        P_ew = Phi + phi * (t3 / 6 * (a * a - 1) + t4 / 24 * (3 * a - a ** 3))
        p0_ew = d * (1 + t3 / 6 * (3 * a - a ** 3) + t4 / 24 * (a ** 4 - 6 * a * a + 3))
        mg = sigma * phi + mu * Phi; k3, k4 = t3 * sigma ** 3, t4 * v * v
        e = -a * phi / v; f = (a * a - 1) * phi / (v * sigma)
        m_ew = mg + k3 * e / 6 + k4 * f / 24
        print(f'{k:5.0f} {a:5.1f} | {Phi - P_true:+.2e} {P_ew - P_true:+.2e} | {d - p0_true:+.2e} {p0_ew - p0_true:+.2e} | '
              f'{mg - m_true:+.2e} {m_ew - m_true:+.2e}')
# consistency: the corrections equal mu-derivatives of the card's host correction (k3, k4, sigma fixed)
a = 0.37; sig = 1.1; k3, k4 = 0.2, 0.05; h = 1e-5
def corr(mu):
    aa = mu / sig; ph = stats.norm.pdf(aa); v = sig * sig
    return k3 * (-aa * ph / v) / 6 + k4 * ((aa * aa - 1) * ph / (v * sig)) / 24
mu = a * sig; ph = stats.norm.pdf(a); t3, t4 = k3 / sig ** 3, k4 / sig ** 4
dg = ph * (t3 / 6 * (a * a - 1) + t4 / 24 * (3 * a - a ** 3))
db = ph / sig * (t3 / 6 * (3 * a - a ** 3) + t4 / 24 * (a ** 4 - 6 * a * a + 3))
fd1 = (corr(mu + h) - corr(mu - h)) / (2 * h); fd2 = (corr(mu + h) - 2 * corr(mu) + corr(mu - h)) / h ** 2
print(f'derivative check: dg {dg:.8f} vs d/dmu {fd1:.8f}; dbirth {db:.6f} vs d2/dmu2 {fd2:.6f}')
# second order = the t4 terms PLUS the t3^2 terms (same order in 1/n); He5(-a) = -(a^5-10a^3+15a), He6 even
print('second order (t4 + t3^2), k = 100 and 400:  P(x>0) err, p(0) err, E[relu] err')
for k in (100., 400.):
    t3, t4 = 2 / np.sqrt(k), 6 / k
    Z = stats.gamma(k, loc=-np.sqrt(k), scale=1 / np.sqrt(k))
    for a in (-1.5, -0.5, 0.0, 0.7, 1.5):
        mu = a * sigma; v = sigma ** 2; phi = stats.norm.pdf(a); Phi = stats.norm.cdf(a); d = phi / sigma
        he5m = -(a ** 5 - 10 * a ** 3 + 15 * a); he6 = a ** 6 - 15 * a ** 4 + 45 * a * a - 15; he4 = a ** 4 - 6 * a * a + 3
        P2 = Phi + phi * (t3 / 6 * (a * a - 1) + t4 / 24 * (3 * a - a ** 3) + t3 * t3 / 72 * he5m)
        p2 = d * (1 + t3 / 6 * (3 * a - a ** 3) + t4 / 24 * he4 + t3 * t3 / 72 * he6)
        mg = sigma * phi + mu * Phi
        m2 = mg + t3 * sigma ** 3 * (-a * phi / v) / 6 + t4 * v * v * ((a * a - 1) * phi / (v * sigma)) / 24 + sigma * t3 * t3 / 72 * he4 * phi
        m_true = integrate.quad(lambda z: (mu + sigma * z) * Z.pdf(z), -a, np.inf, limit=200)[0]
        print(f'{k:5.0f} {a:5.1f} | {P2 - Z.sf(-a):+.2e} | {p2 - Z.pdf(-a) / sigma:+.2e} | {m2 - m_true:+.2e}')
