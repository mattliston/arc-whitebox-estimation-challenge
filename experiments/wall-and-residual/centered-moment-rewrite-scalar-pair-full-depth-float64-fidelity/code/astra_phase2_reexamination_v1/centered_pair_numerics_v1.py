"""Same full-pair closure in centered arithmetic, preserving input asymmetry."""
import numpy as np
from centered_relu_marginals_v1 import marginals
from radial_mehler import derivatives


def nonlinear(mu, C, K21, k4, k22_override=None, metric='covariance'):
    v = np.diag(C).copy()
    k3 = np.diag(K21).copy()
    k4 = np.broadcast_to(k4, v.shape)
    m, aux = marginals(mu, v, k3, k4)
    s, a, pdf, g, mg = (aux[key] for key in ('sigma', 'alpha', 'pdf', 'gate', 'gaussian_mean'))
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0)
    T21 = K21 / (v[:, None] * s[None, :]); np.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    T22 = K22 / np.outer(v, v)
    t3, t4 = k3 / s**3, k4 / v**2
    T31 = t4[:, None] * rho
    A, _ = derivatives(a, pdf, g, mg/s, 6)
    q = m / s
    # B0 is never used: Gaussian connected terms begin at r=1 and all
    # r=0 Edgeworth terms differentiate each endpoint at least once.
    F = [np.zeros_like(s), 2*(aux['upper_probability']*mg/s-g*aux['mean_shift']/s),
         2*(g-q*pdf)]
    F += [2*(A[r-1]-q*A[r]) for r in range(3, 7)]

    def pair(X, Y, asymmetry=False):
        ans = np.zeros_like(C)
        rr = np.ones_like(C)
        for r in range(7):
            weight = rr-rr.T if asymmetry else rr
            if r:
                ans += weight*np.outer(X[r], Y[r])
            if r <= 2:
                ec = (T21*np.outer(X[r+2], Y[r+1])+T21.T*np.outer(X[r+1], Y[r+2]))/2
                ec += T22*np.outer(X[r+2], Y[r+2])/4
                ec += (T31*np.outer(X[r+3], Y[r+1])+T31.T*np.outer(X[r+1], Y[r+3]))/6
                if r:
                    ec += (np.outer(t3*X[r+3], Y[r])+np.outer(X[r], t3*Y[r+3]))/6
                    ec += (np.outer(t4*X[r+4], Y[r])+np.outer(X[r], t4*Y[r+4]))/24
                ans += weight*ec
                if asymmetry:
                    ans += rr.T*(T22-T22.T)*np.outer(X[r+2], Y[r+2])/4
            rr = rr*rho/(r+1)
        return ans

    cov = pair(A, A)*np.outer(s, s)
    post21 = pair(F, A)*np.outer(v, s)
    post22 = pair(F, F)*np.outer(v, v)-2*cov*cov
    # The original map uses p21.T, not pair(A,B), which differ when its
    # f32-carried covariance is slightly asymmetric. Keep that difference
    # explicitly using small antisymmetric weights, not subtracting means.
    asym = pair(A, F, asymmetry=True)*np.outer(s, v)
    post22 += 2*m[:, None]*asym + 4*np.outer(m, m)*(cov-cov.T)
    np.fill_diagonal(cov, aux['post_v'])
    np.fill_diagonal(post21, aux['post_k3'])
    np.fill_diagonal(post22, aux['post_k4'])
    denominator = np.trace(cov)**2+2*np.sum(cov*cov) if metric == 'covariance' else len(mu)*(len(mu)+2)
    radial = 3*np.sum(post22)/denominator
    aux['pre_cov'] = C
    aux['birth_beta'] = 2*(aux['upper_probability']*mg-g*aux['mean_shift'])
    return m, cov, post21, radial, aux
