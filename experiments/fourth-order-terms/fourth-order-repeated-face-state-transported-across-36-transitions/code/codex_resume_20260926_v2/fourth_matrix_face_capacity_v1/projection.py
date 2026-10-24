"""Fixed linear restricted-fourth-face projection; no data or label access."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def faces(R, D):
    a = np.diag(D)
    return a.copy(), (D + a[:, None]*R)/2, (a[:, None]+a[None, :]+4*R*D)/6


def vector(k4, q31, q22):
    n = len(k4)
    i, j = np.triu_indices(n, 1)
    return np.concatenate((k4, 2*q31[i, j], 2*q31[j, i], np.sqrt(6)*q22[i, j]))


def project(R, k4, q31, q22):
    n = len(R)
    assert R.shape == (n, n) and np.max(np.abs(R-R.T)) < 1e-12
    assert np.max(np.abs(np.diag(R)-1)) < 1e-12
    assert np.max(np.abs(q22-q22.T)) < 1e-12
    i, j = np.triu_indices(n, 1)
    r = R[i, j]
    A, B, C = 2*q31[i, j], 2*q31[j, i], 6*q22[i, j]
    h = 2+(8/3)*r*r
    u = (5/3)*r
    rx = A+B+(2/3)*r*C
    d = r*r+1/6-u*u/h
    e = 1/6-u*u/h
    ra = r*A+C/6-u*rx/h
    rb = r*B+C/6-u*rx/h
    M = np.zeros((n, n), dtype=np.float64)
    M[i, j] = e
    M[j, i] = e
    np.fill_diagonal(M, 1+np.bincount(i, weights=d, minlength=n)
                        +np.bincount(j, weights=d, minlength=n))
    rhs = k4+np.bincount(i, weights=ra, minlength=n)+np.bincount(j, weights=rb, minlength=n)
    factor = cho_factor(M, lower=True, check_finite=True)
    a = cho_solve(factor, rhs, check_finite=True)
    x = (rx-u*(a[i]+a[j]))/h
    D = np.diag(a)
    D[i, j] = x
    D[j, i] = x
    residual = M@a-rhs
    info = dict(solve_relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), 1e-300)),
                cholesky_diagonal_min=float(np.diag(factor[0]).min()),
                eliminated_offdiagonal_denominator_min=float(h.min()),
                diagonal_unknowns=n, offdiagonal_unknowns=len(i))
    assert info['solve_relative_residual'] < 1e-10
    return D, info
