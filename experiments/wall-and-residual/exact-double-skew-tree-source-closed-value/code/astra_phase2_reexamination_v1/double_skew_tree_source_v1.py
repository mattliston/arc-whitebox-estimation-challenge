"""Exact twelve distinct two-skew tree diagrams, offline NumPy reference."""
import numpy as np


def factors(K, g, d, alive=None):
    A = .5*d[:, None]*K
    B = .5*g[:, None]*K.T
    np.fill_diagonal(A, 0)
    np.fill_diagonal(B, 0)
    if alive is not None:
        mask = alive[:, None]*alive[None, :]
        A *= mask
        B *= mask
    return A, B


def dense(K, g, d, e, h, alive=None):
    A, B = factors(K, g, d, alive)
    eye = np.eye(len(K))
    T = np.einsum('ia,ja,ka,a->ijk', A, A, eye, d)
    T += np.einsum('ia,ja,ka,a->ijk', A, B, eye, e)
    T += np.einsum('ia,ja,ka,a->ijk', B, A, eye, e)
    T += np.einsum('ia,ja,ka,a->ijk', B, B, eye, h)
    T = T + T.transpose(0, 2, 1) + T.transpose(2, 1, 0)
    i, j, k = np.indices(T.shape)
    return T*((i != j) & (i != k) & (j != k))


def plane(K, g, d, e, h, W, alive=None):
    A, B = factors(K, g, d, alive)
    Z = W.T
    X, Y = Z@A, Z@B
    result = (2*Z*X*d)@X.T + (X*X*d)@Z.T
    result += (2*Z*X*e)@Y.T + (2*Z*Y*e)@X.T + (2*X*Y*e)@Z.T
    result += (2*Z*Y*h)@Y.T + (Y*Y*h)@Z.T
    # Coordinate repeats of the unmasked triple are represented exactly by
    # Sym3(star[a,b] e_a e_b e_b)/3; all three-input-distinct terms remain.
    repeated = A*A*d + 2*A*B*e + B*B*h
    star = 3*repeated.T
    SZ = Z@star
    result -= ((2*SZ*Z)@Z.T + (Z*Z)@SZ.T)/3
    return result
