"""Offline exact two-high fourth-sector algebra; no estimator implementation."""
import itertools
import math
import numpy as np


def symmetric_high(array, axes):
    order = tuple(range(array.ndim))
    result = np.zeros_like(array)
    for permutation in itertools.permutations(axes):
        target = list(order)
        for axis, replacement in zip(axes, permutation):
            target[axis] = replacement
        result += array.transpose(target)
    return result / math.factorial(len(axes))


def radial_slices(C, Q, radial):
    CQ = C @ Q
    G = Q.T @ CQ
    return radial / 3 * (G[:, :, None, None] * C[None, None, :, :]
        + np.einsum('ia,jb->abij', CQ, CQ)
        + np.einsum('ib,ja->abij', CQ, CQ))


def batch_slice_statistic(S, G, H, raw, count):
    """S=E[hh'], G=E[zz'], H=E[zh']; raw_abij=E[z_a z_b h_i h_j]."""
    wick = G[:, :, None, None] * S[None, None, :, :]
    wick += np.einsum('ai,bj->abij', H, H)
    wick += np.einsum('aj,bi->abij', H, H)
    factor = count * count / ((count-1) * (count-2) * (count-3))
    return factor * ((count+1) * raw - (count-1) * wick)


def pushed_planes(Q, U, W):
    """Exact planes of the >=2-high sector from q^2 unrestricted matrices."""
    q, n = Q.shape[1], W.shape[1]
    X = W.T @ Q
    V = symmetric_high(np.einsum('abij,ic->abcj', U, Q, optimize=True), (0, 1, 2))
    Z = symmetric_high(np.einsum('abcj,jd->abcd', V, Q, optimize=True), (0, 1, 2, 3))
    Vp = np.einsum('abci,ij->abcj', V, W, optimize=True)
    Up = np.empty((q, q, n, n))
    for a in range(q):
        for b in range(a, q):
            matrix = W.T @ U[a, b] @ W
            matrix = (matrix + matrix.T) / 2
            Up[a, b] = Up[b, a] = matrix
    diagonal = np.diagonal(Up, axis1=2, axis2=3)
    pair = np.einsum('ia,ib->iab', X, X).reshape(n, q*q)
    cube = np.einsum('ia,ib,ic->iabc', X, X, X).reshape(n, q*q*q)
    aa_bb = pair @ diagonal.reshape(q*q, n)
    q22 = aa_bb + aa_bb.T
    q31 = 3 * np.einsum('ia,jb,abi->ij', X, X, diagonal, optimize=True)
    for a in range(q):
        for b in range(q):
            q22 += 4 * X[:, a, None] * X[None, :, b] * Up[a, b]
            q31 += 3 * (X[:, a] * X[:, b])[:, None] * Up[a, b]
    triple_one = np.einsum('ic,abci->iab', X, Vp, optimize=True).reshape(n, q*q)
    cross22 = pair @ triple_one.T
    q22 -= 4 * (cross22 + cross22.T)
    q31 -= 2 * cube @ Vp.reshape(q*q*q, n)
    triple_two = np.einsum('ia,ib,abci->ic', X, X, Vp, optimize=True)
    q31 -= 6 * triple_two @ X.T
    q22 += 3 * pair @ Z.reshape(q*q, q*q) @ pair.T
    q31 += 3 * cube @ Z.reshape(q*q*q, q) @ X.T
    k4 = np.diag(q22).copy()
    return q22, q31, k4


def small_projection(T, Q):
    """Explicit dense definition for certificates only."""
    P = Q @ Q.T
    result = np.zeros_like(T)
    for count, coefficient in ((2, 1), (3, -2), (4, 3)):
        for subset in itertools.combinations(range(4), count):
            term = T
            for axis in subset:
                term = np.moveaxis(np.tensordot(P, term, axes=(1, axis)), 0, axis)
            result += coefficient * term
    return result
