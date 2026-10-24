"""Research-only complete signed CP mode-one Gram, with no label inputs."""
import numpy as np


def full_gram(p, lam):
    h = p.T @ p
    h *= h
    weighted = p * lam[None, :]
    return (weighted @ h) @ weighted.T


def leading_basis(g, rank):
    scale = max(float(np.linalg.norm(g)), np.finfo(np.float64).tiny)
    asymmetry = float(np.linalg.norm(g - g.T)) / scale
    gs = (g + g.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gs)
    q = eigenvectors[:, -rank:][:, ::-1].copy()
    # Fix irrelevant column signs to make the saved basis reproducible.
    pivots = np.argmax(np.abs(q), axis=0)
    q *= np.where(q[pivots, np.arange(rank)] < 0, -1., 1.)[None, :]
    residual = np.linalg.norm(gs @ eigenvectors - eigenvectors * eigenvalues)
    orthogonality = np.linalg.norm(q.T @ q - np.eye(rank)) / np.sqrt(rank)
    metrics = dict(gram_relative_asymmetry=asymmetry,
                   minimum_eigenvalue=float(eigenvalues[0]),
                   maximum_eigenvalue=float(eigenvalues[-1]),
                   eigendecomposition_relative_residual=float(residual / scale),
                   basis_relative_orthogonality=float(orthogonality),
                   one_mode_retained_energy_fraction=float(
                       eigenvalues[-rank:].sum() / eigenvalues.sum())
                   if eigenvalues.sum() else None)
    return q, eigenvalues, metrics


def contract(p, lam, a, b):
    return p @ (lam[:, None] * (p.T @ a) * (p.T @ b))


def plane(p, lam):
    return (p * p) @ (p * lam[None, :]).T


def project(p, q):
    return q @ (q.T @ p)
