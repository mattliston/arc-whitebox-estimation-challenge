"""FP64 research contractions; no estimator or target access."""
import math
import numpy as np


class Products:
    def __init__(self):
        self.records = []

    def mm(self, a, b, tag):
        assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
        self.records.append(dict(tag=tag, shapes=[list(a.shape), list(b.shape)],
                                 fp64_leading_billed_flops=4*int(a.shape[0])*int(a.shape[1])*int(b.shape[1])))
        return a @ b


def merge(q, f, lam, direction, products):
    u, r = np.linalg.qr(q, mode='reduced')
    rank = q.shape[1]
    assert q.shape[0] >= rank and np.linalg.matrix_rank(r) == rank
    a = products.mm(r, f, 'construction_change_coordinates')
    z = (u.T @ direction)
    length = float(np.linalg.norm(z))
    assert np.isfinite(length) and length > 1e-12
    z = z / length
    weights = lam * (a.T @ z)
    sliced = products.mm(a * weights[None, :], a.T, 'construction_slice')
    snorm = float(np.linalg.norm(sliced))
    symmetry = float(np.linalg.norm(sliced-sliced.T)/max(snorm,1e-300))
    assert symmetry <= 1e-10
    sliced = (sliced+sliced.T)/2
    eigenvalues, v = np.linalg.eigh(sliced)
    coordinates = products.mm(v.T, a, 'construction_atom_queries')
    coefficient = np.sum(coordinates*coordinates*coordinates*lam[None,:],axis=1)
    basis = products.mm(u, v, 'construction_physical_basis')
    checks = dict(qr_reconstruction=float(np.linalg.norm(products.mm(u,r,'qualification_qr')-q)/max(np.linalg.norm(q),1e-300)),
                  orthogonality=float(np.linalg.norm(products.mm(basis.T,basis,'qualification_orthogonality')-np.eye(rank))/np.sqrt(rank)),
                  slice_symmetry=symmetry,
                  eigen_residual=float(np.linalg.norm(products.mm(sliced,v,'qualification_eigensystem')-v*eigenvalues[None,:])/max(snorm,1e-300)))
    assert all(np.isfinite(value) and value <= 1e-10 for value in checks.values())
    assert all(np.isfinite(x).all() for x in (basis,coefficient,a,eigenvalues))
    return basis, coefficient, a, u, eigenvalues, checks


def contract(p, lam, a, b, products, tag):
    aa = products.mm(p.T, a, tag+'_leg_a')
    bb = products.mm(p.T, b, tag+'_leg_b')
    return products.mm(p, lam[:,None]*aa*bb, tag+'_read')


def plane(p, lam, products, tag):
    return products.mm(p*p, (p*lam[None,:]).T, tag)


def norm_squared(p, lam, products, block=256):
    # All signed cross terms, bounded memory, no stored full Gram.
    totals, absolute = [], []
    for start in range(0, p.shape[1], block):
        stop = min(start+block,p.shape[1])
        g = products.mm(p[:,start:stop].T,p,'diagnostic_norm_gram')
        values = (g*g*g)*lam[start:stop,None]*lam[None,:]
        totals.append(float(np.sum(values,dtype=np.float64)))
        absolute.append(float(np.sum(np.abs(values),dtype=np.float64)))
    return math.fsum(totals), math.fsum(absolute)
