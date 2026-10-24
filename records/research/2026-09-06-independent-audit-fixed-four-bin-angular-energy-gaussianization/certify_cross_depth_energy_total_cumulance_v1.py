"""Tiny target-free algebra only; no estimator, target, Torch, or data imports.

Companion to CROSS_DEPTH_ANGULAR_ENERGY_INFORMATION_PROTOCOL_V1.md.
Execution prints a receipt; it does not create or replace any file.
"""
import json
import math
from itertools import combinations

import numpy as np


def t3(z):
    return np.einsum('ni,nj,nk->ijk', z, z, z) / len(z)


def t4(z):
    return np.einsum('ni,nj,nk,nl->ijkl', z, z, z, z) / len(z)


def wick(c):
    return (np.einsum('ij,kl->ijkl', c, c)
            + np.einsum('ik,jl->ijkl', c, c)
            + np.einsum('il,jk->ijkl', c, c))


def sym_dc(d, c):
    return (np.einsum('i,jk->ijk', d, c)
            + np.einsum('j,ik->ijk', d, c)
            + np.einsum('k,ij->ijk', d, c))


def sym_dt(d, third):
    return (np.einsum('i,jkl->ijkl', d, third)
            + np.einsum('j,ikl->ijkl', d, third)
            + np.einsum('k,ijl->ijkl', d, third)
            + np.einsum('l,ijk->ijkl', d, third))


def sym_ddc(d, c):
    result = np.zeros((len(d),) * 4)
    for slots in combinations(range(4), 2):
        rest = tuple(j for j in range(4) if j not in slots)
        for ix in np.ndindex(result.shape):
            result[ix] += d[ix[slots[0]]] * d[ix[slots[1]]] * c[ix[rest[0]], ix[rest[1]]]
    return result


def relative_max(a, b):
    return float(np.max(np.abs(a-b)) / max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b)))))


def main():
    rng = np.random.Generator(np.random.PCG64DXSM(90620261931))
    counts = np.array([5, 7, 11, 13])
    labels = np.repeat(np.arange(4), counts)
    x = rng.standard_normal((int(counts.sum()), 4))
    x += .13 * x*x + np.array([.8, -.3, .2, .4])[None, :] * labels[:, None]
    mu = x.mean(0)
    z = x-mu
    c = z.T@z / len(z)
    third = t3(z)
    fourth = t4(z)-wick(c)
    c_parts = np.zeros_like(c)
    third_parts = np.zeros_like(third)
    gaussian_third = np.zeros_like(third)
    within_third = np.zeros_like(third)
    fourth_raw_parts = np.zeros_like(fourth)
    means, covs, thirds = [], [], []
    scalar_raw = np.zeros(4)
    for b, count in enumerate(counts):
        p = count/len(x)
        xb = x[labels == b]
        mb = xb.mean(0)
        zb = xb-mb
        cb = zb.T@zb / count
        tb = t3(zb)
        qb = t4(zb)-wick(cb)
        d = mb-mu
        gtb = sym_dc(d, cb)+np.einsum('i,j,k->ijk', d, d, d)
        c_parts += p*(cb+np.outer(d,d))
        third_parts += p*(tb+gtb)
        gaussian_third += p*gtb
        within_third += p*tb
        fourth_raw_parts += p*(qb+wick(cb)+sym_dt(d,tb)+sym_ddc(d,cb)
                               + np.einsum('i,j,k,l->ijkl', d,d,d,d))
        means.append(mb)
        covs.append(cb)
        thirds.append(np.einsum('iii->i', tb))
        shift = np.array([.7, -.1, 1.2, -.8])
        y = xb-shift
        ybar = y.mean(0)
        m2 = np.mean(y*y, 0)-ybar*ybar
        m3 = np.mean(y*y*y, 0)-3*ybar*np.mean(y*y, 0)+2*ybar**3
        scalar_raw += p*(m3+3*d*m2+d**3)

    errors = dict(
        total_covariance=relative_max(c, c_parts),
        total_third=relative_max(third, third_parts),
        gaussianization_discarded_third=relative_max(third-gaussian_third, within_third),
        total_fourth=relative_max(fourth, fourth_raw_parts-wick(c)),
        scalar_shifted_raw_identity=relative_max(np.einsum('iii->i', third), scalar_raw),
    )

    # Every component-mean difference is perpendicular to this null direction.
    # Untied covariance does not create a three-transverse-leg Gaussian T3.
    dmat = np.array(means)-mu
    _, _, vh = np.linalg.svd(dmat, full_matrices=True)
    u = vh[-1]
    errors['gaussian_three_transverse_legs'] = abs(float(np.einsum('ijk,i,j,k', gaussian_third,u,u,u)))
    errors['component_mean_null'] = float(np.max(np.abs(dmat@u)))

    pa = np.array([.16,.24,.28,.32])
    pb = np.array([.19,.20,.30,.31])
    ka = np.array(thirds)
    kb = ka + .07*np.arange(16).reshape(4,4)/16
    a = np.array([.2,-.7,.3,.9])
    oa = a*np.einsum('b,bi->i',pa,ka)
    ob = a*np.einsum('b,bi->i',pb,kb)
    membership = a*np.einsum('b,bi->i',pa-pb,(ka+kb)/2)
    within = a*np.einsum('b,bi->i',(pa+pb)/2,ka-kb)
    errors['replica_membership_decomposition'] = relative_max(oa-ob,membership+within)
    cross = float(np.mean(oa*ob))
    pooled = float(.5*(np.mean(oa*oa)+np.mean(ob*ob)))
    noise = float(.5*np.mean((oa-ob)**2))
    errors['cross_energy_identity'] = abs(cross-(pooled-noise))

    # A_l(scale*x) = A_l(x) for an explicitly bias-free tiny ReLU network.
    h = rng.standard_normal((11,4))
    scale = np.linspace(.2,2.2,len(h))[:,None]
    hs = h*scale
    norm = np.sum(h*h,axis=1)
    norms = np.sum(hs*hs,axis=1)
    energy_error = 0.
    for _ in range(3):
        w = rng.standard_normal((4,4))/2
        h = np.maximum(h@w,0)
        hs = np.maximum(hs@w,0)
        energy_error = max(energy_error,relative_max(4*np.sum(h*h,axis=1)/norm,
                                                    4*np.sum(hs*hs,axis=1)/norms))
    errors['angular_energy_homogeneity'] = energy_error
    assert max(errors.values()) < 2e-12, errors
    print(json.dumps(dict(passed=True, maximum_error=max(errors.values()), errors=errors,
                          target_access=False, torch_imported=False,
                          qualification='Finite-atom algebra only; no accuracy, statistical bias, '
                          'sampling fidelity, conditional-state source, or deployment claim.'), indent=2))


if __name__ == '__main__':
    main()
