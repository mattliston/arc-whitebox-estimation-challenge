"""Fuse exact FULL augmented 211 - 2*m*111 trace into matrix kernels.

Three n^3 products and three n^2*r products before covariance centering.
The CP repeated plane is assumed available from the existing carrier.
Only small synthetic arrays are tested here; not a production estimator.
"""
from itertools import product
from pathlib import Path
import hashlib
import json
import time
import torch
from mlp_kprop.diagslice import zero_repeated, diagslice
from mlp_kprop.harmonic import HTensor, DS_harmonic_proj
from mlp_kprop.partitions import check_vec_partition, vec_part_coef
import mlp_kprop.kprop_harmonic as kp
from certify_augmented_fourth_trace_v2 import TraceFirst, dense_reference, relerr


def repeated_correction(ev, h, w):
    n = ev.n
    ans = torch.zeros((n, n))
    for part, blocks in kp.get_all_terms(3, d_max=4, augment=True):
        if part != (2, 1, 1):
            continue
        ks = check_vec_partition(blocks, 3)
        initial = [h[ks[0]], w[ks[1]], w[ks[2]]]
        for summands in product(*(ev.block_expansion(b) for b in blocks)):
            coef = vec_part_coef(blocks, divide_fac=True)
            nodes = [x.clone() for x in initial]
            edges = {}
            cp = 0
            for c, factors, has_cp in summands:
                coef *= c
                cp += has_cp
                for axes, arr in factors:
                    if len(axes) == 1:
                        nodes[axes[0]] *= arr
                    else:
                        i, j = axes
                        if i > j:
                            i, j, arr = j, i, arr.T
                        edges[i, j] = edges.get((i, j), 1.)*arr
            ones = torch.ones((n, n))
            e01, e02, e12 = (edges.get(pair, ones) for pair in ((0, 1), (0, 2), (1, 2)))
            ti = ev.small_slice((2, 1)) if cp else ones
            tj = ti.T
            v0, v1, v2 = nodes
            ans += coef*((v0*v1)[:, None]*v2[None, :]*e01.diag()[:, None]*e02*e12*ti
                         +v1[:, None]*(v0*v2)[None, :]*e01.T*e02.diag()[None, :]*e12*tj)
    return zero_repeated(ans)


def fused(ev, postmean):
    w = [ev.wick(k, 1) for k in range(7)]
    h = [ev.wick(k, 2)-2*postmean*w[k] for k in range(7)]
    c = ev.cov
    k21 = ev.small_slice((2, 1))
    k22 = ev.small_slice((2, 2))
    heavy22 = k22/4 + c*c/2
    g, d = w[1], w[2]
    # Two-edge paths centered on the spectator and covariance triangle.
    gram = c.T @ (h[2][:, None]*c)                    # nnn 1
    out = gram*(g[:, None]*g[None, :] + c*d[:, None]*d[None, :])
    heavy = (h[3][:, None]*(k21*g[None, :]/2+heavy22*d[None, :])
             +h[2][:, None]*k21.T*d[None, :]/2)
    cross = (heavy.T @ c)*g[None, :]                 # nnn 2
    out += cross+cross.T
    # Paths centered at either untraced output: spectator contractions
    # are matrix-vector products, not matrix-matrix products.
    legs = [(1, 1, c, 1), (2, 1, k21/2, 2),
            (1, 2, k21.T/2, 2), (2, 2, heavy22, 2)]
    centered = torch.zeros_like(c)
    for al, bl, sl, cl in legs:
        projected = sl @ h[bl]
        for ar, br, sr, cr in legs:
            if cl+cr <= 3:
                centered += (projected*w[al+ar])[:, None]*sr*w[br][None, :]
    out += centered+centered.T
    # All-distinct input third cumulant, including its three edge products.
    p, lam = ev.p, ev.lam
    k3trace = (p*((p*h[1][:, None]).sum(0)*lam)) @ p.T       # nnr 1
    out += k3trace*(g[:, None]*g[None, :] + c*d[:, None]*d[None, :])
    edge = c.T @ (h[2][:, None]*p)                         # nnr 2
    k3edge = ((edge*p*lam) @ p.T)*d[:, None]*g[None, :]      # nnr 3
    out += k3edge+k3edge.T
    # Input fourth tensor Sym(A,B), repeated spectator then repeated output.
    a, b = ev.core, ev.metric
    ab = a.T @ (h[2][:, None]*b)                           # nnn 3
    q0 = ((h[2]*a.diag()).sum()*b+(h[2]*b.diag()).sum()*a+2*(ab+ab.T))/12
    out += q0*g[:, None]*g[None, :]
    av, bv = a.T@h[1], b.T@h[1]
    qi = (a.diag()[:, None]*bv[None, :]+b.diag()[:, None]*av[None, :]
          +2*av[:, None]*b+2*bv[:, None]*a)/12
    qi *= d[:, None]*g[None, :]
    out += qi+qi.T
    return zero_repeated(out)-repeated_correction(ev, h, w)


def run(n, seed):
    torch.manual_seed(seed)
    m = torch.randn(n)*.4
    x = torch.randn(n, n)
    c = x@x.T/n + .6*torch.eye(n)
    p = torch.randn(n, n+5)/n**.5
    lam = torch.randn(n+5)*.07
    a = torch.randn(n, n)*.05
    a = (a+a.T)/2
    x = torch.randn(n, n)
    b = x@x.T/n+.3*torch.eye(n)
    tower = {1: HTensor(m, 0), 2: HTensor(c, 0),
             3: HTensor(torch.einsum('ir,jr,kr,r->ijk', p, p, p, lam), 0),
             4: HTensor(a, 1, metric=b)}
    pk, kd = dense_reference(tower, 'full')
    ev = TraceFirst(m, c, p, lam, a, b)
    pm = pk[1].to_tensor()
    f = fused(ev, pm)
    ref = pk[4].get_slice((2, 1, 1)).sum(0)-2*(pk[3].get_slice((1, 1, 1))*pm[:, None, None]).sum(0)
    co = zero_repeated(kd[2].to_tensor())
    k31 = kd[4].get_slice((3, 1))
    r = f-2*co.T@co+k31+k31.T
    r.diagonal().copy_(diagslice(kd[4], (2, 2)).sum(0))
    core = 6*r/(n+4)-3*r.trace()*torch.eye(n)/((n+2)*(n+4))
    core_ref = DS_harmonic_proj(kd[4], r_out=1).core
    row = dict(n=n, seed=seed, fused_partial_trace_relative_error=relerr(f, ref),
               projected_core_relative_error=relerr(core, core_ref))
    row['passed'] = max(row['fused_partial_trace_relative_error'], row['projected_core_relative_error']) < 1e-10
    print(json.dumps(row), flush=True)
    return row


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    out = here/'augmented_fourth_trace_fused_certificate_v1.json'
    assert not out.exists()
    started = time.time()
    rows = [run(n, seed) for n, seed in ((8, 91935), (16, 919100))]
    receipt = dict(rows=rows, passed=all(x['passed'] for x in rows), wall_seconds=time.time()-started,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   per_layer_core_acquisition_products=dict(nnn=4, nnr=3),
                   exclusions='Cost count excludes existing K21/K22 reads, low-order vector/elementwise operations, residual core and metric pushes, and K4-to-K3 source/carry. Not an estimator bill or accuracy result.')
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    assert receipt['passed'], receipt
