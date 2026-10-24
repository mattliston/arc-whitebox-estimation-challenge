"""Small-width certificate: sum the spectator BEFORE constructing a 211 tensor.

The trace-first path constructs only vectors/matrices and CP factors. Dense
tensors appear solely in the width-8/16 independent primary-code reference.
No weights, labels, estimator execution, or wide tensors are used.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import time

import torch
from mlp_kprop.cumulants import DS_pK_to_K
from mlp_kprop.diagslice import DSTower, diagslice, eval_part
from mlp_kprop.harmonic import HTensor, DS_harmonic_proj
import mlp_kprop.kprop_harmonic as kp
from mlp_kprop.partitions import check_vec_partition, vec_part_coef
from mlp_kprop.tensor_utils import symmetrize, zero_repeated
from mlp_kprop.wick import relu_wick_coef


class TraceFirst:
    def __init__(self, mean, cov, p, lam, core, metric):
        self.mean, self.cov = mean, cov
        self.p, self.lam = p, lam
        self.core, self.metric = core, metric
        self.n = len(mean)
        self.counts = Counter()

    @lru_cache(None)
    def wick(self, k, power):
        return relu_wick_coef(self.mean, self.cov.diag(), k, power)

    @lru_cache(None)
    def small_slice(self, vec):
        degree = sum(vec)
        assert 1 <= len(vec) <= 2
        if degree == 1:
            return self.mean
        if degree == 2:
            return self.cov.diag() if len(vec) == 1 else self.cov
        if degree == 3:
            if len(vec) == 1:
                return (self.p ** 3 * self.lam).sum(1)
            self.counts['cached_k3_plane_nnr'] += 1
            return (self.p ** vec[0] * self.lam) @ (self.p ** vec[1]).T
        assert degree == 4
        a, b = self.core, self.metric
        if len(vec) == 1:
            return a.diag() * b.diag()
        if vec == (2, 2):
            return (a.diag()[:, None]*b.diag()[None, :]
                    + b.diag()[:, None]*a.diag()[None, :] + 4*a*b)/6
        if vec == (3, 1):
            return (a.diag()[:, None]*b + b.diag()[:, None]*a)/2
        assert vec == (1, 3)
        return self.small_slice((3, 1)).T

    def block_expansion(self, block):
        """List of (coefficient, [(axes, array)], cp_present) summands."""
        axes = tuple(i for i, x in enumerate(block) if x)
        vec = tuple(x for x in block if x)
        if len(axes) <= 2:
            return [(1., [(axes, self.small_slice(vec))], False)]
        if sum(block) == 3:
            assert block == (1, 1, 1)
            return [(1., [], True)]
        assert sum(block) == 4
        legs = tuple(i for i, x in enumerate(block) for _ in range(x))
        ans = []
        # Six choices for the two core legs, including their multiplicities.
        for chosen in combinations(range(4), 2):
            other = tuple(i for i in range(4) if i not in chosen)
            factors = []
            for matrix, pair in ((self.core, chosen), (self.metric, other)):
                i, j = legs[pair[0]], legs[pair[1]]
                factors.append(((i,), matrix.diag()) if i == j else
                               ((i, j), matrix))
            ans.append((1/6, factors, False))
        return ans

    def contract(self, factors, cp, nodes):
        """Trace axis0 and remove a=i/a=j; at most two-index arrays here."""
        n = self.n
        nodes = [v.clone() for v in nodes]
        edges = {}
        for axes, arr in factors:
            if len(axes) == 1:
                nodes[axes[0]] *= arr
            else:
                i, j = axes
                if i > j:
                    i, j, arr = j, i, arr.T
                edges[(i, j)] = edges.get((i, j), 1.) * arr
        ones = torch.ones((n, n))
        e01 = edges.get((0, 1), ones)
        e02 = edges.get((0, 2), ones)
        e12 = edges.get((1, 2), ones)
        v0, v1, v2 = nodes
        if not cp:
            if (0, 1) not in edges:
                summed = (v0[:, None]*e02).sum(0)[None, :].expand(n, n)
            elif (0, 2) not in edges:
                summed = (v0[:, None]*e01).sum(0)[:, None].expand(n, n)
            else:
                self.counts['pair_trace_nnn'] += 1
                summed = e01.T @ (v0[:, None]*e02)
            ti = torch.ones((n, n))
            tj = ti
        else:
            assert not ((0, 1) in edges and (0, 2) in edges)
            p, lam = self.p, self.lam
            if (0, 1) in edges:
                self.counts['cp_edge_nnr'] += 1
                left = (e01.T @ (v0[:, None]*p)) * p * lam
                self.counts['cp_trace_nnr'] += 1
                summed = left @ p.T
            elif (0, 2) in edges:
                self.counts['cp_edge_nnr'] += 1
                right = (e02.T @ (v0[:, None]*p)) * p * lam
                self.counts['cp_trace_nnr'] += 1
                summed = p @ right.T
            else:
                weights = (v0[:, None]*p).sum(0) * lam
                self.counts['cp_trace_nnr'] += 1
                summed = (p*weights) @ p.T
            ti = self.small_slice((2, 1))
            tj = ti.T
        out = summed * e12 * v1[:, None]*v2[None, :]
        # Inclusion/exclusion only for i!=j (we zero the diagonal afterwards).
        # a=i and a=j are disjoint on these output entries.
        at_i = (v0*v1)[:, None]*v2[None, :] * e01.diag()[:, None]*e02*e12*ti
        at_j = v1[:, None]*(v0*v2)[None, :] * e01.T*e02.diag()[None, :]*e12*tj
        return zero_repeated(out-at_i-at_j)

    def slice_trace(self, output, mode, spectator=None):
        ans = torch.zeros((self.n, self.n))
        for intpart, blocks in kp.get_all_terms(3, d_max=4, augment=True):
            if intpart != output:
                continue
            if mode == 'factorized' and not kp.factored_keeps_term(3, output, blocks):
                continue
            ks = check_vec_partition(blocks, 3)
            nodes = [self.wick(k, p) for k, p in zip(ks, output)]
            if spectator is not None:
                nodes[0] = nodes[0] * spectator
            coef = vec_part_coef(blocks, divide_fac=True)
            for factorset in product(*(self.block_expansion(b) for b in blocks)):
                c = coef
                factors = []
                cp_count = 0
                for bcoef, bfactors, bcp in factorset:
                    c *= bcoef
                    factors.extend(bfactors)
                    cp_count += bcp
                assert cp_count <= 1
                ans += c*self.contract(factors, bool(cp_count), nodes)
        return ans


def dense_reference(tower, mode):
    mean, var = tower[1].core, tower[2].core.diag()
    @lru_cache(None)
    def wick(k, p):
        return relu_wick_coef(mean, var, k, p)
    slices = {}
    for part, terms in kp.get_all_terms_iso(3, d_max=4, augment=True).items():
        acc = torch.zeros((len(mean),)*len(part))
        for blocks, count in terms.items():
            if mode == 'factorized' and not kp.factored_keeps_term(3, part, blocks):
                continue
            term = eval_part(tower, blocks, len(part), output_zero_repeated=True)
            acc += count*kp.multiply_wicks(term, check_vec_partition(blocks, len(part)), part, wick)
        slices[part] = symmetrize(acc, vec=part)
    pk = DSTower.from_slices(slices, autozero=True)
    return pk, DS_pK_to_K(pk)


def relerr(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))


def run(n, seed, mode):
    torch.manual_seed(seed)
    mean = torch.randn(n)*.45
    x = torch.randn(n, n)
    cov = x@x.T/n + .5*torch.eye(n)
    p = torch.randn(n, n+3)/n**.5
    lam = torch.randn(n+3)*.04
    core = torch.randn(n, n)*.03
    core = (core+core.T)/2
    x = torch.randn(n, n)
    metric = x@x.T/n + .2*torch.eye(n)
    k3 = torch.einsum('ir,jr,kr,r->ijk', p, p, p, lam)
    tower = {1: HTensor(mean, 0), 2: HTensor(cov, 0),
             3: HTensor(k3, 0), 4: HTensor(core, 1, metric=metric)}
    pk, kd = dense_reference(tower, mode)
    ev = TraceFirst(mean, cov, p, lam, core, metric)
    postmean = pk[1].to_tensor()
    t211 = ev.slice_trace((2, 1, 1), mode)
    t111 = ev.slice_trace((1, 1, 1), mode, spectator=postmean)
    ref211 = pk[4].get_slice((2, 1, 1)).sum(0)
    ref111 = (pk[3].get_slice((1, 1, 1))*postmean[:, None, None]).sum(0)
    postcov = kd[2].to_tensor()
    c_off = zero_repeated(postcov)
    k31 = kd[4].get_slice((3, 1))
    trace = t211 - 2*t111 - 2*(c_off.T@c_off) + k31 + k31.T
    trace.diagonal().copy_(diagslice(kd[4], (2, 2)).sum(0))
    dense4 = kd[4].to_tensor()
    reftrace = torch.einsum('aaij->ij', dense4)
    core_from_trace = 6*trace/(n+4) - 3*trace.trace()*torch.eye(n)/((n+2)*(n+4))
    refproj = DS_harmonic_proj(kd[4], r_out=1).core
    reformed = HTensor(core_from_trace, 1).to_tensor()
    reconstructed_trace = torch.einsum('aaij->ij', reformed)
    row = dict(n=n, seed=seed, mode=mode,
        pk211_trace_relative_error=relerr(t211, ref211),
        weighted_pk111_trace_relative_error=relerr(t111, ref111),
        complete_fourth_trace_relative_error=relerr(trace, reftrace),
        harmonic_core_relative_error=relerr(core_from_trace, refproj),
        reconstructed_trace_relative_error=relerr(reconstructed_trace, reftrace),
        unoptimized_operation_counts=dict(ev.counts))
    row['passed'] = max(v for k, v in row.items() if k.endswith('_relative_error')) < 1e-10
    print(json.dumps(row), flush=True)
    return row


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    out = here/'augmented_fourth_trace_certificate_v1.json'
    assert not out.exists(), out
    started = time.time()
    rows = [run(n, seed, mode) for n, seed in ((8, 81835), (16, 818100))
            for mode in ('full', 'factorized')]
    receipt = dict(rows=rows, passed=all(x['passed'] for x in rows),
                   wall_seconds=time.time()-started,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   qualification='Small synthetic CPU float64 algebra certificate only; no candidate accuracy or cost claim.')
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    assert receipt['passed'], receipt
