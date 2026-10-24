"""Finite atomic law plus analytic chi moments; no network data/targets."""
from pathlib import Path
from itertools import product, combinations
from math import factorial
import hashlib
import json
import resource
import time
import numpy as np
import mpmath as mp

HERE = Path(__file__).resolve().parent
OUT = HERE/'input_radius_recombination_certificate_v1.json'
assert not OUT.exists()
mp.mp.dps = 80


def partitions(items):
    if not items:
        yield []
        return
    first, tail = items[0], items[1:]
    for rest in partitions(tail):
        yield [[first]]+rest
        for j in range(len(rest)):
            yield rest[:j]+[[first]+rest[j]]+rest[j+1:]


def cumulant_tensor(raw, order, dim):
    answer = np.empty((dim,)*order)
    partitions_list = list(partitions(list(range(order))))
    for indices in product(range(dim), repeat=order):
        value = 0.
        for groups in partitions_list:
            term = (-1)**(len(groups)-1)*factorial(len(groups)-1)
            for group in groups:
                term *= raw[len(group)][tuple(indices[j] for j in group)]
            value += term
        answer[indices] = value
    return answer


def raw_tensor(atoms, weights, order):
    answer = np.zeros((atoms.shape[1],)*order)
    for atom, weight in zip(atoms, weights):
        term = atom
        for _ in range(order-1):
            term = np.multiply.outer(term, atom)
        answer += weight*term
    return answer


atoms = np.array([[.2, .3, 1.], [.7, 1.5, .3], [1., .1, .6],
                  [1.4, .4, .8], [.6, 1.1, 1.2], [.3, .2, .5]])
weights = np.arange(1., 7.)/21
raw = {k: raw_tensor(atoms, weights, k) for k in range(1, 5)}
mu, C, T, Q = [cumulant_tensor(raw, k, 3) for k in range(1, 5)]
records = []
start = time.time()
for n in (16, 1024):
    c_exact = mp.sqrt(mp.mpf(2)/n)*mp.gamma(mp.mpf(n+1)/2)/mp.gamma(mp.mpf(n)/2)
    d_exact = 1-c_exact*c_exact
    c, d = float(c_exact), float(d_exact)
    a3, a4 = float(c_exact*(1+mp.mpf(1)/n)), float(1+mp.mpf(2)/n)
    beta = float(c_exact*(mp.mpf(1)/n-2*d_exact))
    k4s = float(4*d_exact-mp.mpf(2)/n+4*d_exact/n-6*d_exact*d_exact)
    c_mut = float(1+mp.mpf(2)/n-c_exact*c_exact*(1+mp.mpf(1)/n))
    factors = {1: c, 2: 1., 3: a3, 4: a4}
    gaussian_raw = {k: factors[k]*raw[k] for k in raw}
    actual = {k: cumulant_tensor(gaussian_raw, k, 3) for k in raw}
    expected = {1: c*mu, 2: C+d*np.outer(mu, mu),
                3: np.empty((3,)*3), 4: np.empty((3,)*4)}
    for i, j, k in product(range(3), repeat=3):
        expected[3][i, j, k] = (a3*T[i, j, k]+c/n*(mu[i]*C[j, k]+mu[j]*C[i, k]+mu[k]*C[i, j])
                                 +beta*mu[i]*mu[j]*mu[k])
    for ix in product(range(3), repeat=4):
        i, j, k, l = ix
        cc = C[i, j]*C[k, l]+C[i, k]*C[j, l]+C[i, l]*C[j, k]
        mut = sum(mu[ix[pos]]*T[tuple(ix[j] for j in range(4) if j != pos)] for pos in range(4))
        mumuc = 0.
        for left in combinations(range(4), 2):
            right = [j for j in range(4) if j not in left]
            mumuc += mu[ix[left[0]]]*mu[ix[left[1]]]*C[ix[right[0]], ix[right[1]]]
        expected[4][ix] = a4*Q[ix]+2/n*cc+c_mut*mut+2*d/n*mumuc+k4s*np.prod(mu[list(ix)])
    errors = {str(k): float(np.max(np.abs(actual[k]-expected[k]))) for k in actual}
    inverse_error = max(float(np.max(np.abs(gaussian_raw[k]/factors[k]-raw[k]))) for k in raw)
    records.append(dict(n=n, c=c, radial_variance=d, a3=a3, a4=a4,
                        third_radius_rank_one_coefficient=beta,
                        fourth_radius_rank_one_coefficient=k4s,
                        fourth_mu_T_coefficient=c_mut,
                        max_absolute_errors=errors, inverse_raw_error=inverse_error,
                        passed=max(errors.values()) <= 1e-12 and inverse_error <= 1e-12))
result = dict(records=records, passed=all(r['passed'] for r in records),
              wall_s=time.time()-start,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Six fixed positive atoms in three output dimensions, exact finite raw moments, independent set-partition cumulants, 80-digit chi-radius scalar coefficients. No network weights, targets, Monte Carlo, closure run, inference cost measurement, fit or score.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
