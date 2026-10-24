"""Independent formal cumulant certificate and actual helper/map fixtures."""
import ast
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import json
from math import comb, factorial
from pathlib import Path
import types

import numpy as np
from math import erf, sqrt

def ndtr(value):
    return np.vectorize(lambda x: .5*(1+erf(float(x)/sqrt(2))), otypes=[float])(value)

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(*args):
    out = {}
    for a in args:
        for k, v in a.items():
            out[k] = out.get(k, Q(0)) + v
    return {k: v for k, v in out.items() if v}


def scale(a, c):
    return {k: v*c for k, v in a.items() if v*c}


def mul(a, b):
    out = {}
    for (ar, ac), av in a.items():
        for (br, bc), bv in b.items():
            if ar+br <= 2 and ac+bc <= 2:
                k = (ar+br, ac+bc)
                out[k] = out.get(k, Q(0)) + av*bv
    return {k: v for k, v in out.items() if v}


@lru_cache(None)
def moment(i, j):
    # Exact moment-cumulant recurrence. No derivative allocations are used.
    if i == 0 and j == 0:
        return {(0, 0): Q(1)}
    if i == 0:
        return moment(j, 0)
    out = {}
    if i >= 2:
        out = add(out, scale(moment(i-2, j), i-1))
    if j:
        out = add(out, scale(mul({(0, 1): Q(1)}, moment(i-1, j-1)), j))
    for a in range(4):
        b = 3-a
        if a > i-1 or b > j:
            continue
        if a+1 == 4 or b == 4:
            cumulant = {(1, 0): Q(1)}
        elif a+1 == 2:
            cumulant = {(1, 0): Q(1, 3), (1, 2): Q(2, 3)}
        else:
            cumulant = {(1, 1): Q(1)}
        out = add(out, scale(mul(cumulant, moment(i-1-a, j-b)),
                             comb(i-1, a)*comb(j, b)))
    return out


def normal_moment(k):
    return 0 if k % 2 else factorial(k)//(2**(k//2)*factorial(k//2))


def derivative_expectation(coefficients, order):
    return sum(c*Q(factorial(i), factorial(i-order))*normal_moment(i-order)
               for i, c in enumerate(coefficients) if i >= order)


def cov44_reference(f, g):
    out = {}
    for i, fi in enumerate(f):
        for j, gj in enumerate(g):
            centered = add(moment(i,j), scale(mul(moment(i,0), moment(0,j)), -1))
            out = add(out, scale(centered, fi*gj))
    return {c: v for (r,c),v in out.items() if r == 2}


def cov44_derivative(f, g):
    F = [derivative_expectation(f,k) for k in range(11)]
    G = [derivative_expectation(g,k) for k in range(11)]
    out = {}
    # Independent multinomial enumeration of Delta_C^4.
    for a in range(5):
        for b in range(5-a):
            c = 4-a-b
            x = 2*a+b
            coefficient = Q(factorial(4)*2**b, factorial(a)*factorial(b)*factorial(c))
            for t in range(max(0, 3-b)):
                if b == 0 and t == 0 and x in (0,8):
                    continue
                degree = b+t
                value = coefficient*F[x+t]*G[8-x+t]/factorial(t)/1152
                out[degree] = out.get(degree,Q(0)) + value
    out[0] = out.get(0,Q(0)) - F[4]*G[4]/576
    return {k:v for k,v in out.items() if v}


def load_functions(path, include):
    source = path.read_text()
    nodes = {n.name:n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    ns = dict(fnp=np, flops=types.SimpleNamespace(stats=types.SimpleNamespace(
        norm=types.SimpleNamespace(cdf=ndtr))))
    exec('\n\n'.join(ast.get_source_segment(source,nodes[name]) for name in include), ns)
    return ns


def main():
    out = HERE / 'unit_v2'
    out.mkdir(exist_ok=False)
    build = json.loads((HERE/'build.json').read_text())
    assert sha(HERE/'candidate.py') == build['candidate_sha256']
    names = ['gaussian_raw','derivatives','_pair_series_fused','_r44_marginal',
             '_r44_pair','_nonlinear_pair_batch','nonlinear_scalar']
    ns = load_functions(HERE/'candidate.py',names)
    parent = load_functions(HERE.parent/'weight5_reuse_v1/candidate.py',
                            [x for x in names if not x.startswith('_r44')])
    polys = [[Q(1)], [Q(0),Q(1)], [Q(0),Q(0),Q(1)],
             [Q((-1)**k,k+1) for k in range(9)],
             [Q(k%3-1,2**k) for k in range(11)]]
    formal_cases = 0
    for f in polys:
        for g in polys:
            assert cov44_reference(f,g) == cov44_derivative(f,g)
            formal_cases += 1
    n = len(polys)
    F = np.array([[float(derivative_expectation(f,k)) for f in polys]
                  for k in range(11)], dtype=np.float32)[None]
    rho = np.array([[((i+j)%5-2)/10 for j in range(n)] for i in range(n)],dtype=np.float32)
    radial = np.full(n,.05,dtype=np.float32)
    observed = ns['_r44_pair'](F,F,rho,radial)[0]
    reference = np.array([[sum(float(v)*float(rho[i,j])**k
                        for k,v in cov44_reference(f,g).items())*float(radial[0])**2
                        for j,g in enumerate(polys)] for i,f in enumerate(polys)])
    relative = float(np.linalg.norm(observed-reference)/np.linalg.norm(reference))
    symmetry = float(np.max(np.abs(observed-observed.T)) / np.max(np.abs(reference)))
    assert relative <= 2e-5 and symmetry <= 2e-5
    assert np.array_equal(ns['_r44_pair'](F,F,rho,np.zeros(n,np.float32)), np.zeros((1,n,n),np.float32))
    assert np.array_equal(observed[0],np.zeros(n,np.float32))
    assert np.array_equal(observed[:,0],np.zeros(n,np.float32))
    assert np.array_equal(ns['_r44_pair'](F,F,rho,-radial), observed[None])

    alpha = np.array([-4.,-2.,-1.,0.,1.,2.,4.])
    sigma = np.linspace(.3,1.7,len(alpha))
    v = sigma*sigma
    gaussian = ns['gaussian_raw'](alpha*sigma,v)
    s,a,pdf,g,mg,*unused = gaussian
    obs = ns['_r44_marginal'](s,a,pdf,g,mg,.05*v*v,v)
    errs = []
    for p,got in enumerate(obs,1):
        degree = 7-p
        coef = np.zeros(degree+1);coef[-1] = 1
        h = np.polynomial.hermite_e.hermeval(alpha,coef)
        expected = sigma**p*.05**2/1152*factorial(p)*(-1)**degree*h*pdf
        error = float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
        errs.append(error)
        assert error <= 1e-12

    rng = np.random.default_rng(19091944)
    A = rng.normal(size=(8,8))
    C = (A@A.T)/8+np.eye(8)
    mu = np.linspace(-2,2,8)*np.sqrt(np.diag(C))
    K21 = rng.normal(size=(8,8))*.001
    for r in (0.,-.05,.05):
        v = np.diag(C)
        k4 = r*v*v
        k22 = r*(np.outer(v,v)+2*C*C)/3
        got = ns['_nonlinear_pair_batch'](mu,C,K21,k4,k22,'covariance')
        assert all(np.isfinite(x).all() for x in got[:4])
        scalar = ns['nonlinear_scalar'](mu,v,np.diag(K21),k4)
        assert np.array_equal(scalar[0],got[0])
        for key in ('post_v','post_k3','post_k4'):
            assert np.array_equal(scalar[1][key],got[4][key])
        if r == 0:
            old = parent['_nonlinear_pair_batch'](mu,C,K21,k4,k22,'covariance')
            assert all(np.array_equal(x,y) for x,y in zip(got[:4],old[:4]))
            assert all(np.array_equal(got[4][key],old[4][key]) for key in old[4])
    receipt = dict(passed=True,formal_polynomial_cases=formal_cases,
                   pair_relative_error=relative,pair_symmetry_error=symmetry,
                   marginal_relative_errors=errs,zero_radial_parent_exact=True,
                   scalar_pair_marginals_exact=True,constant_covariance_zero=True,
                   radial_sign_invariant=True,truth_accessed=False,
                   numpy_version=np.__version__,candidate_sha256=sha(HERE/'candidate.py'),
                   unit_sha256=sha(__file__),kernel_sha256=sha(HERE/'kernel.py'))
    with (out/'result.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
