"""Fixed n4 tensor and source certificate, no weights or targets."""
from pathlib import Path
from itertools import product, combinations, permutations
import hashlib
import json
import resource
import time
import numpy as np
import chi_radius_trace_shape_v1 as model

HERE = Path(__file__).resolve().parent
OUT = HERE/'chi_radius_trace_shape_certificate_v1.json'
assert not OUT.exists()
start = time.time()
rng = np.random.default_rng(930101)
n = 4
matrix = rng.normal(size=(n, n))
B = matrix@matrix.T/n+.4*np.eye(n)
mu = rng.normal(size=n)*.3
factor = rng.normal(size=(n, 7))*.3
TA = np.einsum('ir,jr,kr,r->ijk', factor, factor, factor, rng.normal(size=7))
rA = .073  # One fixed, non-tuned algebra fixture.
c, delta, a3, a4 = model.C1, model.D1, model.A3, model.A4
beta, u, z, w = model.BETA, model.U4, model.Z4, model.W4
m = c*mu
C = B+delta*np.outer(mu, mu)
TG = np.empty_like(TA)
for i, j, k in product(range(n), repeat=3):
    TG[i, j, k] = (a3*TA[i, j, k]+c/model.INPUT_DIM*(mu[i]*B[j, k]+mu[j]*B[i, k]+mu[k]*B[i, j])
                   +beta*mu[i]*mu[j]*mu[k])
Q = np.empty((n,)*4)
R = np.empty_like(Q)
PAIR = np.empty_like(Q)
for ix in product(range(n), repeat=4):
    i, j, k, l = ix
    pairing = B[i, j]*B[k, l]+B[i, k]*B[j, l]+B[i, l]*B[j, k]
    muT = sum(mu[ix[p]]*TA[tuple(ix[j] for j in range(4) if j != p)] for p in range(4))
    mumuB = 0.
    for left in combinations(range(4), 2):
        right = [p for p in range(4) if p not in left]
        mumuB += mu[ix[left[0]]]*mu[ix[left[1]]]*B[ix[right[0]], ix[right[1]]]
    R[ix] = u*muT+z*mumuB+w*np.prod(mu[list(ix)])
    PAIR[ix] = pairing
    Q[ix] = (a4*rA/3+2/model.INPUT_DIM)*pairing+R[ix]
tau = float(np.einsum('iijj->', Q))
radial = 3*tau/(np.trace(C)**2+2*np.sum(C*C))
K = np.einsum('iij->ij', TG)
result = model.gaussian_planes(m, C, K, radial, bank_fourth_trace=tau)
expected = {'k22': np.einsum('iijj->ij', Q), 'k31': np.einsum('iiij->ij', Q),
            'k4': np.einsum('iiii->i', Q)}
errors = {key: float(np.max(np.abs(result[key]-value))) for key, value in expected.items()}
errors.update(angular_mean=float(np.max(np.abs(result['mu_A']-mu))),
              angular_covariance=float(np.max(np.abs(result['C_A']-B))),
              angular_third_plane=float(np.max(np.abs(result['K21_A']-np.einsum('iij->ij', TA)))),
              angular_scalar=abs(result['angular_radial']-rA),
              diagonals=max(float(np.max(np.abs(np.diag(result[key])-result['k4']))) for key in ('k22', 'k31')))
trace_error = abs(result['trace_new']-tau)/max(abs(tau), 1e-30)
full_radial_trace = float(np.einsum('iijj->', R+2/model.INPUT_DIM*PAIR))
DB = np.trace(B)**2+2*np.sum(B*B)
lambda_full = (tau-full_radial_trace)/DB
cancellation_error = abs(lambda_full+2/model.INPUT_DIM-result['eta'])
zero_mean = model.gaussian_planes(np.zeros(n), C, K, radial)
zero_mean_error = max(float(np.max(np.abs(zero_mean['k22']-radial/3*(np.outer(np.diag(C), np.diag(C))+2*C*C)))),
                      float(np.max(np.abs(zero_mean['k31']-radial*np.diag(C)[:, None]*C))))
gaussian_input = model.gaussian_planes(np.zeros(n), C, np.zeros_like(C), 0.)
gaussian_input_error = max(float(np.max(np.abs(gaussian_input[k]))) for k in ('k22', 'k31', 'k4'))
angular_input_scalar_error = abs(gaussian_input['angular_radial']+6/(model.INPUT_DIM+2))

# Independent dense contraction of p*Sym4(m,T) verifies the two nonlocal
# fourth-to-third source terms discussed in the audit.
pcoef = u/(a3*c)
gate = rng.uniform(.2, .8, n)
d0 = rng.uniform(.1, .6, n)
QmT = np.empty((n,)*4)
for ix in product(range(n), repeat=4):
    QmT[ix] = pcoef*sum(m[ix[pos]]*TG[tuple(ix[j] for j in range(4) if j != pos)] for pos in range(4))
source = np.empty_like(TG)
expected_source = np.empty_like(TG)
h = gate*m
M = (gate[:, None]*d0[None, :]*K.T+gate[None, :]*d0[:, None]*K)/2
for i, j, k in product(range(n), repeat=3):
    source[i, j, k] = .5*(gate[i]*gate[j]*d0[k]*QmT[i, j, k, k]
                             +gate[i]*gate[k]*d0[j]*QmT[i, k, j, j]
                             +gate[j]*gate[k]*d0[i]*QmT[j, k, i, i])
    expected_source[i, j, k] = pcoef*(h[i]*M[j, k]+h[j]*M[i, k]+h[k]*M[i, j])
    expected_source[i, j, k] += pcoef*TG[i, j, k]*(d0[i]*m[i]*gate[j]*gate[k]
                                          +gate[i]*d0[j]*m[j]*gate[k]
                                          +gate[i]*gate[j]*d0[k]*m[k])
source_error = float(np.max(np.abs(source-expected_source)))
report = dict(errors=errors, trace_relative_error=trace_error,
              radial_cancellation_error=cancellation_error,
              zero_mean_error=zero_mean_error,
              gaussian_input_zero_error=gaussian_input_error,
              gaussian_input_angular_scalar_error=angular_input_scalar_error,
              nonlocal_fourth_to_third_source_error=source_error,
              passed=max(list(errors.values())+[trace_error, cancellation_error, zero_mean_error,
                          gaussian_input_error, angular_input_scalar_error, source_error]) <= 1e-11,
              wall_s=time.time()-start,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256((HERE/'chi_radius_trace_shape_v1.py').read_bytes()).hexdigest(),
              qualification='Fixed output-dimension4 dense tensor, physical input dimension1024, one fixed angular scalar. Exact trace ownership, repeated fourth planes and nonlocal source decomposition; no network weights, targets, source fit, scalar sweep, wide test, inference meter or score.')
OUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
