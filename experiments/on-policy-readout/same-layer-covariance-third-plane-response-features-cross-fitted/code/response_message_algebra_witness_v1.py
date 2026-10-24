"""Small label-free checks of message orientation and Gaussian mean transport."""
from pathlib import Path
import json
import math

import numpy as np

rng = np.random.default_rng(9272026)
n = 7
s = np.exp(rng.normal(size=n))
a = rng.normal(size=n)
p = np.exp(-a*a/2)/math.sqrt(2*math.pi)
phi = np.column_stack((p, a*p, (a*a-1)*p))
raw = rng.normal(size=(n,n))
C = raw@raw.T
K = rng.normal(size=(n,n))
R = C/(s[:,None]*s[None,:])
T = K/((s*s)[:,None]*s[None,:])
np.fill_diagonal(R, 0)
np.fill_diagonal(T, 0)
fast = np.concatenate((R@phi/math.sqrt(n), (R*R)@phi,
                       T@phi/math.sqrt(n), T.T@phi/math.sqrt(n)), axis=1)
direct = np.zeros_like(fast)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        for k in range(3):
            direct[i,k] += C[i,j]/(s[i]*s[j])*phi[j,k]/math.sqrt(n)
            direct[i,k+3] += (C[i,j]/(s[i]*s[j]))**2*phi[j,k]
            direct[i,k+6] += K[i,j]/(s[i]**2*s[j])*phi[j,k]/math.sqrt(n)
            direct[i,k+9] += K[j,i]/(s[j]**2*s[i])*phi[j,k]/math.sqrt(n)
message_error = float(np.max(np.abs(fast-direct)))
assert message_error < 1e-11

def relu_gaussian_mean(mu, sd):
    alpha = mu/sd
    gate = np.asarray([.5*math.erfc(-float(x)/math.sqrt(2)) for x in alpha])
    pdf = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
    return mu*gate+sd*pdf, gate

weights = [rng.normal(size=(n,n))/math.sqrt(n) for _ in range(4)]
scales = [np.exp(rng.normal(scale=.2,size=n)) for _ in weights]
sources = [rng.normal(scale=.1,size=n) for _ in weights]
initial = rng.normal(scale=.2,size=n)
mu, derivative = initial.copy(), np.zeros(n)
for W, sd, source in zip(weights, scales, sources):
    mu, gate = relu_gaussian_mean(W.T@mu, sd)
    derivative = gate*(W.T@derivative)+source

def perturbed(epsilon):
    mu = initial.copy()
    for W, sd, source in zip(weights, scales, sources):
        mu, _ = relu_gaussian_mean(W.T@mu, sd)
        mu += epsilon*source
    return mu

epsilon = 1e-5
finite_difference = (perturbed(epsilon)-perturbed(-epsilon))/(2*epsilon)
transport_error = float(np.max(np.abs(derivative-finite_difference)))
assert transport_error < 1e-9
report = dict(message_orientation_max_abs=message_error,
              frozen_gaussian_transport_max_abs=transport_error,
              endpoint_truth_accessed=False,
              scope='Small nonsymmetric K21 contraction and derivative of the fixed-variance Gaussian mean map, not the native full estimator Jacobian.')
with Path(__file__).with_suffix('.json').open('x') as fp:
    json.dump(report, fp, indent=2)
print(json.dumps(report))
