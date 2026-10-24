"""Integrate the Gaussian radius analytically, propagate angular moments.

For this PHASE 2 bias-free ReLU network, f(r u)=r f(u). Write a standard
Gaussian input as (chi_n/sqrt(n)) times an independent uniform point on
the radius-sqrt(n) sphere. The target Gaussian mean is therefore c_n
times the angular mean, with c_n=E[chi_n]/sqrt(n).

The angular input has C=I, k3=0 and k4_iiii=-6/(n+2), represented exactly
by the current radial fourth-order ansatz. Higher angular cumulants are
not propagated: this is a new approximate moment trajectory, not an
assertion that truncated Gaussian and angular recursions are identical.
Unlike the closed cubature branch, this method draws no sample points.
"""
import numpy as np
from scipy.special import gammaln
import radial_binary_compact as base
from radial_compact_radialfeed_v1 import PREDICT_SOURCE
from radial_mehler import make_nonlinear


def gaussian_radius_mean(n):
    return float(np.exp(.5*np.log(2/n)+gammaln((n+1)/2)-gammaln(n/2)))


source = PREDICT_SOURCE
old = "    radial = 0.0"
new = "    radial = -6.0/(n+2)\n    gaussian_mean_scale = gaussian_radius_mean(n)"
assert source.count(old) == 1
source = source.replace(old, new)
old = "    return np.stack(means), states"
assert source.count(old) == 1
source = source.replace(old, "    return gaussian_mean_scale*np.stack(means), states")
env = dict(vars(base), nonlinear=make_nonlinear(6, 2, True, True), gaussian_radius_mean=gaussian_radius_mean)
exec(source, env)
predict = env["predict"]
