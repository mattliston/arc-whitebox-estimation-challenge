"""Keep the original pair map outside the target-free qualified domain."""
import numpy as np
from gamma_radial_pair_v1 import pair, CONDITIONAL


def guarded_pair(mu, covariance, third21, radial, count=8):
    if np.isfinite(radial) and 0 <= radial <= .10:
        return pair(mu, covariance, third21, radial, count)
    v = np.diag(covariance)
    k4 = radial*v*v
    k22 = radial*(np.outer(v, v)+2*covariance*covariance)/3
    original = CONDITIONAL(mu, covariance, third21, k4, k22, 'covariance')
    return original[0], original[1]
