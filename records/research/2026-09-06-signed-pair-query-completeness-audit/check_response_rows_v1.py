"""Certify only the exact positive-linear limit and scalar CP contraction."""
import json
from pathlib import Path
import numpy as np
from radial_mehler import make_nonlinear
from radial_response_rows_v1 import omitted_radial_source

out = Path(__file__).resolve().parent / "response_rows_certificate_v1.json"
if out.exists():
    raise FileExistsError(out)
rng = np.random.default_rng(48523)
n, r = 12, 19
p = rng.normal(size=(n, r)) * .03
lam = rng.normal(size=r)
cp = (p*p) @ (p*lam).T
diag = np.diag(cp).copy()
omitted = np.arange(n) < 4
alive = np.ones(n, dtype=bool)
dropped = cp.copy(); dropped[omitted] = 0
np.fill_diagonal(dropped, diag)
A = rng.normal(size=(n, n)) / np.sqrt(n)
C = A @ A.T + np.eye(n)
s = np.sqrt(np.diag(C))
mu = s * rng.uniform(-1, 1, n)
mu[omitted] = 12 * s[omitted]
k4 = np.diag(C)**2 * .003
k22 = .003 * (np.outer(np.diag(C), np.diag(C)) + 2*C*C) / 3
fn = make_nonlinear(6, 2, True, True)
m, cov, target, radial, aux = fn(mu, C, cp, k4, k22, "covariance")
m0, cov0, target0, radial0, aux0 = fn(mu, C, dropped, k4, k22, "covariance")
source = omitted_radial_source(p, lam, diag, omitted, alive, aux0, m0)
factor = (aux0["gate"]-m0*aux0["pdf"]/aux0["sigma"])*omitted
beta = 2*(aux0["gaussian_mean"]-m0*aux0["gate"])
dense_source = 6*np.sum((cp-dropped)*factor[:, None]*beta[None, :])
assert abs(source-dense_source) < 1e-15
scale = aux["gate"][:, None]**2 * aux["gate"][None, :]
rest_error = float(np.max(np.abs((target-scale*cp)-(target0-scale*dropped))))
corrected_radial = radial0 + source/(np.trace(cov0)**2+2*np.sum(cov0*cov0))
radial_error = float(abs(corrected_radial-radial))
assert np.array_equal(m,m0)
assert np.max(np.abs(cov-cov0)) < 1e-14
assert rest_error < 1e-12, rest_error
assert radial_error < 1e-13, radial_error
record = dict(source_abs_error=float(abs(source-dense_source)),
              linear_limit_star_rest_max_abs=rest_error,
              linear_limit_radial_abs_error=radial_error,
              qualification="Only exact positive-linear limit certified; gate .99 is approximate and requires an accuracy screen.")
out.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
