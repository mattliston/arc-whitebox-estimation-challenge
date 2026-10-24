from pathlib import Path
import hashlib
import json
import math
import numpy as np
from fields import source, terminal

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
seal = json.loads((HERE/'seal.json').read_text())
for name, digest in seal['sources'].items():
    assert sha(HERE/name) == digest
n, k = 7, 4
rng = np.random.default_rng(2026092728)
C = rng.normal(size=(n, n)) * .02
H = rng.normal(size=(n, k)); s = np.exp(rng.normal(size=n)*.1)
W = rng.normal(size=(n, n)); t = np.exp(rng.normal(size=n)*.1)
g = rng.uniform(.1, .9, n); p = rng.uniform(.01, .4, n)
copies = [x.copy() for x in (C, s, H, W, t, g, p)]
U, V = source(C, s, H)
R = [[0. if i == j else float(C[i,j]/s[i]/s[j]) for j in range(n)] for i in range(n)]
hc = [[float(H[j,a])-math.fsum(float(H[q,a]) for q in range(n))/n for a in range(k)] for j in range(n)]
u = np.array([[math.fsum(R[i][j]*hc[j][a] for j in range(n))/math.sqrt(n) for a in range(k)] for i in range(n)])
v = np.array([[math.fsum(R[i][j]**2*hc[j][a] for j in range(n)) for a in range(k)] for i in range(n)])
z = terminal(U, V, s, W, t, g, p)
reference = np.array([[math.fsum(float(W[i,j]*s[i])*u[i,a] for i in range(n))*g[j]/t[j] for a in range(k)] +
                      [math.fsum(float(W[i,j]**2*s[i]**2)*v[i,a] for i in range(n))*p[j]/(2*t[j]**2) for a in range(k)] for j in range(n)])
perm = rng.permutation(n)
pu, pv = source(C[np.ix_(perm,perm)], s[perm], H[perm])
pz = terminal(pu, pv, s[perm], W[np.ix_(perm,perm)], t[perm], g[perm], p[perm])
errors = dict(source_u=float(np.max(np.abs(U-u))), source_v=float(np.max(np.abs(V-v))),
              terminal=float(np.max(np.abs(z-reference))), permutation=float(np.max(np.abs(pz-z[perm]))))
assert max(errors.values()) < 2e-14, errors
assert all(np.array_equal(x,y) for x,y in zip((C,s,H,W,t,g,p),copies))
assert np.array_equal(z, terminal(*source(C,s,H),s,W,t,g,p))
result = dict(passed=True, errors=errors, input_bytes_unchanged=True, repeat_exact=True,
              seal_sha256=sha(HERE/'seal.json'), targets_accessed=False)
with (HERE/'unit_result.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
