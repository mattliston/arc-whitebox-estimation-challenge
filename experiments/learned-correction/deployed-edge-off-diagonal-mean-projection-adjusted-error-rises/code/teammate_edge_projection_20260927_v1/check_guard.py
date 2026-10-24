"""Small scalar fixtures for the candidate guard; no numerical libraries."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
THRESHOLD = 2.0 ** -20


def project(edge, m):
    norm_sq = sum(x*x for x in m)
    if not norm_sq > 0:
        return edge
    h = [x/math.sqrt(norm_sq) for x in m]
    d = [1-2*x*x for x in h]
    if not all(x > THRESHOLD for x in d):
        return edge
    n = len(m)
    A = [[edge[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)]
    b = [sum(A[i][j]*h[j] for j in range(n)) for i in range(n)]
    v = [h[i]/d[i] for i in range(n)]
    bd = [b[i]/d[i] for i in range(n)]
    scale = sum(h[i]*bd[i] for i in range(n))/(1+sum(h[i]*v[i] for i in range(n)))
    u = [2*(bd[i]-v[i]*scale) for i in range(n)]
    return [[edge[i][i] if i == j else A[i][j]-(u[i]*h[j]+h[i]*u[j])*.5
             for j in range(n)] for i in range(n)]


A = [[float((i+1)*(j+1)-2*(i+j)+1)/7 for j in range(4)] for i in range(4)]
rows = []
for name, m in [('zero',[0.,0.,0.,0.]), ('underflow',[1e-300]*4),
                ('dominant',[1.,0.,0.,0.]), ('two_support',[1.,1.,0.,0.]),
                ('near_boundary',[1.,1.,1e-5,1e-5])]:
    out = project(A,m)
    assert out is A
    rows.append(dict(case=name, unchanged_same_object=True))

max_abs = 0.0
for m in ([1.,1.,1.,1.], [1.,1.,2.,2.], [-1.,1.,-2.,2.]):
    X=project(A,m)
    assert X is not A
    assert all(X[i][i] == A[i][i] for i in range(4))
    assert all(X[i][j] == X[j][i] for i in range(4) for j in range(4))
    residue=max(abs(sum(X[i][j]*m[j] for j in range(4) if i != j)) for i in range(4))
    max_abs=max(max_abs,residue)
    assert residue < 1e-13
    Y=project(X,m)
    assert max(abs(X[i][j]-Y[i][j]) for i in range(4) for j in range(4)) < 1e-13

record=dict(passed=True, guard_rows=rows, max_offdiag_mean_residual=max_abs,
            normal_fixtures=3, no_numerical_libraries=True, estimator_imported=False,
            limitation='Scalar mirror of helper; real FNP native evaluation remains root-owned.')
(HERE/'guard_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
