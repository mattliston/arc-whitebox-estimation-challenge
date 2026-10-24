"""Independent Fraction/dense-solve witnesses; no NumPy or estimator/data."""
from fractions import Fraction as F
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def solve(A, b):
    n = len(b)
    C = [[F(v) for v in row] + [F(y)] for row, y in zip(A, b)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if C[i][k])
        C[k], C[pivot] = C[pivot], C[k]
        q = C[k][k]
        C[k] = [v/q for v in C[k]]
        for i in range(n):
            if i != k:
                q = C[i][k]
                C[i] = [x-q*y for x,y in zip(C[i],C[k])]
    return [row[-1] for row in C]

def project(E, m):
    n = len(m)
    S = sum(x*x for x in m)
    A = [[F(E[i][j]) if i != j else F(0) for j in range(n)] for i in range(n)]
    b = [sum(A[i][j]*m[j] for j in range(n)) for i in range(n)]
    L = [[(S-2*m[i]*m[i] if i == j else 0)+m[i]*m[j] for j in range(n)] for i in range(n)]
    v = solve(L,b)
    D = [S-2*x*x for x in m]
    assert all(D)
    B = [b[i]/D[i] for i in range(n)]
    H = [m[i]/D[i] for i in range(n)]
    c = sum(m[i]*B[i] for i in range(n))/(1+sum(m[i]*H[i] for i in range(n)))
    assert v == [B[i]-H[i]*c for i in range(n)]
    X = [[A[i][j]-m[i]*v[j]-v[i]*m[j] if i != j else F(E[i][i]) for j in range(n)] for i in range(n)]
    assert all(X[i][j] == X[j][i] for i in range(n) for j in range(n))
    assert all(X[i][i] == E[i][i] for i in range(n))
    assert all(sum(X[i][j]*m[j] for j in range(n)) == E[i][i]*m[i] for i in range(n))
    return X

rows=[]
for mean in ([1,1,1], [1,1,2,2], [1,2,2,3,3]):
    m=list(map(F,mean));n=len(m)
    A=[[F((i+1)*(j+1)-2*(i+j)+1,7) for j in range(n)] for i in range(n)]
    B=[[F((i-j)**2-3,11) for j in range(n)] for i in range(n)]
    X=project(A,m);Y=project(B,m)
    assert project(X,m) == X
    inner=lambda U,V:sum(U[i][j]*V[i][j] for i in range(n) for j in range(n) if i != j)
    assert inner(X,B)==inner(A,Y)
    residual=[[A[i][j]-X[i][j] for j in range(n)] for i in range(n)]
    assert inner(residual,Y)==0
    rows.append(dict(n=n,mean=mean,exact_diagonal=True,offdiagonal_annihilation=True,
                     dense_and_rank_one_solve_identical=True,idempotent=True,
                     offdiagonal_self_adjoint=True,orthogonal_residual=True))
record=dict(passed=True,rows=rows,numerical_libraries_or_data_used=False,
            scope='offdiag(X)m=0; Xm=diag(E)m with original diagonal preserved')
with (HERE/'edge_projection_exact_result.json').open('x') as f:
    json.dump(record,f,indent=2)
print(json.dumps(record))
