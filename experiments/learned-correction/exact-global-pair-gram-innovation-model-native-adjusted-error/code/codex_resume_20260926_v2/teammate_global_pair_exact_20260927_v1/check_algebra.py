"""Tiny exact rational Gram/sketch-variance witnesses, no arrays or labels."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
dot=lambda x,y:sum(a*b for a,b in zip(x,y))
matrices=[[[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],
    [[F(1,2),F(-1,3),F(2,5)],[F(3,7),F(-1,2),F(1,4)],[F(-2,3),F(1,5),F(5,9)]]]
rows=[]
for R in matrices:
    n=len(R)
    G=[[dot(x,y) for y in R] for x in R]
    assert all(G[i][j]==G[j][i] for i in range(n) for j in range(n))
    for t in product((-1,0,1),repeat=n):
        quad=sum(F(t[i])*G[i][j]*t[j] for i in range(n) for j in range(n))
        expected=sum(sum(F(t[i])*R[i][k] for i in range(n))**2 for k in range(n))
        assert quad==expected and quad>=0
    signs=list(product((-1,1),repeat=n))
    for i in range(n):
        for j in range(n):
            values=[dot(R[i],z)*dot(R[j],z) for z in signs]
            mean=sum(values)/len(values)
            var=sum((x-mean)**2 for x in values)/len(values)
            formula=dot(R[i],R[i])*dot(R[j],R[j])+G[i][j]**2-2*sum(x*x*y*y for x,y in zip(R[i],R[j]))
            assert mean==G[i][j] and var==formula
    rows.append(dict(size=n,gram_symmetric=True,quadratic_identity=True,
        ideal_sign_unbiasedness=True,exact_variance_formula=True))
assert matrices[1][0][1]!=matrices[1][1][0]
R=matrices[1]
R2=[[sum(R[i][k]*R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
assert R2!=[[dot(x,y) for y in R] for x in R]
record=dict(passed=True,rows=rows,asymmetric_R_squared_distinguished_from_RRt=True,
    identity_offdiag_width64_variance='1/64',source_only=True,label_payloads_read=False,
    numerical_libraries=False,model_or_estimator_imported=False)
(HERE/'algebra_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
