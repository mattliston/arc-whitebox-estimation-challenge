"""Finite rational algebra cases and source binding; no arrays/model execution."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
cases=0
for tiny in (False,True):
 for columns in (False,True):
  for zeros in (False,True):
   for raw in (None,(F(2,81),F(0),F(11,28)),(F(-1,3),F(2,9),F(0))):
    n=3;scale=F(1,10**12) if tiny else F(1)
    bp=[[scale*F((i+1)*(j-1),7) for j in range(n)] for i in range(n)]
    legs=[[[F((i-j)*(a+2)+i+1,9+a+j) if i!=j else F(0) for j in range(n)] for i in range(n)] for a in range(3)]
    c=[[F(0) if zeros else F((-1)**(a+j)*(a+j+1),5) for j in range(n)] for a in range(3)] if columns else [F(0) if zeros else F((-1)**a*(a+1),5) for a in range(3)]
    coeff=lambda a,j:c[a][j] if columns else c[a]
    for j in range(n):
     den=max(sum(bp[i][j]**2 for i in range(n)),F(1,10**20))
     pr=[sum(bp[i][j]*legs[a][i][j] for i in range(n))/den for a in (0,2)]
     forced=[coeff(0,j),F(5,9),coeff(2,j)]
     wp=forced[0]*pr[0]+forced[2]*pr[1]
     ce=[forced[a]+(raw[a] if raw is not None else 0) for a in range(3)]
     for i in range(n):
      old=forced[0]*(legs[0][i][j]-bp[i][j]*pr[0])+forced[1]*legs[1][i][j]+forced[2]*(legs[2][i][j]-bp[i][j]*pr[1])
      if raw is not None:old+=sum(raw[a]*legs[a][i][j] for a in range(3))
      new=sum(ce[a]*legs[a][i][j] for a in range(3))-bp[i][j]*wp
      assert old==new
    cases+=1
iv=json.loads((D/'source_inverse.json').read_text());parent=Path(iv['parent']['path']).read_bytes();source=(D/'candidate.py').read_text()
assert hashlib.sha256(parent).hexdigest()==iv['parent']['sha256']
assert source.count(iv['after'])==1 and source.replace(iv['after'],iv['before'],1).encode()==parent
assert iv['after'].startswith('    if _BD_J1_PHYS is None:\n'+''.join('    '+line if line.strip() else line for line in iv['before'].splitlines(True)))
print(json.dumps(dict(passed=True,rational_cases=cases,scalar_and_column_coefficients=True,zero_coefficients=True,floored_denominator=True,raw_none_and_signed=True,fallback_verbatim=True,source_inverse=True,arrays_or_models=False)))
