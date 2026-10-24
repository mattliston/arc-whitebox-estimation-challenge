"""Exact scalar compensation identity and source inverse; stdlib only."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
cases=0
for old in (F(0),F(1),F(-3,7),F(2**24),F(1,2**40)):
 for new in (F(0),F(1),F(-4,9),F(2**24+1),F(-1,2**40)):
  for weight in (F(0),F(2,5),F(-7,3)):
   for quadratic in (F(0),F(5,8)):
    diff=new-old
    assert (2*old*diff+diff*diff)*weight+quadratic*diff==(new+old)*diff*weight+quadratic*diff
    cases+=1
iv=json.loads((D/'source_inverse.json').read_text());parent=Path(iv['parent']['path']).read_bytes();source=(D/'candidate.py').read_text()
assert hashlib.sha256(parent).hexdigest()==iv['parent']['sha256']
assert source.count(iv['after'])==1 and source.replace(iv['after'],iv['before'],1).encode()==parent
assert iv['after'].index('del new')>iv['after'].index('delta = ')
print(json.dumps(dict(passed=True,rational_cases=cases,zero_and_signed=True,source_inverse=True,arrays_or_models=False)))
