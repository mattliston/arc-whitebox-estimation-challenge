"""Stdlib exact-rational formula check; no arrays, model, labels or numerical libraries."""
from fractions import Fraction as F
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
iv=json.loads((D/'source_inverse.json').read_text())
s=(D/'candidate.py').read_text(); parent=Path(iv['parent']['path']).read_bytes()
assert hashlib.sha256(parent).hexdigest()==iv['parent']['sha256']
assert s.count(iv['after'])==1 and s.replace(iv['after'],iv['before'],1).encode()==parent
ast.parse(s)
w=[[F(1,2),F(-1,3),F(0)], [F(-2,3),F(1,4),F(1,5)], [F(1,6),F(-1,2),F(3,4)]]
q=[[F(1),F(4)], [F(9),F(0)], [F(4),F(1)]]
def check(power,source):
    mean=[sum(row[c] for row in source)/len(source) for c in range(2)]
    for j in range(3):
        mass=sum(w[i][j]**power for i in range(3))
        for c in range(2):
            exact=sum(w[i][j]**power*source[i][c] for i in range(3))
            approx=mass*mean[c]
            residual=sum(w[i][j]**power*(source[i][c]-mean[c]) for i in range(3))
            assert exact-approx==residual and approx>=0
            if all(source[i][c]==source[0][c] for i in range(3)):assert exact==approx
for power in (4,6):
    check(power,q);check(power,[[F(2),F(3)]]*3);check(power,[[F(0),F(0)]]*3)
# The retained E2 slice has a singleton batch dimension and unchanged contraction operands.
assert "_energy_powers[:1]" in iv['after'] and "k['SQ'][:1]" in iv['after']
assert 'axis=2' in iv['after'] and 'axis=1' in iv['after']
assert 'fnp.concatenate((_energy_two, _energy_high), axis=0)' in iv['after']
print(json.dumps(dict(passed=True,exact_rational_identity=True,constant_source_exact=True,zero_source_exact=True,nonnegative=True,source_inverse=True,model_or_array_execution=False)))
