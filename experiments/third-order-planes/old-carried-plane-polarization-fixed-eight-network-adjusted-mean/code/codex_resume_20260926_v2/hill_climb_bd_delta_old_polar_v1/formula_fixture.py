"""Signed-pair polarization and explicit row-map identity; stdlib only."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
D=Path(__file__).resolve().parent;cases=0
for u in (F(0),F(2),F(-3,7),F(2**20)):
 for v in (F(0),F(1,2**20),F(-2,3),F(4)):
  for lam in (F(0),F(-2,7),F(3,5)):
   for fu,fv in ((F(2,3),F(-5,7)),(F(0),F(0))):
    plus=u+v;minus=u-v
    su=4*lam*u*v;dv=2*lam*(u*u+v*v)
    assert su==lam*plus*plus-lam*minus*minus
    assert dv==lam*plus*plus+lam*minus*minus
    assert su*fu+dv*fv==lam*plus*plus*(fu+fv)-lam*minus*minus*(fu-fv)
    cases+=1
for outer,inner in ((list(range(8)),[5,0,7,2]),([7,3,1,5,0],[4,0,2]),(list(range(8)),list(range(8)))):
 physical=[outer[i] for i in inner];inverse=[len(physical)]*8
 for local,absolute in enumerate(physical):inverse[absolute]=local
 values=[F(i+1,7) for i in range(len(physical))]+[F(0)]
 for r in range(8):assert values[inverse[r]]==(values[physical.index(r)] if r in physical else 0)
iv=json.loads((D/'source_inverse.json').read_text());parent=Path(iv['parent']['path']).read_bytes();source=(D/'candidate.py').read_text()
assert hashlib.sha256(parent).hexdigest()==iv['parent']['sha256']
suffix='\n'+(D/'fragment.py').read_text();assert source.endswith(suffix);source=source[:-len(suffix)]
for edit in reversed(iv['edits']):assert source.count(edit['after'])==1;source=source.replace(edit['after'],edit['before'],1)
assert source.encode()==parent
print(json.dumps(dict(passed=True,rational_cases=cases,physical_row_maps=3,source_inverse=True,arrays_or_models=False)))
