"""Exact scalar identities and source binding only; no arrays or model execution."""
from pathlib import Path
from fractions import Fraction as F
import ast, hashlib, json, math, struct
D = Path(__file__).resolve().parent
def mul(a,b):
    r=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):r[i+j]+=x*y
    return r
def sub(a,b):
    r=[(a[i] if i<len(a) else 0)-(b[i] if i<len(b) else 0) for i in range(max(len(a),len(b)))]
    while len(r)>1 and r[-1]==0:r.pop()
    return r
def der(a):return [i*a[i] for i in range(1,len(a))]
N5=[0,945,0,105,0,1];D5=[945,0,420,0,15]
N6=[0,10395,0,1260,0,21];D6=[10395,0,4725,0,210,0,1]
for N,Q,degree,sign in ((N5,D5,10,1),(N6,D6,12,-1)):
    residual=sub(sub(mul(der(N),Q),mul(N,der(Q))),sub(mul(Q,Q),mul(N,N)))
    assert residual==[0]*degree+[sign]
assert sub(mul(N5,D6),mul(N6,D5))==[0]*11+[1]
p=[893025,-99225,6300,-315,15];zmax=F(225,16)
bern=[sum(F(p[i])*zmax**i*F(math.comb(k,i),math.comb(4,i)) for i in range(k+1)) for k in range(5)]
assert all(x>0 for x in bern)
c=struct.unpack('f',struct.pack('f',3.6467385292053223))[0];z=c*c
r5=c*(945+z*(105+z))/(945+z*(420+15*z))
gap=c**11/((945+z*(420+15*z))*(10395+z*(4725+z*(210+z))))
u=2**-24;g=6*u/(1-6*u);roundoff=(1+g)/(1-g)*(1+u)-1
assert gap+abs(1-r5)+roundoff*1.000001 < .001467
iv=json.loads((D/'source_inverse.json').read_text());s=(D/'candidate.py').read_text();parent=Path(iv['parent']['path']).read_bytes()
assert hashlib.sha256(parent).hexdigest()==iv['parent']['sha256']
assert s.endswith(iv['suffix']) and s[:-len(iv['suffix'])].replace(iv['after'],iv['before'],1).encode()==parent
ast.parse(s)
print(json.dumps(dict(passed=True,ODE_identities=True,monotone_through_3_75=True,real_error_bound=gap+abs(1-r5),standard_FP32_bound=.001467,source_inverse=True,model_or_array_execution=False)))
