import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
n = 1024
c = math.exp(.5*math.log(2/n)+math.lgamma((n+1)/2)-math.lgamma(n/2))
a3, a4 = (n+1)*c/n, (n+2)/n
r = -2/(n+2)
phi = 1/math.sqrt(2*math.pi)
h, q3, q4 = phi/c, 2*phi/a3, 1.5/a4
s2 = .5+r/2-h*q3
s3 = q4-3*h*q3+3*h*h-.75-3*r/8
s4 = q4-4*h*q3+6*h*h-6*h**4-.75-3*r/16
delta = r/4+2*h*h/(n+1)
rng = np.random.default_rng(54219)
errors = []
for _ in range(32):
    A = rng.normal(size=(3,3)); C = A@A.T
    v, ca, cb, cab = C[2,2], C[0,2], C[1,2], C[0,1]
    sigma = math.sqrt(v); m = sigma*h
    y2, y3, y4 = v/2, q3*sigma**3, q4*v*v
    ab_y = phi*sigma*(cab+ca*cb/v)/a3
    ab_y2 = (v*cab+2*ca*cb)/(2*a4)
    k_ab_y = ab_y-m*cab
    k_a_y2 = ca*(y3/v-m)
    k_ab_y2 = ab_y2-2*m*ab_y+(2*m*m-v/2)*cab-.5*ca*cb
    k_a_y3 = ca*(y4/v-3*m*y3/v+3*m*m-.75*v)
    k_y4 = y4-4*m*y3+6*m*m*y2-3*m**4-3*(y2-m*m)**2
    expected = [
        -m*cab/(n+1)+y3*ca*cb/(2*v*v),
        ca*m*(n-1)/(n+1),
        r*(v*cab+2*ca*cb)/4 + delta*v*cab+s2*ca*cb,
        3*r*v*ca/8+v*s3*ca,
        3*r*v*v/16+v*v*s4,
    ]
    observed = [k_ab_y,k_a_y2,k_ab_y2,k_a_y3,k_y4]
    errors.append(max(abs(a-b)/max(1.,abs(a),abs(b)) for a,b in zip(observed,expected)))
assert max(errors) < 5e-13, max(errors)

# Execute only the pure polynomial builder and new small L0 helper with NumPy.
parent = D.parent/'astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py'
lines = parent.read_text().splitlines(keepends=True)
tree = ast.parse(''.join(lines))
node = next(z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name=='_k4l_tables')
scope = {}
exec(''.join(lines[node.lineno-1:node.end_lineno]),scope)
tabs=scope['_k4l_tables']()
cache={'idxB':np.asarray(tabs[2]),'coefB':np.asarray(tabs[3])}
scope.update(_rc_math=math,_GR_C1=c,_GR_C3=a3,_GR_C4=a4,_GR_N=n,_GR_INPUT_RADIAL=3*r,
             _S19_BSRC_DIV=6.299047545852794,fnp=np,_E2Hold=lambda:lambda x:x,
             _k4l_const=lambda:cache,_K4L_RAW={},_K4L_B={},_K4L_STATS={},
             _k4l_birth=lambda *a:None)
fragment=D/'fragment_k4l_l0_v2.py.txt'
exec(fragment.read_text(),scope)
sigma=rng.uniform(.5,2,size=8); v=sigma*sigma
tp=rng.uniform(1,2,size=8); tm=rng.uniform(-2,-1,size=8)
selected=np.arange(8)
aux={'sigma':sigma,'pre_v':v,'gr_raw3':q3*sigma**3}
scope['_k4l_birth'](0,aux,sigma*h,selected,tp,tm)
coef=scope['_K4L_RAW'][0].astype(float)
Y=rng.normal(size=(5,8)); E=rng.normal(size=(5,8))
p=Y+tp*E; dd=(tp-tm)*E
actual=np.sum(dd*(coef[0]*p**3+dd*(coef[1]*p*p+dd*(coef[2]*p+coef[3]*dd))),axis=1)
expected=np.sum(6*s2*Y*Y*E*E+4*(v*s3)*Y*E**3+(v*v*s4)*E**4,axis=1)
rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
assert rel<2e-6,rel
out={'status':'PASS_ALGEBRA_AND_SMALL_NUMPY_HELPER','n':n,'c':c,'a3':a3,'a4':a4,
     'r':r,'S2':s2,'S3_normalized':s3,'S4_normalized':s4,'trace_innovation_delta':delta,
     'max_mixed_cumulant_identity_relative':max(errors),'float32_K4L_polynomial_relative':rel,
     'fragment_sha256':hashlib.sha256(fragment.read_bytes()).hexdigest(),
     'scope':'One-hub exact raw-moment identities and local polynomial ownership only; no full NP closure or native-model accuracy claim.'}
(D/'TINY_v2.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
