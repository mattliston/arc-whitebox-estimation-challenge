"""Independent physical normal equations and scalar score accumulation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CAP=HERE.parent/'fourth_matrix_face_capacity_v1'
out=HERE/'verification.json';assert not out.exists()
assessment=json.loads((HERE/'assessment.json').read_text())
z=dict(np.load(HERE/'capture.npz',allow_pickle=False))
assert hashlib.sha256((HERE/'capture.npz').read_bytes()).hexdigest()==assessment['capture_sha256']
C=z['post12_covariance'];D=z['Dpost12'];v=np.diag(C);a=np.diag(D)
mask=np.outer(z['alive'],z['alive'])
q31=z['analytic_post31']*mask;q22=z['analytic_post22']*mask
np.fill_diagonal(q31,z['analytic_k4']);np.fill_diagonal(q22,z['analytic_k4']);q22=(q22+q22.T)/2
e4=v*a-z['analytic_k4']
e31=(v[:,None]*D+a[:,None]*C)/2-q31
e22=(v[:,None]*a[None,:]+a[:,None]*v[None,:]+4*C*D)/6-q22
np.fill_diagonal(e31,0);np.fill_diagonal(e22,0)
gd=2*v*e4+4*np.sum(C*e31,axis=1)+2*(e22@v)
go=4*(v[:,None]*e31+v[None,:]*e31.T)+8*C*e22;np.fill_diagonal(go,0)
gradient=max(float(np.max(np.abs(gd))),float(np.max(np.abs(go))))
assert gradient<1e-10
p=dict(np.load(HERE/'assessment.npz',allow_pickle=False))
t=dict(np.load(CAP/'values.npz',allow_pickle=False))
assert hashlib.sha256((HERE/'assessment.npz').read_bytes()).hexdigest()==assessment['output_sha256']

def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a.flat,b.flat))

discrepancy=0.
for name in ('mean','covariance','next_mean'):
 ta,tb,pp=t[f'target_A_{name}'],t[f'target_B_{name}'],p[name]
 s=dot(ta,tb);e=dot(ta-pp,tb-pp)
 values=dict(baseline_replica_cross_energy=s,residual_replica_cross_energy=e,signal_residual_ratio=e/s,
             ordinary_prediction_targetA=dot(pp-ta,pp-ta)/dot(ta,ta),ordinary_prediction_targetB=dot(pp-tb,pp-tb)/dot(tb,tb))
 for key,value in values.items():discrepancy=max(discrepancy,abs(value-assessment['consumers'][name][key])/abs(value))
assert discrepancy<1e-12
record=dict(passed=True,physical_normal_gradient_max=gradient,math_fsum_score_max_relative=discrepancy,
            assessment_sha256=hashlib.sha256((HERE/'assessment.json').read_bytes()).hexdigest())
with out.open('x') as o:json.dump(record,o,indent=2);o.write('\n')
print(json.dumps(record,indent=2),flush=True)
