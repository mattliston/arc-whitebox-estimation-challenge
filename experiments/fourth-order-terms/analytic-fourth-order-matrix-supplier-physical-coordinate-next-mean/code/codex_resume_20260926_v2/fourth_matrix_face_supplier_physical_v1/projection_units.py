"""Signed physical-metric projection against explicit least squares."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fourth_matrix_face_capacity_v1'))
from projection import vector
from physical_projection import faces,project
out=HERE/'projection_units.json';assert not out.exists()
rng=np.random.default_rng(961557);records=[]
for kind in ('positive','signed','wide_scales'):
 n=6;B=rng.normal(size=(n,n));C=B@B.T/8
 if kind=='signed':C[0,0]*=-1;C[2,2]*=-1
 if kind=='wide_scales':
  C[0,:]=C[:,0]=0;C[0,0]=-1e-6
  C[1,:]=C[:,1]=0;C[1,1]=1e-4
 entries=[(i,j) for i in range(n) for j in range(i,n)]
 D0=rng.normal(size=(n,n));D0=(D0+D0.T)/2
 Dr,_=project(C,*faces(C,D0))
 design=[]
 for i,j in entries:
  E=np.zeros_like(C);E[i,j]=E[j,i]=1
  design.append(vector(*faces(C,E)))
 L=np.column_stack(design)
 k=rng.normal(size=n);q31=rng.normal(size=(n,n));q22=rng.normal(size=(n,n));q22=(q22+q22.T)/2
 target=(k,q31,q22);D,info=project(C,*target)
 x,*_=np.linalg.lstsq(L,vector(*target),rcond=None)
 actual=np.array([D[i,j] for i,j in entries])
 lsrelative=float(np.linalg.norm(x-actual)/np.linalg.norm(x))
 recovery=float(np.max(np.abs(Dr-D0)))
 normal=float(np.max(np.abs(L.T@(L@actual-vector(*target)))))
 assert max(lsrelative,recovery,normal)<1e-8,(kind,lsrelative,recovery,normal)
 records.append(dict(kind=kind,explicit_lstsq_relative=lsrelative,recovery_max=recovery,normal_equation_max=normal,solve=info))
with out.open('x') as o:json.dump(dict(passed=True,records=records),o,indent=2);o.write('\n')
print(json.dumps(dict(passed=True,records=records),indent=2),flush=True)
