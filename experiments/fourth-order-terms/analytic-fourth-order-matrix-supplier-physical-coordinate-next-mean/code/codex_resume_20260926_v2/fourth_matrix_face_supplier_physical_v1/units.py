"""Supplier algebra qualification using synthetic states only."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'astra_phase2_reexamination_v1'))
from supplier import make_supplier
from current_pair_oracle_math_v1 import independent_fourth_map

out=HERE/'units.json';assert not out.exists()
new=make_supplier();old=independent_fourth_map()
rng=np.random.default_rng(541826)
n=5
B=rng.normal(size=(n,n+4));C=B@B.T
C=C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
np.fill_diagonal(C,1)
K21=rng.normal(size=(n,n))*.001
k4=rng.normal(size=n)*.002
k22=rng.normal(size=(n,n))*.001;k22=(k22+k22.T)/2;np.fill_diagonal(k22,k4)
q31=rng.normal(size=(n,n))*.001;np.fill_diagonal(q31,0)
mu=rng.normal(size=n)
a=new(mu,C,K21,k4,k22,'covariance',q31_override=q31)
b=old(mu,C,K21,k4,k22,'covariance',q31_override=q31)
parity=[float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(a[:4],b[:4])]
assert max(parity)==0
positive=new(np.full(n,12.),C,K21,k4,k22,'covariance',q31_override=q31)
linear31=positive[4]['post31'].copy();np.fill_diagonal(linear31,0)
linear_error=float(np.max(np.abs(linear31-q31)))
assert linear_error<1e-9,linear_error
independent=new(mu,np.eye(n),np.zeros((n,n)),np.zeros(n),np.zeros((n,n)),'covariance')
ind=independent[4]
p31=ind['post31'].copy();np.fill_diagonal(p31,0)
p22=ind['post22'].copy();np.fill_diagonal(p22,0)
ind_error=max(float(np.max(np.abs(p31))),float(np.max(np.abs(p22))))
diag_error=max(float(np.max(np.abs(np.diag(ind['post31'])-ind['post_k4']))),
               float(np.max(np.abs(np.diag(ind['post22'])-ind['post_k4']))))
assert max(ind_error,diag_error)<1e-12
record=dict(passed=True,parent_output_max=parity,linear_region_q31_max=linear_error,
            independent_gaussian_offdiagonal_max=ind_error,scalar_diagonal_max=diag_error,
            challenge_arrays_accessed=False)
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2),flush=True)
