"""Tiny synthetic-only paired cubic mode-1 Gram proof; no candidate/model imports."""
import json,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent

def factors(a,b,lam):
    x=np.sum(a*a,axis=0);y=np.sum(b*b,axis=0);dot=np.sum(a*b,axis=0)
    cross=np.minimum(dot*dot,x*y) # Cauchy roundoff clamp only
    s=np.sqrt(np.maximum(x*y-cross,0)*(x*y+cross))
    den=np.sqrt(x*x+y*y+2*s)
    inv=np.divide(np.abs(lam),den,out=np.zeros_like(den),where=den>0)
    aa=(x*x+s)*inv;ab=-cross*inv;bb=(y*y+s)*inv
    return a*aa+b*ab,a*ab+b*bb

rng=np.random.default_rng(20261014);n=7;p=4
a=rng.normal(size=(n,p));b=rng.normal(size=(n,p));lam=np.array([.3,-.7,0.,1.2])
cases=[('general',a,b),('equal',a,a.copy()),('opposite',a,-a),('zero',np.zeros_like(a),np.zeros_like(a)),('near_parallel',a,a+1e-7*b)]
checks=[]
for name,a,b in cases:
    za,zb=factors(a,b,lam);metric=za@za.T+zb@zb.T
    truth=np.zeros((n,n));expanded=np.zeros_like(truth);hmin=0.
    for j in range(p):
        x,y=a[:,j],b[:,j];t=lam[j]*(np.einsum('i,j,k->ijk',x,x,x)-np.einsum('i,j,k->ijk',y,y,y))
        truth+=t.reshape(n,-1)@t.reshape(n,-1).T
        c=(x+y)/2;h=(x-y)/2;C=c@c;H=h@h;d=c@h
        expanded+=lam[j]**2*((8*C*H+8*d*d)*np.outer(c,c)+(4*(C*C+H*H)+8*d*d)*np.outer(h,h)+8*(C+H)*d*(np.outer(c,h)+np.outer(h,c)))
        block=lam[j]**2*np.array([[(x@x)**2,-(x@y)**2],[-(x@y)**2,(y@y)**2]])
        hmin=min(hmin,float(np.linalg.eigvalsh(block).min()))
    scale=max(float(np.linalg.norm(truth)),1.)
    err=float(np.linalg.norm(metric-truth)/scale);expanderr=float(np.linalg.norm(expanded-truth)/scale)
    assert err<1e-12 and expanderr<1e-12 and hmin>-1e-10,(name,err,expanderr,hmin)
    checks.append(dict(case=name,relative_tensor_gram_error=err,relative_ch_expansion_error=expanderr,min_H_eigenvalue=hmin))
# Original J remains the physical bank: sketch/core uses metric columns only.
a,b=cases[0][1:];za,zb=factors(a,b,lam);J=np.concatenate([a,b],axis=1);Z=np.concatenate([za,zb],axis=1);K=Z@Z.T
rows=np.sort(np.argsort(-np.diag(K),kind='stable')[:4]);Y=K[:,rows];Q,R=np.linalg.qr(Y,mode='reduced')
core=np.linalg.solve(Q[rows],R.T);core=(core+core.T)/2;Fj=Q.T@J
assert np.linalg.norm(Q@Fj-Q@Q.T@J)<1e-12
residual_min=float(np.linalg.eigvalsh(K-Q@core@Q.T).min());assert residual_min>-1e-9
# Bounded all-three-join arithmetic; 3 cohorts/join, <=1024 pairs/cohort.
m=1024;pairs_per_join=3072;total_pairs=3*pairs_per_join
helper_upper=14*m*total_pairs+80*total_pairs
old_diagonal_saved=2*m*total_pairs
old_selected_weight_product_saved=2*pairs_per_join*(384+320+384)
extra_upper=helper_upper-old_diagonal_saved-old_selected_weight_product_saved
out=dict(status='TINY_SOURCE_ONLY_PASS',checks=checks,original_Fj_projection_preserved=True,nystrom_residual_min_eigenvalue=residual_min,
    exact_scope='Sum of exact per-pair cubic mode-1 Grams, not the mode-1 Gram of the summed tensor: cross-pair/cohort contractions remain absent.',
    sqrt_rule='Principal2x2 sqrt via (H+sqrt(detH)I)/sqrt(traceH+2sqrt(detH)); float64 scalar norms; Cauchy roundoff clamp; zero block maps to zero.',
    old_core_limit='Transported old core is a frozen birth metric carried by congruence, not an exact current cubic-mode1 Gram after nonorthogonal W/gates. All joins must use the same declared proxy.',
    history=dict(nearby='physicist/RESULT.md:166-173 and t120_worker_pairsketch.py:94-103',old_pairk3_mse_pct=-.147,old_native_adjusted_measured=False,old_scope='c/h diagonal approximation; omits dot(c,h)^2 and cross c/h terms',not_duplicate='K4metric only changes K4L isotropic beta mean metric, torch_rowcuts_v15.py:223-230'),
    arithmetic_upper=dict(total_pairs=total_pairs,max_rows=m,helper_flops=helper_upper,diagonal_flops_saved=old_diagonal_saved,selected_weight_product_flops_saved=old_selected_weight_product_saved,extra_flops=extra_upper,pct_of_d401_136_bill=100*extra_upper/254201904929),
    cost_limits='Arithmetic ceiling only, excludes allocator/operation-request/fast-kernel effects; whole native bill required. Main Y/QR/core/Fj shapes unchanged. No extra old-F reconstruction.',
    minimal_integration='At active _astra_bcc_push_impl joins7/10/13 construct same-width paired Z from original J and exact cohort signedlambda; use sum(Z²) and Z@Z[rows].T for diagonal/Y, retain existing ascending selected-row-ID order, unchanged QR/Nystrom core solve and original Fj=Q.T@J; discard Z after Y. No JEO/UC32/rank changes.',
    numerical_calls=0,candidate_source_created=False)
p=D/'TINY_ALGEBRA_AND_COST_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],max_gram_error=max(x['relative_tensor_gram_error']for x in checks),extra_flops=extra_upper,arithmetic_pct=out['arithmetic_upper']['pct_of_d401_136_bill'],receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
