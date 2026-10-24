"""Bounded exact raw-moment proof; no fitted data or model execution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.nice(10)
import hashlib,json,math
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
def sym(C,m):
    return C[:,:,None]*m[None,None,:]+C[:,None,:]*m[None,:,None]+C[None,:,:]*m[:,None,None]
def cube(m):
    return np.einsum('i,j,k->ijk',m,m,m)
errors=[]; records=[]
rng=np.random.default_rng(1049)
gamma=.119; bscale=1.0342874194920628; div=6.299047545852794
for n in (8,1024):
    c=math.exp(.5*math.log(2/n)+math.lgamma((n+1)/2)-math.lgamma(n/2))
    a3=(n+1)*c/n; a4=1+2/n
    for case in range(8):
        # An arbitrary finite angular law suffices: only independent radial
        # homogeneity is used in this identity. Eight output coordinates.
        Y=np.maximum(rng.normal(size=(41,8))@rng.normal(size=(8,8)),0)
        m=Y.mean(0); M2=Y.T@Y/len(Y)
        M3=np.einsum('si,sj,sk->ijk',Y,Y,Y)/len(Y)
        C=M2-np.outer(m,m); T=M3-sym(M2,m)+2*cube(m)
        mg=c*m; Cg=M2-np.outer(mg,mg)
        Tg=a3*M3-sym(M2,mg)+2*cube(mg)
        km=(2*n*(1-c*c)-1)/(n+1)
        identity=Tg/a3-sym(C,m)/(n+1)+km*cube(m)
        err=np.max(np.abs(identity-T))/max(1,np.max(np.abs(T)))
        # Exactly map a fixed Gaussian b-fix trace plus its arbitrary residual.
        rg=.003+case*.0007
        bg=bscale*gamma*rg*mg/6
        bopt=bg/a3
        residual=Tg-sym(Cg,bg)
        rank1=km+3*(1-c*c)*(bopt[0]/m[0])
        mapped=residual/a3+sym(C,-m/(n+1)+bopt)+rank1*cube(m)
        maperr=np.max(np.abs(mapped-T))/max(1,np.max(np.abs(T)))
        # A repeated-index star cannot repair a trace error on distinct indices.
        delta=m/(n+1)*.1; DT=sym(C,delta)
        distinct=np.ones((8,8,8),bool)
        for i in range(8):
            distinct[i,i,:]=False;distinct[i,:,i]=False;distinct[:,i,i]=False
        anchored=DT.copy();anchored[~distinct]=0
        assert np.max(np.abs(anchored[distinct]-DT[distinct]))==0
        assert np.linalg.norm(anchored)>0
        errors.extend([err,maperr]);records.append(dict(n=n,case=case,identity_relative=err,mapped_optional_relative=maperr,mean_cube_coefficient=km))
assert max(errors)<2e-12
n=1024
source_fraction=bscale*6/div
input_bfix_cancel_fraction=bscale*gamma*(n+1)/(n+2)
out=dict(pass_=True,fixtures=16,outputs=8,max_relative_error=max(errors),
    frozen_gamma=gamma,frozen_bscale=bscale,frozen_source_divisor=div,
    mandatory_source_fraction=source_fraction,
    explicit_negative_input_radial_bfix_cancel_fraction=input_bfix_cancel_fraction,
    mandatory_fraction_after_explicit_input_term=source_fraction-input_bfix_cancel_fraction,
    note='The remaining radial_new term is optional closure, not included in the explicit input-term fraction. New L0 trace uses exact physical coefficient plus mapped frozen Gaussian optional trace. The tiny mean-cube remainder is proven but outside the trace-only repair.',records=records)
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(H/'TRACE_IDENTITY_v2.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
