"""Preregistered d2 constant/weightedmean exact-inner control variate."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import ast,json,time,resource,gc
import numpy as np
D=Path(__file__).resolve().parent;P=D.parent/'paired_column_sample_v1'

def funcs(path,names,ns):
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)

class Hold(list):
    def __call__(self,x):self.append(x);return x

CAP=D.parent.parent/'astra_oct18_join_trace_b_v1/capture35_200_v3'
ns=dict(CAP=CAP,np=np,json=json)
funcs(P/'probe_v1.py',{'load'},ns)
helpers=dict(fnp=np,_E2Hold=Hold)
funcs(P/'fragment_pps_half_L11_14_HELD_v2.py',{'_pcs18_select','_pcs18_score_chunk'},helpers)

def orth(columns):
    basis=[]
    for x in columns.T:
        y=x.copy();norm=np.linalg.norm(y)
        for _ in range(2):
            for q in basis:y-=q*(q@y)
        yn=np.linalg.norm(y)
        if yn>np.finfo(float).eps*max(columns.shape)*norm:basis.append(y/yn)
    return np.column_stack(basis)

def run(rank):
    tree=ns['load'](rank);before=tree['before'];after=tree['after'];state=after['state']
    pp=after['pp'];old=int(before['state']['columns']);lam=before['lam'].astype('d')
    rr=before['response_indices'].astype(int);cc=before['live_indices'].astype(int)
    base=state['uvp'][1];Fu,Fv,blocks,ru=base
    Fu=Fu.astype('d');Fv=Fv.astype('d');r=int(Fv.shape[0]);pairs=int(Fv.shape[1])
    V=np.empty((r,old));plus=[];minus=[];offset=0
    for a,h in blocks:
        a,h=int(a),int(h);u=np.zeros((r,h));u[:ru]=Fu[:,offset:offset+h];v=Fv[:,offset:offset+h]
        V[:,a:a+h]=u+v;V[:,a+h:a+2*h]=u-v;offset+=h
        plus.extend(range(a,a+h));minus.extend(range(a+h,a+2*h))
    plus=np.asarray(plus);minus=np.asarray(minus)
    U=np.zeros((1024,r));U[before['output_rows'].astype(int)]=state['pre_basis'].astype('d')
    if state['lazy'] is not None:U=U@state['lazy']['map'].astype('d')
    Ur,Uc=U[rr],U[cc]
    X=np.asarray(pp[np.ix_(rr,np.arange(old))],dtype='d');mean_r=X@np.abs(lam[:old]);left=X*X;del X
    Y=np.asarray(pp[np.ix_(cc,np.arange(old))],dtype='d');mean_c=Y@np.abs(lam[:old]);del Y
    scores=np.concatenate([helpers['_pcs18_score_chunk'](left,lam[:old],pp,cc,int(a),int(h)) for a,h in blocks])
    selected,weight=helpers['_pcs18_select'](scores,pairs//2)
    ip,im=plus[selected],minus[selected]
    A=(left*lam[:old])@V.T
    S=(left[:,ip]*(lam[ip]*weight))@V[:,ip].T+(left[:,im]*(lam[im]*weight))@V[:,im].T
    E=A-S
    # Fixed before outcomes; no unavailable mu/gate was reconstructed or guessed.
    # Means fixed above; avoid retaining full old/live column matrices.
    Q=orth(np.column_stack((np.ones(len(rr)),mean_r)))
    R=np.column_stack((np.ones(len(cc)),mean_c))
    Z=orth(Uc.T@R)
    QE=Q@(Q.T@E);EZ=(E@Z)@Z.T
    correction=QE+EZ-Q@(Q.T@EZ)
    residual=E-correction
    native_old=A@Uc.T;delta0=-E@Uc.T;delta1=-residual@Uc.T
    # All young/recent columns untouched. Native old rightframe replaces the
    # actualpp right in this denominator, matching the proposed inner hook.
    young=np.zeros_like(native_old);actual_old=np.zeros_like(native_old)
    for start in range(0,len(lam),512):
        stop=min(start+512,len(lam));cols=np.arange(start,stop)
        xb=np.asarray(pp[np.ix_(rr,cols)],dtype='d');yb=np.asarray(pp[np.ix_(cc,cols)],dtype='d')
        # Split the one block straddling the old boundary, without changing sums.
        cut=max(0,min(stop,old)-start)
        if cut:actual_old+=(xb[:,:cut]**2*lam[start:start+cut])@yb[:,:cut].T
        if cut<stop-start:young+=(xb[:,cut:]**2*lam[start+cut:stop])@yb[:,cut:].T
        del xb,yb
    total=native_old+young
    actual_total=actual_old+young
    mask=rr[:,None]!=cc[None,:]
    def score(delta):
        fixed=np.where(mask,delta,0.)
        return dict(relative_total_cp_offdiag=float(np.linalg.norm(delta[mask])/np.linalg.norm(total[mask])),
                    relative_total_cp_exact_diagonal=float(np.linalg.norm(fixed)/np.linalg.norm(total)))
    result=dict(rank=rank,layer=13,rows=len(rr),cols=len(cc),old_columns=old,
        basis='d2 constant and oldcolumn abs(lambda)-weightedmean; coreZ=orth(Ulive.T@rightdirections)',
        row_rank=Q.shape[1],core_rank=Z.shape[1],before=score(delta0),after=score(delta1),
        error_amplitude_recovered_pct=float(100*(1-np.linalg.norm(delta1[mask])/np.linalg.norm(delta0[mask]))),
        inner_error_amplitude_recovered_pct=float(100*(1-np.linalg.norm(residual)/np.linalg.norm(E))),
        row_constraint_relative=float(np.linalg.norm(Q.T@residual)/np.linalg.norm(E)),
        core_constraint_relative=float(np.linalg.norm(residual@Z)/np.linalg.norm(E)),
        native_vs_actual_pp_cp_relative=float(np.linalg.norm(total-actual_total)/np.linalg.norm(actual_total)),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert result['peak_rss_bytes']<1_000_000_000
    print(json.dumps(result),flush=True)
    return result

start=time.monotonic();rows=[]
for rank in (35,200):rows.append(run(rank));gc.collect()
result=dict(status='FIXED_D2_SAVED_GATE',rows=rows,elapsed_s=time.monotonic()-start,
    models=0,sdk_calls=0,required_error_amplitude_recovery_pct=85,
    limitations=['Saved rawCP only; no foldedNP weights captured.','No mu/gate captured, so this is the approvedconstant/weightedmean diagnostic, not derivativebasis.',
      'No basisgrid or coefficientsearch.','Qualifiedroute notcaptured: existingfastrescue activated in sourcecapture.',
      'Controlvariate computedonfullresponse/liveplanes; exactdiagonal retained. No endpointMSEinference.'])
(D/'RESULT_D2_v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'elapsed_s':result['elapsed_s']}))
