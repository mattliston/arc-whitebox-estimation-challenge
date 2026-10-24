"""One fixed pair-tensor membership arm, saved historical joins, no labels."""
import gc,json,re,resource,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
LAB=R.parent/'opus_fix336687_v1/joinsketch'
OUT=R/'TENSOR_MEMBERSHIP_RESULT_v1.json'; assert not OUT.exists()
def guard():
    text=subprocess.check_output(['memory_pressure'],text=True)
    free=int(re.search(r'System-wide memory free percentage: (\d+)%',text).group(1))
    assert free>=50,free
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss<1_000_000_000,rss
    return free
def run(rank,layer):
    f0=guard(); kc={7:2048,13:1536}[layer]
    z=np.load(LAB/f'join{rank}_L{layer}.npz')
    J=z['J'].astype('f'); lam=z['lam'].astype('f');w=lam*lam
    old=int(z['old_columns'])
    dg=(J*J)@w
    if old:
        Qp,core,F=(z[k].astype('f') for k in ('Qp','core','F'))
        Qc=Qp@core;dg+=np.sum(Qc*Qp,axis=1)
        X=np.concatenate((Qp@F,J),axis=1)
    else:
        X=J
    ri=np.sort(np.argsort(-dg.astype('d'),kind='stable')[:384])
    oldY=Qc@Qp[ri].T if old else np.zeros((J.shape[0],384),dtype='f')
    cn=np.sum(J*J,axis=0)*w
    splits=np.flatnonzero(np.sign(lam[1:])!=np.sign(lam[:-1]))+1
    widths=np.diff(np.r_[0,splits,len(lam)]);assert len(widths)%2==0
    assert np.all(widths[::2]==widths[1::2]); starts=np.r_[0,np.cumsum(widths)]
    ip=np.concatenate([np.arange(starts[c],starts[c+1]) for c in range(0,len(widths),2)])
    im=np.concatenate([np.arange(starts[c],starts[c+1]) for c in range(1,len(widths),2)])
    assert np.array_equal(lam[ip],-lam[im])
    pairenergy=cn[ip]+cn[im]
    P=J[:,ip].astype('d');M=J[:,im].astype('d')
    np2=np.einsum('ij,ij->j',P,P);nq2=np.einsum('ij,ij->j',M,M)
    dot=np.einsum('ij,ij->j',P,M)
    tensor=lam[ip].astype('d')**2*(np2**3+nq2**3-2*dot**3)
    assert np.min(tensor)>=0
    del P,M
    pk=np.argsort(-pairenergy.astype('d'),kind='stable')[:kc//2]
    tk=np.argsort(-tensor,kind='stable')[:kc//2]
    cols=dict(current_jslite=np.argsort(-cn.astype('d'),kind='stable')[:kc],
      paired_energy=np.sort(np.r_[ip[pk],im[pk]]),
      paired_tensor=np.sort(np.r_[ip[tk],im[tk]]))
    L=np.concatenate((z['lam_old'].astype('d'),lam.astype('d')))
    W=z['w32'][z['previous']].T[::16].astype('d')
    # Signed full-content target; no observations or endpoint truth loaded.
    X64=X.astype('d');exact=W@X64;truth=(exact*exact*L)@exact.T
    arms={}
    for name,c in cols.items():
        Jc=J[:,c]
        Y=Jc@(w[c,None]*Jc[ri].T)+oldY
        Q=np.linalg.qr(Y)[0].astype('d')
        pushed=(W@Q)@(Q.T@X64)
        plane=(pushed*pushed*L)@pushed.T
        e=plane-truth;k3=float(np.sum(np.diag(e)**2));np.fill_diagonal(e,0)
        arms[name]=dict(cp_error_squared=float(np.sum(e*e)),k3_error_squared=k3)
    base=arms['current_jslite']
    for a in arms.values():
        a['cp_ratio']=a['cp_error_squared']/base['cp_error_squared']
        a['k3_ratio']=a['k3_error_squared']/base['k3_error_squared']
    row=dict(rank=rank,layer=layer,m=J.shape[0],K=J.shape[1],kc=kc,arms=arms,
      tensor_vs_current_column_overlap=len(np.intersect1d(cols['paired_tensor'],cols['current_jslite']))/kc,
      tensor_vs_energy_pair_overlap=len(np.intersect1d(tk,pk))/(kc//2),
      current_column_energy_vs_tensor_score_correlation=float(np.corrcoef(pairenergy,tensor)[0,1]),
      extra_dot_arithmetic_estimate=int(J.shape[0]*J.shape[1]),
      note_cost='Approximate pairing dot arithmetic only; norm2 is already present. Gather, float64 and request costs unpriced.',
      free_start=f0,free_end=guard())
    print(json.dumps(row),flush=True)
    return row
t0=time.monotonic();rows=[]
for rank in (35,200):
    rows.append(run(rank,7));gc.collect()
stop=all(r['arms']['paired_tensor']['cp_ratio']>1.05 for r in rows)
if not stop:
    for rank in (35,200):
        rows.append(run(rank,13));gc.collect()
out=dict(status='STOP_L7_BOTH_CP_WORSE_OVER5PCT' if stop else 'FIXED_SAVED_PROBE_COMPLETE',rows=rows,
  rule='L7 ranks35/200 first; stop if both paired-tensor CP squared errors >1.05x current JSK-lite; otherwise same fixed L13 ranks.',
  caveat='Historical pre-JSK input states, regenerated current selection. No labels, native endpoint, adjusted score, or candidate.',
  elapsed_s=time.monotonic()-t0,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,banked_gain=0)
with OUT.open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items()if k!='rows'}),flush=True)
