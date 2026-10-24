"""Two fixed paired-column sampling rules on existing raw CP captures only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json, time, resource
import numpy as np
D=Path(__file__).resolve().parent
CAP=D.parent.parent/'astra_oct18_join_trace_b_v1/capture35_200_v3'
assert json.loads((CAP/'GUARD_v1.json').read_text())['whole_group_retired']

def load(rank):
    path=CAP/f'capture_rank{rank}_L13'
    meta=json.loads((path/'arrays.json').read_text())
    def arr(key):
        a=meta['arrays'][key]
        return np.memmap(path/a['file'],mode='r',dtype=a['dtype'],shape=tuple(a['shape']))
    def decode(v):
        if isinstance(v,dict):
            if 'array' in v:return arr(v['array'])
            if 'alias' in v:return arr(v['alias'])
            return {k:decode(x) for k,x in v.items()}
        if isinstance(v,list):return [decode(x) for x in v]
        return v
    return decode(meta['tree'])

def pps(score, count):
    # A random uniform phase would give inclusion probability p exactly.
    # This experiment fixes phase=.5 before reading any fidelity result.
    score=np.maximum(score, np.max(score)*1e-15)
    lo,hi=0.,count/np.min(score)
    for _ in range(80):
        mid=(lo+hi)/2
        if np.minimum(1.,mid*score).sum()<count:lo=mid
        else:hi=mid
    p=np.minimum(1.,hi*score)
    selected=np.searchsorted(np.cumsum(p),np.arange(count)+.5)
    assert len(np.unique(selected))==count
    return selected,1./p[selected],p

def run(rank):
    tree=load(rank);before=tree['before'];after=tree['after']
    old=int(before['state']['columns']);pp=after['pp'];lam=before['lam'].astype('d')
    pairs=[]
    for start,width in after['state']['cohorts']:
        start,width=int(start),int(width)
        if width and start+width<=old:
            half=width//2
            pairs.extend(zip(range(start,start+half),range(start+half,start+width)))
    plus,minus=np.asarray(pairs,dtype=int).T;P=len(plus)
    assert P*2==old and P%2==0
    assert np.array_equal(lam[minus],-lam[plus])
    rr=before['response_indices'].astype(int);cc=before['live_indices'].astype(int)
    # Fixed score: exact pair-contribution Frobenius norm on the whole raw
    # response-by-live block, using column dot products rather than full planes.
    xp=np.asarray(pp[np.ix_(rr,plus)],dtype='d');xm=np.asarray(pp[np.ix_(rr,minus)],dtype='d')
    ap=xp*xp;am=xm*xm
    aa=(np.sum(ap*ap,axis=0),np.sum(am*am,axis=0),np.sum(ap*am,axis=0))
    del xp,xm,ap,am
    bp=np.asarray(pp[np.ix_(cc,plus)],dtype='d')*lam[plus]
    bm=np.asarray(pp[np.ix_(cc,minus)],dtype='d')*lam[minus]
    norm2=aa[0]*np.sum(bp*bp,axis=0)+aa[1]*np.sum(bm*bm,axis=0)+2*aa[2]*np.sum(bp*bm,axis=0)
    score=np.sqrt(np.maximum(norm2,0));del bp,bm,aa
    weighted,invp,prob=pps(score,P//2)
    rules=[('uniform_even',np.arange(P)[::2],np.full(P//2,2.)),('pair_contribution_PPS_phase_half',weighted,invp)]
    outcomes=[]
    for label,rowids,colids in [('response_square',rr[::16],rr[::16]),('response_live_rectangle',rr[::16],cc[::16])]:
        X=np.asarray(pp[rowids],dtype='d');Y=np.asarray(pp[colids],dtype='d')
        total=(X*X*lam)@Y.T
        oldterm=(X[:,:old]**2*lam[:old])@Y[:,:old].T
        mask=rowids[:,None]!=colids[None,:]
        tn=np.linalg.norm(total[mask]);on=np.linalg.norm(oldterm[mask])
        for rule,chosen,weight in rules:
            ip,im=plus[chosen],minus[chosen]
            sample=(X[:,ip]**2*(lam[ip]*weight))@Y[:,ip].T+(X[:,im]**2*(lam[im]*weight))@Y[:,im].T
            delta=sample-oldterm
            # Exact diagonal is retained independently by the estimator core.
            fixed=np.where(mask,delta,0.)
            outcomes.append(dict(block=label,rule=rule,sample_pairs=len(chosen),pairs=P,
                total_cp_offdiag_relative_fro=float(np.linalg.norm(delta[mask])/tn),
                total_cp_full_relative_fro_exact_diagonal=float(np.linalg.norm(fixed)/np.linalg.norm(total)),
                old_cp_offdiag_relative_fro=float(np.linalg.norm(delta[mask])/on),
                old_fraction_total_offdiag=float(on/tn),
                max_abs_offdiag=float(np.max(np.abs(delta[mask]))),rows=len(rowids),cols=len(colids)))
    return dict(rank=rank,layer=13,old_columns=old,total_columns=pp.shape[1],pairs=P,
        full_score_rows=len(rr),full_score_columns=len(cc),score_rule='Raw pair K21 Frobenius contribution on whole response-by-live block; no labels',
        pps_inclusion_min=float(prob.min()),pps_inclusion_max=float(prob.max()),
        pps_certainty_pairs=int(np.sum(prob>=1-1e-12)),results=outcomes)

start=time.monotonic();rows=[run(35),run(200)]
result=dict(status='SAVED_RAW_CP_TWO_FIXED_HALF_PAIR_RULES',models=0,sdk_calls=0,rows=rows,
    elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    limitations=['Saved QX and folded old_left absent; rawCP fidelity only, NP-fold fidelity unknown.',
      'Captures use current95c source but existing fastrescue activated; no qualified-route transfer claim.',
      'PPS inverse-inclusion weights are unbiased over uniform phase; this single deterministic phase is not itself an expectation.',
      'Scores measured from full raw response/live columns, while fidelity uses fixed every16th rows; no endpoint labels or parameter search.',
      'Reconstruction, exact k3, young/recent columns, and NP state stay unchanged in the proposed source hook.'])
(D/'RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
