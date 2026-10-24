import gc,hashlib,resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from feature import *
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
metadata=read(HERE/'inputs.json')['rows'];rows=[];first=None;start=time.monotonic()
for call,row in enumerate([metadata[0],metadata[1],metadata[0]]):
    check(row['capture']);check(row['parent_prediction'])
    with np.load(row['capture']['path'],allow_pickle=False) as f:cap={k:f[k] for k in f.files}
    digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in cap.items()}
    assert digests=={k:v['sha256'] for k,v in row['array_digests'].items()}
    assert cap['prediction'].tobytes()==np.load(row['parent_prediction']['path'],allow_pickle=False).tobytes()
    B,K=planes(cap);Z=summaries(B,K,cap['sigma'],cap['alpha'])
    X=cap['sigma'][:,None]*np.column_stack([np.ones(1024),cap['H'].astype(float)])
    assert Z.shape==(1024,8) and X.shape==(1024,65) and np.isfinite(Z).all()
    diag_error=relative(np.diag(K),cap['k3']);variance_error=relative(np.diag(B),cap['sigma']**2)
    assert max(diag_error,variance_error)<=1e-3,(diag_error,variance_error)
    # Separate direct sums for eight fixed output rows/columns.
    ix=np.arange(8)*127;W=cap['W'].astype(float);A=cap['A'].astype(float)
    V=W[:,ix].T;U=(A.T@W[:,ix]).T
    C=cap['C'].astype(np.float32).astype(float);C=(C+C.T)*.5
    Bref=V@(C@V.T);kslice=np.zeros((8,8));P=cap['P'][ix].astype(float);lam=cap['lam'].astype(float);b=cap['b'][ix]
    for i in range(8):
        for j in range(8):
            cp=np.sum(P[i]*P[i]*P[j]*lam,dtype=np.float64)
            star=np.sum(2*U[i]*V[i]*V[j]+V[i]*V[i]*U[j],dtype=np.float64)/3
            kslice[i,j]=cp+star+Bref[i,i]*b[j]+2*Bref[i,j]*b[i]
    slice_error=relative(K[np.ix_(ix,ix)],kslice);cov_slice_error=relative(B[np.ix_(ix,ix)],Bref)
    assert max(slice_error,cov_slice_error)<=2e-10,(slice_error,cov_slice_error)
    assert digests=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in cap.items()}
    arrays=dict(X=X,Z=Z,prediction=cap['prediction'],formal_k3=np.diag(K).copy(),native_k3=cap['k3'],
        formal_variance=np.diag(B).copy(),native_variance=cap['sigma']**2)
    output_digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
    if call==0:first=output_digests
    if call==2:assert output_digests==first;dest=Path(rows[0]['file']['path'])
    else:
        dest=HERE/f"features_rank{row['rank']:04d}.npz";np.savez(dest,**arrays)
    records=dict(call=call,rank=row['rank'],file=pin(dest),array_digests=output_digests,
        original_capture=row['capture'],parent_cold_flops=row['parent_cold_flops'],
        diagonal_relative=diag_error,variance_relative=variance_error,slice_relative=slice_error,
        covariance_slice_relative=cov_slice_error,input_preserved=True)
    rows.append(records);print({k:v for k,v in records.items() if k not in ['array_digests','original_capture','file']},flush=True)
    del cap,B,K,Z,X,arrays,P,lam,b,V,U,A,W,C,Bref;gc.collect()
write(HERE/'features_result.json',dict(passed=True,rows=rows,names=NAMES,repeat_bytes_exact=True,
    targets_not_decoded=True,new_native_calls=0,wall_s=time.monotonic()-start,
    production_cost_measured=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
write(HERE/'features_sealed.json',dict(pins=[pin(HERE/'features_result.json')]+[r['file'] for r in rows[:2]],targets_not_decoded=True))
