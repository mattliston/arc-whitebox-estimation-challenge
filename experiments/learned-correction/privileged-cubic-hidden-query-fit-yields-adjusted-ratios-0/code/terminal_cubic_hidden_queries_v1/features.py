import gc,hashlib,json,resource,time
from common import *
from helper import cubic_hidden_features
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();verify('captures_sealed.json')
    recs=read(HERE/'capture/result.json')['records'];rows=[];first=None;start=time.monotonic()
    for row in recs:
        with np.load(row['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
        args={k:cap[k] for k in ('P','lam','A','C','W','b','H','sigma','alpha','pdf')}
        before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in args.items()}
        with flops.BudgetContext(flop_budget=2**35,quiet=True) as ctx:
            aa={k:fnp.asarray(v) for k,v in args.items()};values=cubic_hidden_features(fnp,**aa)
        actual=[np.asarray(v).copy() for v in values];bill=ctx.summary_dict()['flops_used'];assert bill<=20_000_000_000,bill
        assert before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in args.items()}
        assert actual[0].shape==(1024,65) and all(np.isfinite(v).all() for v in actual)
        # Independent explicit matrix owners/association in FP64.
        P=cap['P'].astype(np.float64);lam=cap['lam'].astype(np.float64);V=cap['W'].astype(np.float64).T;U=V@cap['A'].astype(np.float64)
        C=cap['C'].astype(np.float32).astype(np.float64);B=V@((C+C.T)*.5)@V.T
        X=np.column_stack([np.ones(1024),cap['H'].astype(np.float64)]);norm=np.linalg.norm(X,axis=0);Q=X/np.maximum(norm,1e-12)[None,:];r=Q/cap['sigma'][:,None]
        pr=P.T@r;vr=V.T@r;ur=U.T@r;br=B@r;b=cap['b']
        query=(P*lam[None,:])@(pr*pr)+(U@(vr*vr)+2*V@(vr*ur))/3+b[:,None]*np.einsum('ic,ic->c',r,br)[None,:]+2*br*(b@r)[None,:]
        ref=(-cap['alpha']*cap['pdf']/6)[:,None]*query
        relative=float(np.linalg.norm(actual[0]-ref)/max(np.linalg.norm(ref),1e-30));assert relative<=2e-10,relative
        formal=np.sum(P*P*P*lam[None,:],axis=1)+np.sum(U*V*V,axis=1)+3*b*np.diag(B)
        diag_relative=float(np.linalg.norm(formal-cap['k3'])/max(np.linalg.norm(cap['k3']),1e-30));assert diag_relative<=1e-3,diag_relative
        affine=cap['sigma'][:,None]*X
        arrays=dict(feature=actual[0],query=actual[1],directions=actual[2],norms=actual[3],affine=affine,prediction=cap['prediction'],formal_diagonal=formal,native_diagonal=cap['k3'])
        digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        if row['call']==0:first=(digests,bill)
        if row['call']==2:assert first==(digests,bill)
        dest=HERE/f"features_call{row['call']:02d}_r{row['rank']:04d}.npz";np.savez(dest,**arrays)
        rec=dict(call=row['call'],rank=row['rank'],file=pin(dest),helper_flops=bill,parent_flops=row['flops'],reference_relative=relative,formal_diagonal_relative=diag_relative,input_preserved=True,array_digests=digests);rows.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k!='array_digests'}),flush=True)
        del aa,values,actual,cap,args,P,V,U,B,C,query,ref,arrays;gc.collect()
    write(HERE/'features_result.json',dict(passed=True,rows=rows,targets_opened=False,repeat_bytes_and_bill_exact=True,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
