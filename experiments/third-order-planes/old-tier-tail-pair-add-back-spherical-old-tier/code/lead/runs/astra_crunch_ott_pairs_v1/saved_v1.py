"""Fixed128/256 signed-pair CP addback on archived OTT factors, no terminal targets."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent;O=R.parents[1];START=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pairs(F,coeff):
    cuts=np.r_[0,np.flatnonzero(np.sign(coeff[1:])!=np.sign(coeff[:-1]))+1,len(coeff)]
    pp=[];mm=[]
    for j in range(0,len(cuts)-1,2):
        a,b,c=map(int,cuts[j:j+3]);assert b-a==c-b and np.array_equal(coeff[a:b],-coeff[b:c])
        pp.extend(range(a,b));mm.extend(range(b,c))
    pp=np.array(pp);mm=np.array(mm);c=(F[:,pp]+F[:,mm])*.5;d=(F[:,pp]-F[:,mm])*.5
    ce=c.copy();ce[256:]=0;C=np.sum(ce*ce,axis=0);DD=np.sum(d*d,axis=0);cd=np.sum(ce*d,axis=0)
    en=coeff[pp]**2*(12*C*C*DD+4*DD**3+24*(C+DD)*cd*cd)
    fp=ce+d;fm=ce-d
    naive=coeff[pp]**2*(np.sum(fp*fp,axis=0)**3+np.sum(fm*fm,axis=0)**3-2*np.sum(fp*fm,axis=0)**3)
    assert np.isfinite(en).all() and np.all(en>=0)
    return pp,mm,c,d,ce,en,dict(relative_naive_vs_positive_energy=float(np.linalg.norm(naive-en)/np.linalg.norm(en)),naive_negative_count=int(sum(naive<0)))
def main():
    P=O/'engineer/a35_join/state/s39_rank0651.npz';A=O/'engineer/a36_carrier/state/s38_rank0651.npz'
    S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    ent=json.loads((S/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][651];wp=S/Path(ent['path']).parent/'weights.npy'
    z=np.load(P);J=z['J_13'].astype(float);lam=z['lam_13'].astype(float);Qp=z['Qp_13'].astype(float);core=z['core_13'].astype(float);prev=z['prev_13'];lo=z['lamold_12'].astype(float)
    saved=z['old_12'].astype(float);full=np.zeros((1024,saved.shape[1]));full[z['rows_12']]=saved;obs=(full*z['gate_12'][:,None])[prev];valid=np.abs(obs).sum(axis=1)>0
    oldF=np.linalg.lstsq(Qp[valid],obs[valid],rcond=None)[0];olderr=float(np.linalg.norm(Qp[valid]@oldF-obs[valid])/np.linalg.norm(obs[valid]));del full,saved,obs
    G=(J*(lam*lam)[None,:])@J.T+Qp@core@Qp.T;G=(G+G.T)*.5;ix=np.sort(np.argsort(-np.diag(G),kind='stable')[:384]);Q=np.linalg.qr(G[:,ix],mode='reduced')[0];del G
    Fj=Q.T@J;overlap=Q.T@Qp;W=np.load(wp,mmap_mode='r')[13,prev,:].astype(float);U=W.T@Q;Um=U@overlap;del J,W,Q,Qp,core
    blocks=[];base=np.zeros(1024);truth=np.zeros(1024);records=[]
    for bi,(name,X,F,coeff) in enumerate([('old_map',Um,oldF,lo),('new_join',U,Fj,lam)]):
        pp,mm,c,d,ce,en,proof=pairs(F,coeff);rho=float(np.mean(X*X));n2=np.sum(F*F,axis=0);v=F@(coeff*n2);base+=3*rho*(X@v)
        ac=X[:,:min(256,X.shape[1])]@ce[:min(256,X.shape[1])];ad=X@d
        cubes=coeff[pp]*(6*ac*ac*ad+2*ad*ad*ad);truth+=np.sum(cubes,axis=1)
        for i,e in enumerate(en):records.append((float(e*rho**3),bi,i))
        blocks.append(dict(name=name,X=X,F=F,coeff=coeff,pp=pp,mm=mm,c=c,d=d,ce=ce,energy=en,rho=rho,cubes=cubes,proof=proof))
        del ac,ad
    order=sorted(records,key=lambda e:-e[0]);totalenergy=sum(e[0] for e in order)
    aux=np.load(A)['aux_13'];alpha=aux[3];sigma=aux[2];pdf=aux[4];gate=aux[0];sens=-alpha*pdf/(6*sigma*sigma)
    score=np.where((alpha < -2.5)&(gate<.01),alpha+2.5,2.5-alpha);kept=min(1024,((np.count_nonzero(score>=0)+63)//64)*64);tail=np.argsort(-score,kind='stable')[kept:];band=np.flatnonzero((np.abs(alpha)>2.5)&(np.abs(alpha)<=3.))
    results=[]
    for budget in (128,256):
        selected=order[:budget];correction=np.zeros(1024);priced=[];stability=[]
        for bi,b in enumerate(blocks):
            ids=np.array([i for _,bj,i in selected if bj==bi],dtype=int);K=len(ids);X=b['X'];r=X.shape[1];rho=b['rho'];c=b['c'][:,ids];d=b['d'][:,ids];lam=b['coeff'][b['pp'][ids]]
            C=np.sum(c*c,axis=0);DD=np.sum(d*d,axis=0);cd=np.sum(c*d,axis=0)
            vec=np.sum(2*lam*(d*(C+DD)[None,:]+2*c*cd[None,:]),axis=1)
            correction+=np.sum(b['cubes'][:,ids],axis=1)-3*rho*(X@vec)
            ce=b['ce'][:,ids];ac=X[:,:min(256,r)]@ce[:min(256,r)];ad=X@d
            stable=np.sum(lam*(6*ac*ac*ad+2*ad**3),axis=1)
            plus=X@(ce+d);minus=X@(ce-d);naive=np.sum(lam*(plus**3-minus**3),axis=1)
            X32=X.astype('f4');c32=ce.astype('f4');d32=d.astype('f4');l32=lam.astype('f4')
            ac32=X32[:,:min(256,r)]@c32[:min(256,r)];ad32=X32@d32
            stable32=np.sum(l32*(6*ac32*ac32*ad32+2*ad32**3),axis=1,dtype=float)
            p32=X32@(c32+d32);m32=X32@(c32-d32);naive32=np.sum(l32*(p32**3-m32**3),axis=1,dtype=float)
            stability.append(dict(block=b['name'],pairs=K,f64_naive_vs_stable_rel=float(np.linalg.norm(naive-stable)/np.linalg.norm(stable)),f32_stable_rel=float(np.linalg.norm(stable32-stable)/np.linalg.norm(stable)),f32_naive_rel=float(np.linalg.norm(naive32-stable)/np.linalg.norm(stable)),**b['proof']))
            nt=len(tail);nr=b['F'].shape[1];paircount=nr//2
            # Numeric priced bounds; dynamic gathers/sort billed separately in production.
            projection=2*nt*(min(256,r)+r)*K
            cubic=9*nt*K+max(nt*(K-1),0)
            trace_projection=2*nt*r+3*nt
            selection=5*r*paircount+24*paircount # c/d, norms and dot, score; norm2 reuse could lower.
            trace_vector=8*r*K
            priced.append(dict(block=b['name'],pairs=K,rank=r,columns=nr,tail_rows=nt,projection_F=projection,cubic_F=cubic,selected_trace_projection_F=trace_projection,selection_per_new_factor_F=selection,selected_trace_vector_per_new_factor_F=trace_vector,per_call_cold_sum=projection+cubic+trace_projection+selection+trace_vector,per_call_reused_sum=projection+cubic+trace_projection))
        new=base+correction;metrics=[]
        for label,idx in [('all_rows',np.arange(1024)),('rounded_ott_tail',tail),('fixed_2p5_3_band',band)]:
            for weighted in (False,True):
                e0=base[idx]-truth[idx];e1=new[idx]-truth[idx]
                if weighted:e0=e0*sens[idx];e1=e1*sens[idx]
                dlt=e1-e0;n0=float(e0@e0);n1=float(e1@e1)
                metrics.append(dict(rows=label,mean_weighted=weighted,n=len(idx),baseline_error2=n0,new_error2=n1,error2_recovery_pct=100*(1-n1/n0),correction_cosine=float(-e0@dlt/(np.linalg.norm(e0)*np.linalg.norm(dlt)))))
        cold=sum(p['per_call_cold_sum'] for p in priced);warm=sum(p['per_call_reused_sum'] for p in priced)
        results.append(dict(total_pairs=budget,selected_tensor_energy_fraction=sum(v for v,_,_ in selected)/totalenergy,metrics=metrics,pricing=priced,stability=stability,L13_cold_F=cold,L13_cached_F=warm,full_network_8_equal_late_calls_upper_F=8*cold,full_network_8_equal_late_calls_upper_bill_pct=100*8*cold/244596740000,full_network_8_equal_late_calls_cached_bill_pct=100*8*warm/244596740000))
    out=dict(status='SAVED_COMPONENT_PROXY_ONLY',network=651,layer=13,current_common_rank=256,join_rank=384,old_fit_relative_error=olderr,tail_count=len(tail),results=results,archives={str(p):sha(p) for p in(P,A,wp)},script_sha256=sha(__file__),prereg_sha256=sha(D/'PREREG_v1.json'),seconds=time.monotonic()-START,maxRSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,
             price_limits=['Not a production FNP tariff: selection/gather/sort wrapper counts and actualalllayer geometry notyetmeasured.','8 equal L13latecalls is explicit scenario, not observednetworkcost.','Diagonal-only addback is onlyonepart ofOTT30 fullrowrestoration; cannotcarryOTT30terminalgain directly.','Exact selectedradialenergy repair could reuseAselected; notincludedabove.'])
    assert out['seconds']<120 and out['maxRSS']<2_000_000_000
    (D/'SAVED_RESULT_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
