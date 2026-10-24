"""Fixed greedy pairwise-rotation oracle on saved factors, no model or selection grid."""
from pathlib import Path
import hashlib,json,resource,time,os
import numpy as np
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    assert os.environ[key]=='1'
D=Path(__file__).resolve().parent;O=D.parents[3];start=time.monotonic()
P=O/'engineer/a35_join/state/s39_rank0651.npz';A=O/'engineer/a36_carrier/state/s38_rank0651.npz'
S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ent=json.loads((S/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][651]
wp=S/Path(ent['path']).parent/'weights.npy'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prereg=dict(selector='Greedy global maximum absolute normalized full factor Gram cross-correlation; unique tail and retained partner; deterministic row-major ties.',drop=64,
 rotation='Exact symmetric 2x2 factor Gram eigendecomposition; retain major eigenvector, omit minor. Matching U rotation preserves full product.',
 state='Full state and pair-common256 unchanged; independent diagonal remains exact.',
 source_sha256=sha(__file__),inputs={str(p):sha(p) for p in(P,A,wp)},models=0,no_grid=True)
(D/'PREREG_v1.json').open('x').write(json.dumps(prereg,indent=2)+'\n')
z=np.load(P);J=z['J_13'].astype(float);lam=z['lam_13'].astype(float);Qp=z['Qp_13'].astype(float);core=z['core_13'].astype(float);prev=z['prev_13'];lo=z['lamold_12'].astype(float)
saved=z['old_12'].astype(float);full=np.zeros((1024,saved.shape[1]));full[z['rows_12']]=saved;obs=(full*z['gate_12'][:,None])[prev];valid=np.abs(obs).sum(axis=1)>0
oldF=np.linalg.lstsq(Qp[valid],obs[valid],rcond=None)[0];olderr=float(np.linalg.norm(Qp[valid]@oldF-obs[valid])/np.linalg.norm(obs[valid]));del full,saved,obs
G=(J*(lam*lam)[None,:])@J.T+Qp@core@Qp.T;G=(G+G.T)*.5;ix=np.sort(np.argsort(-np.diag(G),kind='stable')[:384]);Q=np.linalg.qr(G[:,ix],mode='reduced')[0];del G
Fj=Q.T@J;overlap=Q.T@Qp;W=np.load(wp,mmap_mode='r')[13,prev,:].astype(float);U=W.T@Q;Um=U@overlap;del J,W,Q,Qp,core
aux=np.load(A)['aux_13'];sigma=aux[2];pdf=aux[4];gate=aux[0]
select=lambda score:np.argsort(-score,kind='stable')[:min(1024,((np.count_nonzero(score>=.01)+31)//32)*32)]
cols=select(gate);rows=select(np.where(gate>=.01,np.minimum(gate,1-gate),-1))
lead=.5*pdf[rows]/sigma[rows];right=gate[cols]
E={k:np.zeros((len(rows),len(cols))) for k in ('uniform','pair_oracle')};blocks=[]
for name,X,F,coeff in [('old_map',Um,oldF,lo),('new_join',U,Fj,lam)]:
    cuts=np.r_[0,np.flatnonzero(np.sign(coeff[1:])!=np.sign(coeff[:-1]))+1,len(coeff)]
    pp=[];mm=[]
    for j in range(0,len(cuts)-1,2):
        a,b,c=map(int,cuts[j:j+3]);assert b-a==c-b and np.array_equal(coeff[a:b],-coeff[b:c]);pp.extend(range(a,b));mm.extend(range(b,c))
    pp=np.array(pp);mm=np.array(mm);fc=.5*(F[:,pp]+F[:,mm]);fd=.5*(F[:,pp]-F[:,mm]);fc[256:]=0
    xc=X[rows,:256]@fc[:256];xd=X[rows]@fd
    Dv=2*coeff[pp][None,:]*(xc*xc+xd*xd)
    rank,pairs=fd.shape;keep=rank-64;tail=np.arange(keep,rank)
    norms=np.sum(fd*fd,axis=1);cross=fd[tail]@fd[:keep].T
    denom=np.sqrt(norms[tail,None]*norms[None,:keep]);score=np.divide(np.abs(cross),denom,out=np.zeros_like(cross),where=denom>0)
    remaining=score.copy();selected=[]
    for j in range(64):
        ti,hi=np.unravel_index(np.argmax(remaining),remaining.shape)
        selected.append((int(hi),int(tail[ti]),float(score[ti,hi])))
        remaining[ti,:]=-np.inf;remaining[:,hi]=-np.inf
    hs=np.array([v[0] for v in selected]);ts=np.array([v[1] for v in selected])
    assert len(set(hs.tolist()))==len(set(ts.tolist()))==64 and set(ts)==set(tail)
    minor=[];major=[];minorU=[];majorU=[];eigvals=[]
    for h,t,_ in selected:
        gram=np.array([[norms[h],fd[h]@fd[t]],[fd[h]@fd[t],norms[t]]])
        ev,rot=np.linalg.eigh(gram);eigvals.append(ev.tolist())
        pair=fd[[h,t]];upair=X[:,[h,t]]
        minor.append(rot[:,0]@pair);major.append(rot[:,1]@pair)
        minorU.append(upair@rot[:,0]);majorU.append(upair@rot[:,1])
    minor=np.asarray(minor);major=np.asarray(major);minorU=np.asarray(minorU).T;majorU=np.asarray(majorU).T
    # Tiny exact factor identity before dropping; full independent CP diagonal is unaffected.
    ref=fd[hs,:24].T@X[:24,hs].T+fd[ts,:24].T@X[:24,ts].T
    restored=major[:,:24].T@majorU[:24].T+minor[:,:24].T@minorU[:24].T
    ident=float(np.max(np.abs(ref-restored)));assert np.allclose(ref,restored,rtol=1e-11,atol=1e-12)
    hu=Dv@fd[tail].T;hp=Dv@minor.T
    E['uniform']+=hu@X[cols][:,tail].T
    E['pair_oracle']+=hp@minorU[cols].T
    blocks.append(dict(name=name,rank=rank,pairs=pairs,rows=len(rows),selected_pairs=selected,
      normalized_cross_min=float(min(v[2] for v in selected)),normalized_cross_mean=float(np.mean([v[2] for v in selected])),
      normalized_cross_max=float(max(v[2] for v in selected)),
      factor_energy_recovery_pct=100*(1-float(np.sum(minor*minor))/float(np.sum(fd[tail]*fd[tail]))),
      inner_error2_recovery_pct=100*(1-float(np.sum(hp*hp))/float(np.sum(hu*hu))),
      product_identity_max_abs=ident,
      exact_oracle_cross_raw_flops=2*64*keep*pairs,
      deployed128_sketch_cross_raw_flops=2*64*keep*128,
      deployed_exact_pair_gram_and_rotation_raw_flops_bound=9*64*pairs+6*len(rows)*64,
      note='Oracle full pair Gram used only for partner selection; deployable bound excludes gathers, stable greedy matching, and actual kernel tariffs.'))
def metrics(e):
    e=e.copy();where={int(c):j for j,c in enumerate(cols)}
    for i,r in enumerate(rows):
        if int(r) in where:e[i,where[int(r)]]=0
    weighted=lead[:,None]*e*right[None,:]
    sym=np.zeros((1024,1024));sym[np.ix_(rows,cols)]=weighted;sym=sym+sym.T
    return dict(raw_offdiag=float(np.sum(e*e)),weighted_directed=float(np.sum(weighted*weighted)),leading_symmetric_covariance=float(np.sum(sym*sym)),symmetry_max_abs=float(np.max(np.abs(sym-sym.T))),diagonal_max_abs=float(np.max(np.abs(np.diag(sym)))))
metric={k:metrics(e) for k,e in E.items()};base=metric['uniform']
recover={q:100*(1-metric['pair_oracle'][q]/base[q]) for q in ('raw_offdiag','weighted_directed','leading_symmetric_covariance')}
out=dict(prereg=prereg,status='SAVED_HISTORICAL_FACTOR_PROXY_ONLY',network=651,layer=13,common_rank=256,tail_cut=64,rows=len(rows),cols=len(cols),old_fit_relative_error=olderr,blocks=blocks,metrics=metric,recovery_pct=recover,seconds=time.monotonic()-start,maxRSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,limits=['Historical S39 factors/S38 response geometry, not current native state or terminal truth.','NP norm-coupling modification of left squares is absent.','One greedy full-Gram partner oracle and exact2x2 eigen rotation; no coefficient/selection search. Historical old block rank320; current state commonly384.'])
assert out['maxRSS']<2_000_000_000
(D/'SAVED_RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
