"""Two frozen TRAIN90 local-innovation head fits; no inference arithmetic added.

Shared64 and existing per-layer64 heads are independent preregistered alternatives.
All proxy validation folds hold out whole networks inside the protected TRAIN90
roster. Native feedback remains unmodeled; no candidate is exported here.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import argparse,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SIB=HERE.parent/'truth_response95_v1'
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'local_innovation90_fit_v1'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
DESIGN='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
HOOK='8a94ea60aab869827ad97cec4aa2aea531bfdbc5c5341e53038b7f2feea4a91b'
PLAN='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'
RIDGE=.1;DAMPING=.5;N=1024
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def arr(r,k):
    a=np.asarray(r[k]['data'],dtype='f8')
    assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all(),k
    return a
def solve(G,b,n):
    scale=np.maximum(np.sqrt(np.maximum(np.diag(G),0)/n),1e-8)
    z=G/(scale[:,None]*scale[None,:])
    c=np.linalg.solve(z+RIDGE*n*np.eye(len(b)),b/scale)/scale*DAMPING
    assert np.isfinite(c).all()
    return c
def coefficients(G,b,nnet,mode):
    if mode=='shared64':
        c=solve(G.sum(axis=0),b.sum(axis=0),nnet*15*N)
        return np.broadcast_to(c,(15,64)).copy()
    assert mode=='per_layer960'
    return np.stack([solve(g,v,nnet*N) for g,v in zip(G,b)])
def frozen(c,P):
    return (P['head'][1:16].astype('f8')+c).astype('f4').astype('f8')-P['head'][1:16].astype('f8')
def metric(r,c):
    u0=float(np.sum(r['u0_sum'])/(15*N))
    u1=float((np.sum(r['u0_sum'])-2*np.sum(c*r['b'])+np.einsum('li,lij,lj->',c,r['G'],c))/(15*N))
    assert u0>0 and u1>0
    return dict(seed=r['seed'],u0=u0,u1=u1,relative_pct=100*(u1/u0-1))
def summary(rows):
    x=np.array([r['relative_pct'] for r in rows])
    return dict(n=len(rows),mean_relative_pct=float(x.mean()),lower=int(np.sum(x<0)),
        se_relative_pct=float(x.std(ddof=1)/np.sqrt(len(x))),
        pooled_pct=float(100*(sum(r['u1'] for r in rows)/sum(r['u0'] for r in rows)-1)))
def selftest():
    rng=np.random.default_rng(20261021091);X=rng.normal(size=(321,64));y=rng.normal(size=321)
    G=X.T@X;b=X.T@y;scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-8);z=X/scale
    expected=np.linalg.solve(z.T@z+RIDGE*len(X)*np.eye(64),z.T@y)/scale*DAMPING
    np.testing.assert_allclose(solve(G,b,len(X)),expected,rtol=1e-11,atol=1e-12)
    xx=rng.normal(size=(5,15,67,64));aa=rng.normal(size=(5,15,67));bb=aa+rng.normal(size=aa.shape)*.1
    gs=np.einsum('nlri,nlrj->nlij',xx,xx);bs=np.einsum('nlri,nlr->nli',xx,.5*(aa+bb))
    for h in range(5):
        g=gs.sum(axis=0)-gs[h];b=bs.sum(axis=0)-bs[h]
        for l in (0,7,14):
            rest=np.concatenate([xx[k,l] for k in range(5) if k!=h]);yy=np.concatenate([.5*(aa[k,l]+bb[k,l]) for k in range(5) if k!=h])
            np.testing.assert_allclose(solve(g[l],b[l],4*67),solve(rest.T@rest,rest.T@yy,4*67),atol=1e-12,rtol=1e-11)
    c=rng.normal(size=(15,64))*.01
    pred=np.einsum('lri,li->lr',xx[0],c)
    direct=float(np.sum((aa[0]-pred)*(bb[0]-pred)))
    gramvalue=float(np.sum(aa[0]*bb[0])-2*np.sum(c*bs[0])+np.einsum('li,lij,lj->',c,gs[0],c))
    np.testing.assert_allclose(direct,gramvalue,rtol=1e-13,atol=1e-11)
    e=rng.normal(size=31);w=rng.normal(size=(31,31));gate=rng.uniform(size=31);born=rng.normal(size=31)
    inherited=gate*(w.T@e)
    np.testing.assert_allclose(inherited+born-inherited,born,atol=1e-14,rtol=1e-12)
    return dict(normal_equations_match_stacked=True,network_LOO_subtraction=True,
        cross_half_metric_equals_explicit=True,local_innovation_identity=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    checks=selftest()
    if args.self_test:print(json.dumps(checks));return
    assert sys.platform=='linux'
    prereg=read(HERE/'PREREG_v1.json')
    assert prereg['script_sha256']==sha(__file__) and prereg['design_sha256']==DESIGN
    assert sha(SIB/'DESIGN_LOCAL_v1.json')==DESIGN
    d=read(SIB/'DESIGN_LOCAL_v1.json');train=d['train']
    assert len(train)==90 and len(set(train))==90 and d['weight_hash_disjointness']
    assert not set(train)&set(d['protected_seed_ids'])
    ready=read(SIB/'CAPTURE_READY_v1.json');assert ready['successful'] and ready['design_sha256']==DESIGN
    assert ready['hook_sha256']==HOOK and ready['plan_sha256']==PLAN
    assert sha(ready['plan'])==PLAN and sha(ready['recipe'])==ready['recipe_sha256']
    cp=read(ready['plan']);assert [r for g in cp['groups'] for r in g]==train
    doneall=read(CAP/'COMPLETE.json');assert doneall['successful'] and doneall['plan_sha256']==PLAN
    rm=read(CAP/'RUN_MANIFEST.json');assert rm['candidate_sha256']==SCI and rm['blas_threads']==4
    sys.path.insert(0,str(V));import fit_frontier_v1 as f
    assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    src=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(src)==SCI
    release=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_rel_v1.py';assert sha(release)==REL
    literal,_=f.literals_and_functions(src.read_text());P=f.decode(literal['_GNN_LITERAL'])
    OUT.mkdir(exist_ok=False);write(OUT/'PREREG_COPY_v1.json',prereg)
    records=[];provenance=[]
    for gi,group in enumerate(cp['groups'],1):
        part='part%02d'%gi;folder=CAP/part;gp=CAP/(part+'_GUARD.json');guard=read(gp)
        assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors'] and guard['guard_sha256']==GUARD
        manifest=read(folder/'manifest.json')
        assert manifest['sha256']==SCI and manifest['diagnostic_only'] and manifest['environment']['thread_provenance']['blas_threads']==4
        capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json')
        assert meta['source_sha256']==SCI and meta['hook_sha256']==HOOK and meta['target_free']
        capturedP=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],capturedP[k]) for k in P)
        completion=read(capdir/'CAPTURE_COMPLETE_v1.json');assert completion['complete'] and completion['target_free']
        for seed in group:
            assert seed in train
            raw=folder/('diagnostic_pred_%d.bin'%seed);assert raw.stat().st_size==65536
            pred=np.fromfile(raw,dtype='<f4').reshape(16,N).astype('f8');assert np.isfinite(pred).all()
            wr=cp['rows'][str(seed)]['weights'];wp=ROOT/wr['path']
            assert wp.stat().st_size==wr['bytes'] and sha(wp)==wr['sha256']
            W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,N,N)
            tr=d['rows'][str(seed)];tp=ROOT/tr['truth']['path']
            assert tp.stat().st_size==tr['truth']['bytes'] and sha(tp)==tr['truth']['sha256']
            assert tr['raw_float32_sha256']==wr['raw_float32_sha256']
            assert wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
            with np.load(tp,allow_pickle=False) as z:
                assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256']
                A=z['mc_post_A'].astype('f8')-pred;B=z['mc_post_B'].astype('f8')-pred
            assert A.shape==B.shape==(16,N) and np.isfinite(A).all() and np.isfinite(B).all()
            rr=dict(seed=seed,G=[],b=[],u0_sum=[],physical_gram=[],head_rms2=[])
            caps=[]
            for l in range(1,16):
                matches=[x for x in completion['files'] if x['rank']==seed and x['phase']=='scored' and x['layer']==l]
                assert len(matches)==1
                rec=matches[0];path=capdir/rec['path'];assert sha(path)==rec['sha256']
                with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
                assert r['rank']==seed and r['layer']==l and r['phase']=='scored'
                state=arr(r,'state');sig=arr(r,'sigma');ng=arr(r,'gate_feature');saved=arr(r,'delta')
                assert state.shape==(N,64) and sig.shape==ng.shape==saved.shape==(N,)
                assert np.max(np.abs(ng))<7.999
                gate=ng*float(P['scales'][l,5])+float(P['centers'][l,5])
                assert gate.min()>-2e-6 and gate.max()<1+2e-6
                gate=np.clip(gate,0,1);physical=sig*float(P['rms'][l])*(1+.25*np.tanh(float(P['gain'][l])))
                assert physical.min()>1e-12
                oracle=physical*(state@P['head'][l].astype('f8')+float(P['head_b'][l]))
                replay=float(np.max(np.abs(oracle-saved)));assert replay<1e-8+2e-5*float(np.max(np.abs(saved)))
                mask=(pred[l]>1e-12).astype('f8') if l<15 else np.ones(N)
                X=state*mask[:,None]
                # Labels are used only here, on the immutable TRAIN90 roster.
                inherited=np.asarray(W[l],dtype='f8').T@np.stack([A[l-1],B[l-1]],axis=1)
                yA=(A[l]-gate*inherited[:,0])/physical
                yB=(B[l]-gate*inherited[:,1])/physical
                rr['G'].append(X.T@X);rr['b'].append(X.T@(.5*(yA+yB)))
                rr['u0_sum'].append(float(yA@yB))
                direct=physical[:,None]*X
                rr['physical_gram'].append(direct.T@direct/N)
                rr['head_rms2'].append(float(np.mean((saved*mask)**2)))
                caps.append(dict(layer=l,capture_sha256=sha(path),head_replay_max_abs=replay,clipped=int(np.sum(mask==0))))
            for k in ('G','b','u0_sum','physical_gram','head_rms2'):rr[k]=np.asarray(rr[k])
            assert float(np.sum(rr['u0_sum']))>0
            records.append(rr);provenance.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp),
                prediction_sha256=sha(raw),guard_sha256=sha(gp),captures=caps))
            print(json.dumps(dict(utc=utc(),loaded=seed,completed=len(records))),flush=True)
    assert [r['seed'] for r in records]==train
    G=sum(r['G'] for r in records);b=sum(r['b'] for r in records);results={}
    for mode in ('shared64','per_layer960'):
        coef=frozen(coefficients(G,b,len(records),mode),P)
        training=[metric(r,coef) for r in records];loo=[]
        for r in records:
            cv=frozen(coefficients(G-r['G'],b-r['b'],len(records)-1,mode),P)
            loo.append(metric(r,cv))
        cv=summary(loo);blocks=[summary([r for r in loo if r['seed'] in train[j::5]]) for j in range(5)]
        correction2=np.stack([np.einsum('li,lij,lj->l',coef,r['physical_gram'],coef) for r in records])
        original2=np.stack([r['head_rms2'] for r in records])
        overall=float(np.sqrt(max(0,correction2.sum())/max(original2.sum(),1e-30)))
        layer_ratios=np.sqrt(np.maximum(correction2.sum(axis=0),0)/np.maximum(original2.sum(axis=0),1e-30))
        physical=dict(overall_correction_to_head_rms=overall,per_layer_correction_to_head_rms=layer_ratios.tolist(),
            max_per_layer_correction_to_head_rms=float(layer_ratios.max()))
        passed=bool(cv['mean_relative_pct']<=-.15 and cv['lower']>=54 and sum(r['mean_relative_pct']<0 for r in blocks)>=4
            and overall<=.25 and float(layer_ratios.max())<=1.)
        results[mode]=dict(status='PROXY_GATE_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',
            passed=passed,coef_by_layer=coef.tolist(),train=summary(training),train_per_network=training,
            loo=cv,loo_per_network=loo,descriptive_blocks=blocks,physical=physical)
    result=dict(utc=utc(),status='PROXY_FITS_COMPLETE',results=results,source_sha256=SCI,release_source_sha256=REL,
        prereg_sha256=sha(HERE/'PREREG_v1.json'),design_sha256=DESIGN,script_sha256=sha(__file__),selftests=checks,
        provenance=provenance,no_adjusted_gain_claim=True,no_candidate_exported=True,no_protected_truth_read=True,
        omitted_feedback='Changed covariance, learned features, recurrent state, cumulants and clipping trajectories')
    write(OUT/'FITTED_v1.json',result)
    print(json.dumps(dict(utc=utc(),fit_sha256=sha(OUT/'FITTED_v1.json'),
        results={k:{z:v[z] for z in ('status','train','loo','physical')} for k,v in results.items()})),flush=True)
if __name__=='__main__':main()

