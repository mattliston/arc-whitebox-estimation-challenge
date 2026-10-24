"""Fixed first-hidden128 and squared FIRST-hidden128 extension on TRAIN90.
Whole-network LOO refits baseline, target-free residualization and complement.
No validation/fresh targets, native launch or source export; frozen target-free reconstructed first128 only.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;H1=HERE.parent/'breakthrough_v1/first_hidden_readout_v1';SIB=HERE.parent/'truth_response95_v1';V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'first_hidden_squared90_fit_v1'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
DESIGN='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
HOOK='8a94ea60aab869827ad97cec4aa2aea531bfdbc5c5341e53038b7f2feea4a91b'
PLAN='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'
PREREG='f3d542f3633ff983832f7b2dc1c4ff37699bba323d1b4f4af6389bea3c6331e8'
RIDGE=.1;DAMPING=.5;PROJECTION_RIDGE=1e-10;N=1024;D=449
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

CONTINUATION_PIN='dc7d0d973bbc2f136a327778b5fb3f64d8bff79610f9decc6b7812d57cbc1cdd'

CFPATH=H1/'fit_first128_local90_v1.py'
assert sha(CFPATH)=='8e252f559879be4d6c2bceb033bf9d84408ada1c418942bcf4d4d939d009c526'
sp=importlib.util.spec_from_file_location('fixed_baseline_first128',CFPATH)
cf=importlib.util.module_from_spec(sp);sp.loader.exec_module(cf)

def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def arr(r,k):
 a=np.asarray(r[k]['data'],dtype='f8')
 assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all(),k
 return a
def solve(G,b,n):
 scale=np.maximum(np.sqrt(np.maximum(np.diag(G),0)/n),1e-8)
 c=np.linalg.solve(G/(scale[:,None]*scale[None,:])+RIDGE*n*np.eye(len(b)),b/scale)/scale*DAMPING
 assert np.isfinite(c).all()
 return c
def nullspace(E):
 _,s,v=np.linalg.svd(E.astype('f8'),full_matrices=True)
 assert s.min()>1e-8
 U=v[64:].T.copy()
 for j in range(64):
  if U[np.argmax(np.abs(U[:,j])),j]<0:U[:,j]*=-1
 np.testing.assert_allclose(E@U,0,atol=1e-10)
 np.testing.assert_allclose(U.T@U,np.eye(64),atol=1e-12)
 return U
def frozen(c,head,bias):
 result=c.copy()
 result[:64]=(head.astype('f8')+c[:64]).astype('f4').astype('f8')-head.astype('f8')
 result[64]=float(np.float32(float(bias)+c[64]))-float(bias)
 result[65:]=c[65:].astype('f4').astype('f8')
 return result
def first_square(x):
    assert x.shape[-1]==128 and np.isfinite(x).all()
    value=np.asarray(x,dtype=np.float32)
    square=np.multiply(value,value,dtype=np.float32)
    assert square.dtype==np.float32 and np.isfinite(square).all()
    return square.astype('f8')

def one_layer(G,b,n,head,bias,U):
    _,parent=cf.one_layer(G[:321,:321],b[:321],n,head,bias,U)
    baseline=np.zeros(D);baseline[:321]=parent
    Q=np.zeros((D,257));Q[:65,:65]=np.eye(65);Q[65:193,65:129]=U;Q[193:321,129:]=np.eye(128)
    T=np.zeros((D,128));T[321:]=np.eye(128)
    gg=Q.T@G@Q;cross=Q.T@G@T
    scale=np.maximum(np.sqrt(np.maximum(np.diag(gg),0)/n),1e-8)
    A=np.linalg.solve(gg/(scale[:,None]*scale[None,:])+PROJECTION_RIDGE*n*np.eye(257),cross/scale[:,None])/scale[:,None]
    R=T-Q@A;gres=R.T@G@R;gres=.5*(gres+gres.T)
    k=solve(gres,R.T@(b-G@baseline),n)
    candidate=frozen(baseline+R@k,head,bias)
    assert np.isfinite(candidate).all()
    return baseline,candidate
def coefficients(G,b,nnet,P,U):
 result=[one_layer(G[l],b[l],nnet*N,P['head'][l+1],P['head_b'][l+1],U) for l in range(15)]
 return np.stack([x[0] for x in result]),np.stack([x[1] for x in result])
def loss(r,c):
 return float(np.sum(r['u0_sum'])-2*np.sum(c*r['b'])+np.einsum('li,lij,lj->',c,r['G'],c))
def metric(r,base,cand):
 original=float(np.sum(r['u0_sum']));u0=loss(r,base);u1=loss(r,cand)
 assert original>0 and u0>0 and u1>0
 return dict(seed=r['seed'],original=original/(15*N),baseline=u0/(15*N),candidate=u1/(15*N),
  incremental_pct=100*(u1/u0-1),total_pct=100*(u1/original-1),baseline_pct=100*(u0/original-1))
def summary(rows):
 x=np.array([r['incremental_pct'] for r in rows])
 return dict(n=len(rows),mean_incremental_relative_pct=float(x.mean()),incrementally_lower=int(np.sum(x<0)),
  se_incremental_relative_pct=float(x.std(ddof=1)/np.sqrt(len(x))),mean_total_relative_pct=float(np.mean([r['total_pct'] for r in rows])),
  mean_baseline_relative_pct=float(np.mean([r['baseline_pct'] for r in rows])),
  pooled_incremental_pct=float(100*(sum(r['candidate'] for r in rows)/sum(r['baseline'] for r in rows)-1)))
def selftest():
    rng=np.random.default_rng(202610210439);E=rng.normal(size=(64,128));U=nullspace(E)
    hh=rng.normal(size=(703,128));ss=hh@E.T+rng.normal(size=64)
    first=rng.normal(size=(703,128))+hh@rng.normal(size=(128,128))*.1
    square=first_square(first)
    assert not np.array_equal(square,first_square(hh))
    X=np.column_stack([ss,np.ones(703),hh,first,square]);A=rng.normal(size=703);B=A+rng.normal(size=703)*.1
    G=X.T@X;b=X.T@(.5*(A+B));head=rng.normal(size=64).astype('f4');bias=np.float32(.1)
    base,c=one_layer(G,b,len(X),head,bias,U);_,expected=cf.one_layer(G[:321,:321],b[:321],len(X),head,bias,U)
    np.testing.assert_array_equal(base[:321],expected);np.testing.assert_array_equal(base[321:],0)
    np.testing.assert_allclose((A-X@c)@(B-X@c),A@B-2*c@b+c@G@c,rtol=1e-10,atol=1e-10)
    x1,x2=X[:351],X[351:];y=.5*(A+B)
    a1,c1=one_layer(G-x2.T@x2,b-x2.T@y[351:],len(x1),head,bias,U)
    a2,c2=one_layer(x1.T@x1,x1.T@y[:351],len(x1),head,bias,U)
    np.testing.assert_allclose(a1,a2,atol=1e-6,rtol=1e-5);np.testing.assert_allclose(c1,c2,atol=1e-6,rtol=1e-5)
    y_changed=y.copy();y_changed[351:]+=1e9
    a3,c3=one_layer(x1.T@x1,x1.T@y_changed[:351],len(x1),head,bias,U)
    np.testing.assert_array_equal(a2,a3);np.testing.assert_array_equal(c2,c3)
    np.testing.assert_allclose(X@c,ss@c[:64]+c[64]+hh@c[65:193]+first@c[193:321]+square@c[321:],atol=1e-12,rtol=1e-10)
    return dict(independent_fold_first128_baseline=True,cross_half_moment_metric=True,fold_projection_and_scaling=True,float32_deployment_equation=True,exact_float32_first_squared128_feature_deployment=True,first_not_second_hidden_stage=True,excluded_label_mutation_cannot_change_fit=True)

def validate_reconstructed_first128(train):
 return cf.validate_reconstructed_first128(train)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args();checks=selftest()
 if args.self_test:print(json.dumps(checks));return
 assert sys.platform=='linux' and sha(HERE/'PREREG_v1.json')==PREREG
 review=read(HERE/'FIT_REVIEW_v1.json');assert review['successful'] and review['frozen_fit_authorized'] and review['first128_actual_paired_win_confirmed']
 assert review['prereg_sha256']==PREREG and review['fitter_sha256']==sha(__file__)
 prereg=read(HERE/'PREREG_v1.json');assert prereg['fit']['ridge']==RIDGE and prereg['fit']['damping']==DAMPING and prereg['fit']['no_grid']
 assert sha(SIB/'DESIGN_LOCAL_v1.json')==DESIGN
 pricepath=RUNS/'first_hidden_squared90_price_v1/PRICE_v1.json';price=read(pricepath)
 assert sha(pricepath)=='e287cb8d36ad818f539c21a3a28d9a8eb5443166d60b1012b14e1000bab21a02' and price['successful'] and price['passes_price_gate']
 pgpath=RUNS/'first_hidden_squared90_price_v1_GUARD_v1.json';assert sha(pgpath)=='52e69d0d706ff61c46b08f14e93d16f7db9ae2b4201fd122f73565eea2e16e98';pg=read(pgpath)
 assert pg['successful'] and pg['child_exit']==0 and pg['whole_group_retired'] and pg['guard_sha256']==GUARD and not pg['cleanup_errors']
 d=read(SIB/'DESIGN_LOCAL_v1.json');train=d['train']
 hfiles,capture_proof=validate_reconstructed_first128(train)
 parentfit=RUNS/'first_hidden128_local90_fit_v1/FITTED_v1.json';assert sha(parentfit)=='d1bca8878d62fe96ebd18780a5c560f8e1a04d557a41eb928546449906376125'
 parent=read(parentfit);parent_cv={r['seed']:r for r in parent['loo_per_network']}

 assert len(train)==90 and len(set(train))==90 and d['weight_hash_disjointness'] and train==prereg['train']
 assert not set(train)&set(d['protected_seed_ids'])
 ready=read(SIB/'CAPTURE_READY_v1.json');assert ready['successful'] and ready['design_sha256']==DESIGN
 assert ready['hook_sha256']==HOOK and ready['plan_sha256']==PLAN and sha(ready['plan'])==PLAN and sha(ready['recipe'])==ready['recipe_sha256']
 cp=read(ready['plan']);assert [r for g in cp['groups'] for r in g]==train
 doneall=read(CAP/'COMPLETE.json');assert doneall['successful'] and doneall['plan_sha256']==PLAN
 rm=read(CAP/'RUN_MANIFEST.json');assert rm['candidate_sha256']==SCI and rm['blas_threads']==4
 sys.path.insert(0,str(V));import fit_frontier_v1 as f
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 src=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(src)==SCI
 release=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_rel_v1.py';assert sha(release)==REL
 literal,_=f.literals_and_functions(src.read_text());P=f.decode(literal['_GNN_LITERAL']);U=nullspace(P['enc4_w'])
 OUT.mkdir(exist_ok=False);write(OUT/'PREREG_COPY_v1.json',prereg);records=[];provenance=[]
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
   FIRST=[];hprovenance=[]
   for layer in range(1,16):
    hr=hfiles[seed,layer];hp=Path(hr['path'])
    assert sha(hp)==hr['sha256'] and hp.stat().st_size==hr['bytes']==N*128*4
    first=np.fromfile(hp,dtype='<f4').reshape(N,128).astype('f8')
    assert np.isfinite(first).all()
    FIRST.append(first)
    hprovenance.append(dict(layer=layer,path=str(hp),sha256=sha(hp)))
   FIRST=np.stack(FIRST)
   assert FIRST.shape==(15,N,128)
   tr=d['rows'][str(seed)];tp=ROOT/tr['truth']['path']
   assert tp.stat().st_size==tr['truth']['bytes'] and sha(tp)==tr['truth']['sha256']
   assert tr['raw_float32_sha256']==wr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
   with np.load(tp,allow_pickle=False) as z:
    assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256']
    A=z['mc_post_A'].astype('f8')-pred;B=z['mc_post_B'].astype('f8')-pred
   assert A.shape==B.shape==(16,N) and np.isfinite(A).all() and np.isfinite(B).all()
   rr=dict(seed=seed,G=[],b=[],u0_sum=[],physical_gram=[],head_rms2=[]);caps=[]
   for l in range(1,16):
    matches=[x for x in completion['files'] if x['rank']==seed and x['phase']=='scored' and x['layer']==l];assert len(matches)==1
    rec=matches[0];path=capdir/rec['path'];assert sha(path)==rec['sha256']
    with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    assert r['rank']==seed and r['layer']==l and r['phase']=='scored'
    state=arr(r,'state');hidden=arr(r,'hidden2');sig=arr(r,'sigma');ng=arr(r,'gate_feature');saved=arr(r,'delta')
    assert state.shape==(N,64) and hidden.shape==(N,128) and sig.shape==ng.shape==saved.shape==(N,)
    assert np.max(np.abs(ng))<7.999
    gate=ng*float(P['scales'][l,5])+float(P['centers'][l,5]);assert gate.min()>-2e-6 and gate.max()<1+2e-6
    gate=np.clip(gate,0,1);physical=sig*float(P['rms'][l])*(1+.25*np.tanh(float(P['gain'][l])));assert physical.min()>1e-12
    oracle=physical*(state@P['head'][l].astype('f8')+float(P['head_b'][l]))
    replay=float(np.max(np.abs(oracle-saved)));assert replay<1e-8+2e-5*float(np.max(np.abs(saved)))
    state_replay=float(np.max(np.abs(state-(hidden@P['enc4_w'].astype('f8').T+P['enc4_b'].astype('f8')))));assert state_replay<3e-5
    mask=(pred[l]>1e-12).astype('f8') if l<15 else np.ones(N)
    square=first_square(FIRST[l-1])
    X=np.column_stack([state,np.ones(N),hidden,FIRST[l-1],square])*mask[:,None]
    inherited=np.asarray(W[l],dtype='f8').T@np.stack([A[l-1],B[l-1]],axis=1)
    yA=(A[l]-gate*inherited[:,0])/physical;yB=(B[l]-gate*inherited[:,1])/physical
    rr['G'].append(X.T@X);rr['b'].append(X.T@(.5*(yA+yB)));rr['u0_sum'].append(float(yA@yB))
    direct=physical[:,None]*X;rr['physical_gram'].append(direct.T@direct/N);rr['head_rms2'].append(float(np.mean((saved*mask)**2)))
    caps.append(dict(layer=l,capture_sha256=sha(path),head_replay_max_abs=replay,state_replay_max_abs=state_replay,clipped=int(np.sum(mask==0))))
   for k in ('G','b','u0_sum','physical_gram','head_rms2'):rr[k]=np.asarray(rr[k])
   assert float(np.sum(rr['u0_sum']))>0
   records.append(rr);provenance.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp),prediction_sha256=sha(raw),guard_sha256=sha(gp),captures=caps,first128_reconstruction_files=hprovenance))
   print(json.dumps(dict(utc=utc(),loaded=seed,completed=len(records))),flush=True)
 assert [r['seed'] for r in records]==train
 G=sum(r['G'] for r in records);b=sum(r['b'] for r in records)
 baseline,coef=coefficients(G,b,len(records),P,U)
 np.testing.assert_array_equal(baseline[:,:321],np.asarray(parent['coef_by_layer']))
 np.testing.assert_array_equal(baseline[:,321:],0)
 deployed=H1/'native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py'
 assert sha(deployed)=='4da68821a5db9212035531a25dc8ab0c197a44d84a1e5482168d4223d349032c'
 dl,_=f.literals_and_functions(deployed.read_text());Q=f.decode(dl['_GNN_LITERAL'])
 np.testing.assert_array_equal((P['head'][1:].astype('f8')+baseline[:,:64]).astype('f4'),Q['head'][1:])
 np.testing.assert_array_equal((P['head_b'][1:].astype('f8')+baseline[:,64]).astype('f4'),Q['head_b'][1:])
 np.testing.assert_array_equal(baseline[:,65:193].astype('f4'),Q['hidden_head'][1:])
 np.testing.assert_array_equal(baseline[:,193:321].astype('f4'),Q['first_hidden_head'][1:])
 training=[metric(r,baseline,coef) for r in records];loo=[]
 for i,r in enumerate(records):
  bc,cc=coefficients(G-r['G'],b-r['b'],len(records)-1,P,U);row=metric(r,bc,cc);np.testing.assert_allclose(row['baseline'],parent_cv[r['seed']]['candidate'],rtol=2e-9,atol=1e-14);loo.append(row)
  if (i+1)%10==0:print(json.dumps(dict(utc=utc(),LOO_complete=i+1)),flush=True)
 cv=summary(loo);blocks=[summary([r for r in loo if r['seed'] in train[j::5]]) for j in range(5)]
 increment=coef-baseline
 correction2=np.stack([np.einsum('li,lij,lj->l',increment,r['physical_gram'],increment) for r in records])
 original2=np.stack([r['head_rms2'] for r in records])
 overall=float(np.sqrt(max(0,correction2.sum())/max(original2.sum(),1e-30)))
 layer_ratios=np.sqrt(np.maximum(correction2.sum(axis=0),0)/np.maximum(original2.sum(axis=0),1e-30))
 physical=dict(incremental_correction_to_original_head_rms=overall,per_layer_ratios=layer_ratios.tolist(),max_layer_ratio=float(layer_ratios.max()))
 passed=bool(cv['mean_incremental_relative_pct']<=-.15 and cv['incrementally_lower']>=54 and sum(r['mean_incremental_relative_pct']<0 for r in blocks)>=4 and overall<=.25 and float(layer_ratios.max())<=1.)
 result=dict(utc=utc(),status='INCREMENTAL_PROXY_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',passed=passed,
  source_sha256=SCI,release_source_sha256=REL,prereg_sha256=PREREG,design_sha256=DESIGN,script_sha256=sha(__file__),
  price_sha256=sha(pricepath),selftests=checks,train=summary(training),train_per_network=training,loo=cv,loo_per_network=loo,
  descriptive_blocks=blocks,physical=physical,coef_by_layer=coef.tolist(),baseline_coef_by_layer=baseline.tolist(),first128_reconstruction_custody=capture_proof,
  provenance=provenance,no_adjusted_gain_claim=True,no_candidate_exported=True,no_protected_truth_read=True,
  ancestry='Authentic777a TRAIN90; baseline independently refit exact first128. Not on-policy337466; native feedback remains required.', baseline_fitted_sha256=sha(parentfit), baseline_full90_tensor_parity=True, root_fit_review_sha256=sha(HERE/'FIT_REVIEW_v1.json'),
  deployment='Original encoder/state/messages unchanged; head/head_b/hidden_head128/first_hidden_head128 jointly updated plus FIRSThidden float32 square scalar readout before enc2. Zero hidden recomputation; no independent fitted coefficient addition.',
  omitted_feedback='Changed covariance, learned features, recurrent state, cumulants and clipping trajectories')
 write(OUT/'FITTED_v1.json',result)
 print(json.dumps({k:result[k] for k in ('utc','status','train','loo','physical')}),flush=True)
if __name__=='__main__':main()

