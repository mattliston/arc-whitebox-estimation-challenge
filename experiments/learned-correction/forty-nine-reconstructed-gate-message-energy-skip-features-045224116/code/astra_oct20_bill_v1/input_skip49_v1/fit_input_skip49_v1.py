"""Fixed per-layer49-input skip after fold-fit hidden-complement TRAIN90.
Whole-network LOO refits baseline, target-free residualization and complement.
No validation/fresh targets, native launch or source export; authentic reconstructed H49 only.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SIB=HERE.parent/'truth_response95_v1';V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'input_skip49_fit_v1'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
DESIGN='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
HOOK='8a94ea60aab869827ad97cec4aa2aea531bfdbc5c5341e53038b7f2feea4a91b'
PLAN='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'
PREREG='302ac1525406ccb2d7e94c27fe4288f8c495e1ac34d41e0ab07a112579c41ae1'
RIDGE=.1;DAMPING=.5;PROJECTION_RIDGE=1e-10;N=1024;D=242
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

CFPATH=HERE.parent/'hidden_complement90_v1/fit_hidden_complement_v1.py'
assert sha(CFPATH)=='e6f33d68de379965c887cac3e43fbc31766625e66dead02d8bf0308cea2e5021'
sp=importlib.util.spec_from_file_location('fixed_baseline_hidden_complement',CFPATH)
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
def one_layer(G,b,n,head,bias,U):
    _,parent=cf.one_layer(G[:193,:193],b[:193],n,head,bias,U)
    baseline=np.zeros(D);baseline[:193]=parent
    Q=np.zeros((D,129));Q[:65,:65]=np.eye(65);Q[65:193,65:]=U
    T=np.zeros((D,49));T[193:]=np.eye(49)
    gg=Q.T@G@Q;cross=Q.T@G@T
    scale=np.maximum(np.sqrt(np.maximum(np.diag(gg),0)/n),1e-8)
    A=np.linalg.solve(gg/(scale[:,None]*scale[None,:])+PROJECTION_RIDGE*n*np.eye(129),cross/scale[:,None])/scale[:,None]
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
    h49=rng.normal(size=(703,49))+hh@rng.normal(size=(128,49))*.1
    X=np.column_stack([ss,np.ones(703),hh,h49]);A=rng.normal(size=703);B=A+rng.normal(size=703)*.1
    G=X.T@X;b=X.T@(.5*(A+B));head=rng.normal(size=64).astype('f4');bias=np.float32(.1)
    base,c=one_layer(G,b,len(X),head,bias,U);_,expected=cf.one_layer(G[:193,:193],b[:193],len(X),head,bias,U)
    np.testing.assert_array_equal(base[:193],expected);np.testing.assert_array_equal(base[193:],0)
    np.testing.assert_allclose((A-X@c)@(B-X@c),A@B-2*c@b+c@G@c,rtol=1e-10,atol=1e-10)
    x1,x2=X[:351],X[351:];y=.5*(A+B)
    a1,c1=one_layer(G-x2.T@x2,b-x2.T@y[351:],len(x1),head,bias,U)
    a2,c2=one_layer(x1.T@x1,x1.T@y[:351],len(x1),head,bias,U)
    np.testing.assert_allclose(a1,a2,atol=1e-6,rtol=1e-5);np.testing.assert_allclose(c1,c2,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(X@c,ss@c[:64]+c[64]+hh@c[65:193]+h49@c[193:],atol=1e-12,rtol=1e-10)
    return dict(independent_fold_hidden_complement_baseline=True,cross_half_moment_metric=True,fold_projection_and_scaling=True,float32_deployment_equation=True,no_missing_physical_features=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args();checks=selftest()
 if args.self_test:print(json.dumps(checks));return
 assert sys.platform=='linux' and sha(HERE/'PREREG_v1.json')==PREREG
 prereg=read(HERE/'PREREG_v1.json');assert sha(SIB/'DESIGN_LOCAL_v1.json')==DESIGN
 pricepath=RUNS/'input_skip49_price_v1/PRICE_v1.json';price=read(pricepath);pg=read(RUNS/'input_skip49_price_v1_GUARD.json')
 assert price['successful'] and price['passes_price_gate'] and price['source_sha256']=='cac374b14312523330827106e51683915e36007aec03154a99af30894b1a3d0a'
 assert price['script_sha256']==sha(HERE/'price_input_skip49_v1.py') and price['state_path_byte_identical'] and price['mean_live_binding']
 assert pg['successful'] and pg['child_exit']==0 and pg['whole_group_retired'] and pg['guard_sha256']==GUARD and not pg['cleanup_errors']
 d=read(SIB/'DESIGN_LOCAL_v1.json');train=d['train']
 recon={};recon_proof=[]
 for chunk in range(3):
  rp=RUNS/('input_skip49_reconstruct90_part%d_v1'%(chunk+1));complete=read(rp/'RECON_COMPLETE_v1.json');rg=read(RUNS/('input_skip49_reconstruct90_part%d_v1_GUARD.json'%(chunk+1)))
  assert rg['successful'] and rg['whole_group_retired'] and rg['child_exit']==0 and not rg['cleanup_errors'] and rg['guard_sha256']==GUARD
  assert complete['successful'] and complete['chunk']==chunk and complete['train']==train[chunk::3] and complete['source_sha256']==SCI
  assert complete['prereg_sha256']==PREREG and complete['replay_sha256']==prereg['replay_sha256'] and complete['no_truth_arrays']
  assert complete['script_sha256']=='eaab059602e79852042fda207d9775c8f4058b18faa5523d1662ad167e181272'
  for row in complete['rows']:assert row['seed'] not in recon;recon[row['seed']]=row
  recon_proof.append(dict(path=str(rp/'RECON_COMPLETE_v1.json'),sha256=sha(rp/'RECON_COMPLETE_v1.json')))
 assert set(recon)==set(train)
 parentfit=RUNS/'hidden_complement90_fit_v1/FITTED_v1.json';assert sha(parentfit)=='10a538849cccd453c63b946f5ae1ff9ef2de247596a956b0c2176006e2388986'
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
   hr=recon[seed];hp=Path(hr['path']);assert sha(hp)==hr['sha256'] and hp.stat().st_size==hr['bytes'] and hr['weights_sha256']==sha(wp)
   with np.load(hp,allow_pickle=False) as hz:
    assert int(hz['seed'])==seed and str(hz['source_sha256'])==SCI and list(hz['columns'])==prereg['columns'] and str(hz['raw_float32_sha256'])==wr['raw_float32_sha256']
    H49=hz['H49'].astype('f8');assert H49.shape==(15,N,49) and np.isfinite(H49).all()
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
    np.testing.assert_array_equal(H49[l-1,:,0],ng)
    X=np.column_stack([state,np.ones(N),hidden,H49[l-1]])*mask[:,None]
    inherited=np.asarray(W[l],dtype='f8').T@np.stack([A[l-1],B[l-1]],axis=1)
    yA=(A[l]-gate*inherited[:,0])/physical;yB=(B[l]-gate*inherited[:,1])/physical
    rr['G'].append(X.T@X);rr['b'].append(X.T@(.5*(yA+yB)));rr['u0_sum'].append(float(yA@yB))
    direct=physical[:,None]*X;rr['physical_gram'].append(direct.T@direct/N);rr['head_rms2'].append(float(np.mean((saved*mask)**2)))
    caps.append(dict(layer=l,capture_sha256=sha(path),head_replay_max_abs=replay,state_replay_max_abs=state_replay,clipped=int(np.sum(mask==0))))
   for k in ('G','b','u0_sum','physical_gram','head_rms2'):rr[k]=np.asarray(rr[k])
   assert float(np.sum(rr['u0_sum']))>0
   records.append(rr);provenance.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp),prediction_sha256=sha(raw),guard_sha256=sha(gp),captures=caps,H49_sha256=sha(hp)))
   print(json.dumps(dict(utc=utc(),loaded=seed,completed=len(records))),flush=True)
 assert [r['seed'] for r in records]==train
 G=sum(r['G'] for r in records);b=sum(r['b'] for r in records)
 baseline,coef=coefficients(G,b,len(records),P,U);training=[metric(r,baseline,coef) for r in records];loo=[]
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
  descriptive_blocks=blocks,physical=physical,coef_by_layer=coef.tolist(),baseline_coef_by_layer=baseline.tolist(),reconstruction_receipts=recon_proof,
  provenance=provenance,no_adjusted_gain_claim=True,no_candidate_exported=True,no_protected_truth_read=True,
  deployment='Original state/messages unchanged; head/head_b/hidden_head128 updated plus H78@input_head with29 fixedzero entries',
  omitted_feedback='Changed covariance, learned features, recurrent state, cumulants and clipping trajectories')
 write(OUT/'FITTED_v1.json',result)
 print(json.dumps({k:result[k] for k in ('utc','status','train','loo','physical')}),flush=True)
if __name__=='__main__':main()

