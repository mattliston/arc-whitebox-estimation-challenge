"""One fixed TRAIN90 fit of shared all-layer head constants to terminal error.

All folds are network held out inside TRAIN90. No protected truth is read.
The propagated response omits covariance/recurrent-state feedback and is a proxy.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'truth_response90_fit_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
read=lambda p:json.loads(Path(p).read_text())
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
DESIGN='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
RIDGE=.1;DAMPING=.5

def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def arr(r,key):
 a=np.asarray(r[key]['data'],dtype='f8')
 assert list(a.shape)==r[key]['shape'] and np.isfinite(a).all()
 return a

def gram(r):
 J=r['J'];scale=r['u0']
 return J.T@J/scale,J.T@(.5*(r['A']+r['B']))/scale

def solve(G,b,n):
 scale=np.maximum(np.sqrt(np.diag(G)/n),1e-14)
 Z=G/(scale[:,None]*scale[None,:]);y=b/scale
 c=np.linalg.solve(Z+RIDGE*n*np.eye(len(b)),y)/scale
 assert np.isfinite(c).all()
 return c*DAMPING

def metric(r,c):
 change=r['J']@c;u1=float(np.mean((r['A']-change)*(r['B']-change)))
 assert u1>0
 return dict(seed=r['seed'],u0=r['u0'],u1=u1,relative_pct=100*(u1/r['u0']-1),
  correction_rms=float(np.sqrt(np.mean(change*change))),correction_max=float(np.max(np.abs(change))))

def summarize(rows):
 p=np.array([r['relative_pct'] for r in rows]);lr=np.log1p(p/100)
 return dict(n=len(rows),mean_relative_pct=float(p.mean()),geometric_pct=float(100*np.expm1(lr.mean())),
  se_relative_pct=float(p.std(ddof=1)/np.sqrt(len(p))),lower=int(np.sum(p<0)),
  pooled_pct=float(100*(sum(r['u1'] for r in rows)/sum(r['u0'] for r in rows)-1)))

def selftest():
 p=HERE.parent/'propagated_state_response_v1/fit_terminal_response_v2.py'
 assert sha(p)=='2a0091f351f1bcb04de3cf6263a89cc43595e279a201aefe2bc4bd99e6635978'
 spec=importlib.util.spec_from_file_location('root_response_verified',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks=m.selftest();rng=np.random.default_rng(20261021090);rr=[]
 for i in range(6):
  J=rng.normal(size=(97,64));c=rng.normal(size=64)*.02
  rr.append(dict(J=J,A=J@c+rng.normal(size=97)*.2,B=J@c+rng.normal(size=97)*.2,u0=.1+i*.01))
 G=sum(gram(r)[0] for r in rr);b=sum(gram(r)[1] for r in rr)
 np.testing.assert_allclose(solve(G,b,97*6),m.fit(rr)*DAMPING,rtol=2e-12,atol=2e-12)
 for i in range(6):
  gi,bi=gram(rr[i]);expected=m.fit(rr[:i]+rr[i+1:])*DAMPING
  np.testing.assert_allclose(solve(G-gi,b-bi,97*5),expected,rtol=2e-12,atol=2e-12)
 return dict(**checks,pooled_normal_equations_and_LOO_equal_original_stacked_solver=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 sys.path.insert(0,str(V));import fit_frontier_v1 as f
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 checks=selftest()
 if args.self_test:print(json.dumps(checks));return
 assert sys.platform=='linux' and sha(HERE/'DESIGN_LOCAL_v1.json')==DESIGN
 d=read(HERE/'DESIGN_LOCAL_v1.json');train=d['train'];assert len(train)==90 and d['weight_hash_disjointness']
 assert not set(train)&set(d['protected_seed_ids'])
 ready=read(HERE/'CAPTURE_READY_v1.json');assert ready['successful'] and ready['design_sha256']==DESIGN
 assert sha(ready['plan'])==ready['plan_sha256'] and sha(ready['recipe'])==ready['recipe_sha256']
 cp=read(ready['plan']);assert [r for group in cp['groups'] for r in group]==train
 completion=read(CAP/'COMPLETE.json');assert completion['successful'] and completion['plan_sha256']==ready['plan_sha256']
 manifest=read(CAP/'RUN_MANIFEST.json');assert manifest['candidate_sha256']==SCI and manifest['blas_threads']==4
 src=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(src)==SCI
 lit,_=f.literals_and_functions(src.read_text());P=f.decode(lit['_GNN_LITERAL'])
 OUT.mkdir(exist_ok=False);write(OUT/'DESIGN_COPY_v1.json',d)
 records=[];provenance=[];physical=[]
 for gi,group in enumerate(cp['groups'],1):
  part='part%02d'%gi;folder=CAP/part;gp=CAP/(part+'_GUARD.json');g=read(gp)
  assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['guard_sha256']==GUARD
  cm=read(folder/'manifest.json');assert cm['sha256']==SCI and cm['diagnostic_only'] and cm['environment']['thread_provenance']['blas_threads']==4
  capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']==ready['hook_sha256'] and meta['target_free']
  assert all(np.array_equal(v,f.decode(meta['gnn_literal'])[k]) for k,v in P.items())
  done=read(capdir/'CAPTURE_COMPLETE_v1.json');assert done['complete'] and done['target_free']
  for seed in group:
   assert seed in train
   raw=folder/('diagnostic_pred_%d.bin'%seed);assert raw.stat().st_size==65536
   pred=np.fromfile(raw,dtype='<f4').reshape(16,1024).astype('f8');assert np.isfinite(pred).all()
   wr=cp['rows'][str(seed)]['weights'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256']
   W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,1024,1024)
   tr=d['rows'][str(seed)];tp=ROOT/tr['truth']['path'];assert sha(tp)==tr['truth']['sha256']
   assert tr['raw_float32_sha256']==wr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
   with np.load(tp,allow_pickle=False) as z:
    assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256']
    A=z['mc_post_A'][15].astype('f8')-pred[15];B=z['mc_post_B'][15].astype('f8')-pred[15]
   u0=float(np.mean(A*B));assert u0>0
   J=np.zeros((1024,64));caps=[]
   for layer in range(1,16):
    r=next(x for x in done['files'] if x['rank']==seed and x['phase']=='scored' and x['layer']==layer)
    p=capdir/r['path'];assert sha(p)==r['sha256']
    with gzip.open(p,'rt',encoding='utf-8') as stream:z=json.load(stream)
    assert z['rank']==seed and z['layer']==layer and z['phase']=='scored'
    state=arr(z,'state');sig=arr(z,'sigma');ng=arr(z,'gate_feature');saved=arr(z,'delta')
    assert state.shape==(1024,64) and sig.shape==ng.shape==saved.shape==(1024,)
    assert np.max(np.abs(ng))<7.999
    gate=ng*float(P['scales'][layer,5])+float(P['centers'][layer,5]);assert gate.min()>-2e-6 and gate.max()<1+2e-6
    gate=np.clip(gate,0,1);scale=sig*float(P['rms'][layer])*(1+.25*np.tanh(float(P['gain'][layer])))
    assert scale.min()>1e-12
    oracle=scale*(state@P['head'][layer].astype('f8')+float(P['head_b'][layer]))
    err=float(np.max(np.abs(saved-oracle)));assert err<1e-8+2e-5*float(np.max(np.abs(saved)))
    mask=(pred[layer]>1e-12).astype('f8') if layer<15 else np.ones(1024)
    direct=scale[:,None]*state
    J=mask[:,None]*(gate[:,None]*(np.asarray(W[layer],dtype='f8').T@J)+direct)
    physical.append(dict(seed=seed,layer=layer,direct_gram=direct.T@direct/1024,
      original_head_rms=float(np.sqrt(np.mean(saved*saved)))))
    caps.append(dict(layer=layer,sha256=r['sha256'],head_replay_max_abs=err,clipped=int(np.sum(mask==0))))
   rr=dict(seed=seed,J=J,A=A,B=B,u0=u0);rr['G'],rr['b']=gram(rr);records.append(rr)
   provenance.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp),prediction_sha256=sha(raw),guard_sha256=sha(gp),captures=caps))
   print(json.dumps(dict(utc=utc(),loaded_seed=seed,completed=len(records),u0=u0)),flush=True)
 assert [r['seed'] for r in records]==train
 G=sum(r['G'] for r in records);b=sum(r['b'] for r in records)
 coef=solve(G,b,1024*len(records));trainrows=[metric(r,coef) for r in records]
 loo=[metric(r,solve(G-r['G'],b-r['b'],1024*(len(records)-1))) for r in records]
 blocks=[summarize([r for r in loo if r['seed'] in train[i::5]]) for i in range(5)]
 cv=summarize(loo);trainstats=summarize(trainrows)
 passed=bool(cv['mean_relative_pct']<=-.15 and cv['lower']>=54 and sum(x['mean_relative_pct']<0 for x in blocks)>=4)
 for row in physical:
  dg=row.pop('direct_gram');row['direct_correction_rms']=float(np.sqrt(max(0,float(coef@dg@coef))))
 result=dict(utc=utc(),status='PROXY_GATE_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',passed=passed,
  design_sha256=DESIGN,source_sha256=SCI,release_source_sha256=REL,script_sha256=sha(__file__),selftests=checks,
  coef=coef.tolist(),train=trainstats,train_per_network=trainrows,loo=cv,loo_per_network=loo,descriptive_blocks=blocks,
  physical=physical,provenance=provenance,no_held_or_new_fresh_truth=True,no_adjusted_gain_claim=True,
  omitted_feedback='covariance, recurrent state/input features, higher cumulants',no_candidate_exported=True)
 write(OUT/'FITTED_v1.json',result)
 print(json.dumps(dict(status=result['status'],train=trainstats,loo=cv,blocks=blocks,fit_sha256=sha(OUT/'FITTED_v1.json'))),flush=True)

if __name__=='__main__':main()
