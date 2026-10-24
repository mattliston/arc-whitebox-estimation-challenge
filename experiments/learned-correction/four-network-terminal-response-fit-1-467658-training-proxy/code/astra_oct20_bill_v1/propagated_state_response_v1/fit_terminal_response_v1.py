"""Fixed TRAIN4 shared64 fit to terminal propagated mean response.
No validation/held/fresh truth is read. All response coefficients freeze before
native validation; the linearized proxy omits covariance and state feedback.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parent.parent/'astra_oct20_vm_frontier_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f
assert f.sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
sha=f.sha;utc=f.utc;read=lambda p:json.loads(Path(p).read_text())
TRAIN=(35,150,200,250);RIDGE=.1;DAMPING=.5
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
OUT=RUNS/'propagated_state_response_fit_v1';CAP=RUNS/'full_state_response_train4_v1'
def arr(r,key):
 a=np.asarray(r[key]['data'],dtype='f8')
 assert list(a.shape)==r[key]['shape'] and np.isfinite(a).all(),key
 return a
def fit(records):
 X=np.concatenate([r['J']/np.sqrt(r['u0']) for r in records])
 y=np.concatenate([.5*(r['A']+r['B'])/np.sqrt(r['u0']) for r in records])
 scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-14)
 z=X/scale
 coef=np.linalg.solve(z.T@z+RIDGE*len(z)*np.eye(X.shape[1]),z.T@y)/scale
 assert np.isfinite(coef).all()
 return coef
def metric(records,coef):
 results=[]
 for r in records:
  change=r['J']@coef;u1=float(np.mean((r['A']-change)*(r['B']-change)))
  results.append(dict(rank=r['rank'],u0=r['u0'],u1=u1,relative_pct=100*(u1/r['u0']-1),
   correction_rms=float(np.sqrt(np.mean(change*change))),correction_max=float(np.max(np.abs(change)))))
 return dict(networks=results,mean_pct=float(np.mean([x['relative_pct'] for x in results])),
  lower=sum(x['u1']<x['u0'] for x in results))
def selftest():
 rng=np.random.default_rng(2026102107);n,p=11,5
 W=rng.normal(size=(4,n,n))*.15
 B=rng.normal(size=(4,n));sigma=rng.uniform(.3,1.2,(4,n));U=rng.normal(size=(4,n,p))*.03
 erf=np.vectorize(math.erf)
 def forward(c):
  m=np.zeros(n);J=np.zeros((n,p))
  for l in range(4):
   mu=W[l].T@m+B[l];alpha=mu/sigma[l]
   gate=.5*(1+erf(alpha/math.sqrt(2)));pdf=np.exp(-.5*alpha*alpha)/math.sqrt(2*math.pi)
   m=np.maximum(sigma[l]*pdf+mu*gate+U[l]@c,0)
   mask=m>0;J=mask[:,None]*(gate[:,None]*(W[l].T@J)+U[l])
  return m,J
 base,J=forward(np.zeros(p))
 for j in range(p):
  v=np.zeros(p);v[j]=1e-5;fd=(forward(v)[0]-forward(-v)[0])/(2e-5)
  np.testing.assert_allclose(fd,J[:,j],rtol=2e-7,atol=2e-10)
 X=rng.normal(size=(1000,64));b=rng.normal(size=64)*.01;y=X@b
 r=dict(rank=35,J=X,A=y,B=y,u0=float(np.mean(y*y)))
 c=fit([r]);assert metric([r],c)['mean_pct']<-90
 return dict(gaussian_mean_finite_difference=True,terminal_ridge_recovery=True,cross_half_metric=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 checks=selftest()
 if args.self_test:print(json.dumps(checks));return
 assert sys.platform=='linux'
 assert sha(HERE/'PREREG_v1.json')=='c2bf9da8621730c8afd899743ee85cbf5173a6178100dd6644083ca6b1c91abd'
 ready=read(HERE/'CAPTURE_READY_v1.json')
 assert ready['plan_sha256']=='9dd3a1666f38945e5180559152e80c7a778131a66520fae505c1588bbc3bf94a'
 assert read(CAP/'COMPLETE.json')['successful']
 cp=read(ready['plan']);assert cp['groups']==[[35,150,200],[250]]
 planp=V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
 assert sha(planp)=='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
 plan=read(planp)
 parents={'science':V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py',
          'release':V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_rel_v1.py'}
 for role,pin in [('science',SCI),('release',REL)]:assert sha(parents[role])==pin
 lits,_=f.literals_and_functions(parents['science'].read_text());P=f.decode(lits['_GNN_LITERAL'])
 OUT.mkdir(exist_ok=False)
 f.jwrite(OUT/'PREREG_COPY_v1.json',read(HERE/'PREREG_v1.json'))
 bank=RUNS/'msg8_loaderfix337422_control32_markline_blas4_v1'
 records=[];provenance=[];layer_records={}
 for gi,group in enumerate(cp['groups'],1):
  gp='part%02d'%gi;folder=CAP/gp;guard=read(CAP/(gp+'_GUARD.json'))
  assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
  assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
  manifest=read(folder/'manifest.json');assert manifest['sha256']==SCI and manifest['diagnostic_only']
  assert manifest['environment']['thread_provenance']['blas_threads']==4
  capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']=='d5292105219e14fdb9edf04a476888e844e9f518ff428bae675fa146fdc3e1b5'
  assert all(np.array_equal(v,f.decode(meta['gnn_literal'])[k]) for k,v in P.items())
  complete=read(capdir/'CAPTURE_COMPLETE_v1.json');assert complete['complete'] and complete['target_free']
  originals=read(bank/(gp+'_GUARD.json'));assert originals['successful']
  for rank in group:
   assert rank in TRAIN
   raw=folder/('diagnostic_pred_%d.bin'%rank);ref=bank/gp/('pred_%d.bin'%rank)
   assert raw.stat().st_size==65536 and sha(raw)==sha(ref),'Diagnostic changed baseline prediction bytes'
   pred=np.fromfile(raw,dtype='<f4').reshape(16,1024).astype('f8')
   files={}
   for key in ('weights','truth_A','truth_B'):
    rec=plan['rows'][str(rank)][key];path=ROOT/rec['path']
    assert sha(path)==rec['sha256'] and path.stat().st_size==rec['bytes']
    files[key]=dict(path=str(path),sha256=sha(path))
    if key=='weights':weights=np.load(path,allow_pickle=False,mmap_mode='r');assert weights.shape==(16,1024,1024)
    else:
     with np.load(path,allow_pickle=False) as z:files[key]['value']=z[rec['key']].astype('f8')
   yA=files['truth_A'].pop('value')[15]-pred[15];yB=files['truth_B'].pop('value')[15]-pred[15]
   u0=float(np.mean(yA*yB));assert u0>0
   J=np.zeros((1024,64));layers=[];captured=[]
   for layer in range(1,16):
    receipt=next(r for r in complete['files'] if r['rank']==rank and r['phase']=='scored' and r['layer']==layer)
    path=capdir/receipt['path'];assert sha(path)==receipt['sha256']
    with gzip.open(path,'rt',encoding='utf-8') as inp:r=json.load(inp)
    assert r['rank']==rank and r['layer']==layer and r['phase']=='scored'
    state=arr(r,'state');sig=arr(r,'sigma');norm_gate=arr(r,'gate_feature')
    assert state.shape==(1024,64) and sig.shape==(1024,) and norm_gate.shape==(1024,)
    assert np.max(np.abs(norm_gate))<7.999,'Gate feature was clipped'
    gate=norm_gate*float(P['scales'][layer,5])+float(P['centers'][layer,5])
    assert np.min(gate)>-2e-6 and np.max(gate)<1+2e-6
    gate=np.clip(gate,0,1)
    gain=1+.25*np.tanh(float(P['gain'][layer]));physical=sig*float(P['rms'][layer])*gain
    assert np.min(physical)>1e-12
    saved=arr(r,'delta');oracle=physical*(state@P['head'][layer].astype('f8')+float(P['head_b'][layer]))
    error=float(np.max(np.abs(saved-oracle)));assert error<1e-8+2e-5*float(np.max(np.abs(saved)))
    mask=(pred[layer]>1e-12).astype('f8') if layer<15 else np.ones(1024)
    direct=physical[:,None]*state
    J=mask[:,None]*(gate[:,None]*(np.asarray(weights[layer],dtype='f8').T@J)+direct)
    layers.append(dict(layer=layer,J=J.copy(),direct=direct,original_delta=saved))
    captured.append(dict(layer=layer,capture_sha256=sha(path),head_replay_max_abs=error,clipped=int(np.sum(mask==0))))
   records.append(dict(rank=rank,J=J,A=yA,B=yB,u0=u0));layer_records[rank]=layers
   provenance.append(dict(rank=rank,capture_guard_sha256=sha(CAP/(gp+'_GUARD.json')),
     prediction_sha256=sha(raw),original_prediction_byte_equal=True,files=files,captures=captured))
 assert [r['rank'] for r in records]==list(TRAIN)
 coef=fit(records)*DAMPING;loo=[]
 for rank in TRAIN:
  cv=fit([r for r in records if r['rank']!=rank])*DAMPING
  loo.extend(metric([r for r in records if r['rank']==rank],cv)['networks'])
 loo_mean=float(np.mean([r['relative_pct'] for r in loo]))
 Q={k:v.copy() for k,v in P.items()};Q['head'][1:16]=(P['head'][1:16].astype('f8')+coef).astype('f4')
 assert all(np.array_equal(P[k],Q[k]) for k in P if k!='head')
 physical=[]
 for rank,layers in layer_records.items():
  for layer in layers:
   change=layer['direct']@coef;prop=layer['J']@coef
   physical.append(dict(rank=rank,layer=layer['layer'],direct_rms=float(np.sqrt(np.mean(change*change))),
    propagated_rms=float(np.sqrt(np.mean(prop*prop))),max_abs_direct=float(np.max(np.abs(change))),
    original_head_rms=float(np.sqrt(np.mean(layer['original_delta']**2)))))
 passed=bool(loo_mean<=-.1 and sum(r['u1']<r['u0'] for r in loo)>=3 and np.isfinite(coef).all())
 result=dict(utc=utc(),status='PROXY_GATE_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',
  passed=passed,coef=coef.tolist(),train=metric(records,coef),loo=loo,loo_mean_pct=loo_mean,
  physical=physical,provenance=provenance,selftests=checks,prereg_sha256=sha(HERE/'PREREG_v1.json'),
  source_pins=dict(science=SCI,release=REL),script_sha256=sha(__file__),no_adjusted_gain_claim=True,
  omitted_feedback='covariance, learned state/input features and higher cumulants',
  no_validation_held_or_fresh_truth=True)
 f.jwrite(OUT/'FITTED_v1.json',result)
 print(json.dumps(dict(status=result['status'],train=result['train'],loo_mean_pct=loo_mean,
  loo_lower=sum(r['u1']<r['u0'] for r in loo),fit_sha256=sha(OUT/'FITTED_v1.json'))))
if __name__=='__main__':main()

