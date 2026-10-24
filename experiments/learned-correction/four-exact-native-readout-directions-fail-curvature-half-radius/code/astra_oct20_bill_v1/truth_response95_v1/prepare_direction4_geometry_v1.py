"""Four target-free current-source directions; no truth/fit-result array access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='4'
import ast,gzip,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1';CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'truth_response90_direction4_geometry_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def arr(r,k):
 a=np.asarray(r[k]['data'],dtype='f8');assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all();return a
def selftest():
 rng=np.random.default_rng(2026102104);n=13;p=7;r=5
 W=rng.normal(size=(n,n))/np.sqrt(n);V=rng.normal(size=(n,r));Q=rng.normal(size=(n,r,p));coef=rng.normal(size=p)
 av=W.T@V;aq=(W.T@Q.reshape(n,r*p)).reshape(n,r,p)
 var=2*np.sum(av[:,:,None]*aq,axis=1)
 for j in range(p):
  dense=V@Q[:,:,j].T+Q[:,:,j]@V.T
  np.testing.assert_allclose(var[:,j],np.diag(W.T@dense@W),rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(var@coef,2*np.sum(av*(W.T@(Q@coef)),axis=1),rtol=1e-12,atol=1e-12)
 X=rng.normal(size=(30,p));R=X.T@X;Y=rng.normal(size=(40,p));C=Y.T@Y+.1*np.eye(p);L=np.linalg.cholesky(C)
 B=np.linalg.solve(L,R);B=np.linalg.solve(L,B.T).T;val,U=np.linalg.eigh((B+B.T)/2);q=np.linalg.solve(L.T,U)
 np.testing.assert_allclose(R@q,(C@q)*val[None,:],rtol=1e-11,atol=1e-11)
 return dict(all_parameter_covariance_diagonal_matches_dense=True,directional_contraction_matches_scalar=True,generalized_eigenvector_equation=True)
def main():
 checks=selftest()
 if '--self-test' in sys.argv:print(json.dumps(checks));return
 sys.path.insert(0,str(V));import fit_frontier_v1 as f
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 assert sys.platform=='linux'
 d=read(HERE/'DESIGN_LOCAL_v1.json');assert sha(HERE/'DESIGN_LOCAL_v1.json')=='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
 assert sha(HERE/'DIRECTION4_PREREG_v1.json')=='03bd92249d5cbe2915b5f8d102c7f66105efd041bfa7dd235ad0372ac3e539bb'
 prereg=read(HERE/'DIRECTION4_PREREG_v1.json');train=d['train'];assert prereg['geometry_seeds']==train and len(train)==90
 ready=read(HERE/'CAPTURE_READY_v1.json');assert ready['successful'] and sha(ready['plan'])==ready['plan_sha256']
 cp=read(ready['plan']);assert [r for group in cp['groups'] for r in group]==train
 done=read(CAP/'COMPLETE.json');assert done['successful'] and done['plan_sha256']==ready['plan_sha256']
 source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(source)==SCI
 lit,_=f.literals_and_functions(source.read_text());P=f.decode(lit['_GNN_LITERAL'])
 constants={}
 for node in ast.parse(source.read_text()).body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['_GN_RC','_GN_KC']:
   constants[node.targets[0].id]=ast.literal_eval(node.value)
 beta={l:1+constants['_GN_RC'].get(l,0)-constants['_GN_KC'].get(l,0) for l in range(1,15)}
 OUT.mkdir(exist_ok=False);write(OUT/'PREREG_COPY_v1.json',prereg)
 R=np.zeros((64,64));C=np.zeros((64,64));delta_energy=0.;provenance=[];pernet=[]
 for gi,group in enumerate(cp['groups'],1):
  part='part%02d'%gi;folder=CAP/part;guard=read(CAP/(part+'_GUARD.json'))
  assert guard['successful'] and guard['child_exit']==0 and guard['whole_group_retired'] and not guard['cleanup_errors']
  assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
  capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json');complete=read(capdir/'CAPTURE_COMPLETE_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']==ready['hook_sha256'] and complete['complete'] and complete['target_free']
  assert all(np.array_equal(v,f.decode(meta['gnn_literal'])[k]) for k,v in P.items())
  for seed in group:
   pp=folder/f'diagnostic_pred_{seed}.bin';assert pp.stat().st_size==65536
   pred=np.fromfile(pp,dtype='<f4').reshape(16,1024).astype('f8');norm=float(np.mean(pred[15]**2));assert norm>0 and np.isfinite(pred).all()
   wr=cp['rows'][str(seed)]['weights'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256']
   assert wr['raw_float32_sha256']==d['rows'][str(seed)]['raw_float32_sha256'] and wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
   assert '/weights_v1/' in str(wp) and wp.suffix=='.npy'
   W=np.load(wp,mmap_mode='r',allow_pickle=False);J=np.zeros((1024,64));carV=np.empty((1024,0));carQ=np.empty((1024,0,64));c=np.zeros((64,64));energy=0.;layerchecks=[]
   for layer in range(1,16):
    rr=next(r for r in complete['files'] if r['rank']==seed and r['phase']=='scored' and r['layer']==layer)
    path=capdir/rr['path'];assert sha(path)==rr['sha256']
    with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    state=arr(r,'state');sig=arr(r,'sigma');ng=arr(r,'gate_feature');delta=arr(r,'delta')
    gate=np.clip(ng*float(P['scales'][layer,5])+float(P['centers'][layer,5]),0,1)
    w=np.asarray(W[layer],dtype='f8');mu=w.T@pred[layer-1];alpha=mu/sig;pdf=np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi)
    cdf=.5*(1+np.array([math.erf(float(a)/np.sqrt(2)) for a in alpha]));err=float(np.max(np.abs(cdf-gate)));assert err<3e-5
    phys=sig*float(P['rms'][layer])*(1+.25*np.tanh(float(P['gain'][layer])));direct=phys[:,None]*state
    oracle=phys*(state@P['head'][layer].astype('f8')+float(P['head_b'][layer]));replay=float(np.max(np.abs(oracle-delta)));assert replay<1e-8+2e-5*np.max(np.abs(delta))
    c+=direct.T@direct/norm;energy+=float(delta@delta)/norm
    mask=(pred[layer]>1e-12).astype('f8') if layer<15 else np.ones(1024)
    rank=carV.shape[1];av=w.T@carV;aq=(w.T@carQ.reshape(1024,rank*64)).reshape(1024,rank,64)
    dv=2*np.sum(av[:,:,None]*aq,axis=1)
    K=gate[:,None]*(w.T@J)+(.5*pdf/sig)[:,None]*dv
    D=direct+(.5*delta/(sig*sig))[:,None]*dv;J=mask[:,None]*(K+D);deff=J-K
    carV=gate[:,None]*av;carQ=gate[:,None,None]*aq;clip_ratio=0.
    if layer<15:
     bound=2*abs(beta[layer])*np.linalg.norm(np.where(mask==0,np.abs(delta),0.))*np.linalg.norm(K)
     den=2*abs(beta[layer])*np.linalg.norm(pred[layer])*np.linalg.norm(deff)
     clip_ratio=float(bound/max(den,1e-30));assert clip_ratio<.01
     carV=np.column_stack([carV,pred[layer],delta])
     carQ=np.concatenate([carQ,(-beta[layer]*deff)[:,None,:],(-beta[layer]*K)[:,None,:]],axis=1)
    layerchecks.append(dict(layer=layer,capture_sha256=rr['sha256'],gate_replay_max_abs=err,head_replay_max_abs=replay,clip_uncertainty_ratio_bound=clip_ratio))
   gram=J.T@J/norm;R+=gram;C+=c;delta_energy+=energy
   pernet.append(dict(seed=seed,terminal_response_trace=float(np.trace(gram)),insertion_trace=float(np.trace(c)),head_energy=energy))
   provenance.append(dict(seed=seed,weights_sha256=sha(wp),prediction_sha256=sha(pp),captures=layerchecks))
   print(json.dumps(dict(utc=utc(),geometry_completed=len(pernet),seed=seed)),flush=True)
 assert [r['seed'] for r in pernet]==train
 reg=.1*float(np.trace(C))/64;metric=.5*(C+C.T)+reg*np.eye(64);L=np.linalg.cholesky(metric)
 B=np.linalg.solve(L,R);B=np.linalg.solve(L,B.T).T;B=.5*(B+B.T)
 values,vectors=np.linalg.eigh(B);order=np.argsort(values)[::-1][:4];vectors=np.linalg.solve(L.T,vectors[:,order]);directions=[]
 for i,col in enumerate(vectors.T):
  if col[np.argmax(np.abs(col))]<0:col=-col
  amp=.1*np.sqrt(delta_energy/float(col@C@col));step=col*amp
  ratio=float(np.sqrt((step@C@step)/delta_energy));assert abs(ratio-.1)<1e-10 and np.isfinite(step).all()
  directions.append(dict(index=i,head_delta=step.tolist(),generalized_eigenvalue=float(values[order[i]]),physical_head_RMS_fraction=ratio,
   proxy_terminal_response_energy=float(step@R@step),max_abs_head_parameter=float(np.max(np.abs(step)))))
 result=dict(utc=utc(),successful=True,source_sha256=SCI,design_sha256=sha(HERE/'DESIGN_LOCAL_v1.json'),
  prereg_sha256=sha(HERE/'DIRECTION4_PREREG_v1.json'),script_sha256=sha(__file__),no_truth_arrays_or_fit_outputs_read=True,
  geometry_networks=90,fd_train_seeds=prereg['fd_train_seeds'],baseline_anchor_seeds=prereg['baseline_anchor_seeds'],directions=directions,
  metric_regularization=reg,head_energy=delta_energy,per_network=pernet,provenance=provenance,selftests=checks,no_adjusted_gain_claim=True)
 np.savez(OUT/'GRAMS_v1.npz',response=R,insertion=C,metric=metric,eigenvalues=values)
 result['grams_sha256']=sha(OUT/'GRAMS_v1.npz');write(OUT/'DIRECTIONS_v1.json',result)
 print(json.dumps(dict(successful=True,directions_sha256=sha(OUT/'DIRECTIONS_v1.json'),eigenvalues=[r['generalized_eigenvalue'] for r in directions],no_truth_read=True)),flush=True)
if __name__=='__main__':main()
