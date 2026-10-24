"""Fixed authentic-trajectory hidden64 teacher distillation, no truth access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE.parent/'truth_response95_v1';V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
RUNS=Path('/home/arcworker/arc-runs');CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'hidden2_prune90_fit_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f
sha=f.sha;read=lambda p:json.loads(Path(p).read_text());write=f.jwrite
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510';GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
def stats(H,S,physical,head,delta,seed=0,layer=0):
 n=len(H);u=S@head;w=physical**2
 return dict(seed=seed,layer=layer,G=H.T@H/n,T=H.T@S/n,Y=S.T@S/n,h=H.mean(0),s=S.mean(0),
  wg=H.T@(w[:,None]*H)/n,wh=H.T@w/n,wu=float(u@w/n),wtu=H.T@(w*u)/n,wuu=float((w*u)@u/n),w=float(w.mean()),
  delta2=float(np.mean(delta**2)))
def fit(rows,width=64):
 count=len(rows);G=sum(r['G'] for r in rows)/count;T=sum(r['T'] for r in rows)/count;h=sum(r['h'] for r in rows)/count;s=sum(r['s'] for r in rows)/count
 C=G-np.outer(h,h);B=T-np.outer(h,s);C=(C+C.T)/2;res=C.copy();cross=B.copy();chosen=[];tol=1e-14*float(np.max(np.diag(C)))
 for _ in range(width):
  diagonal=np.diag(res).copy();score=np.sum(cross*cross,axis=1)/np.maximum(diagonal,tol);score[diagonal<=tol]=-np.inf;score[chosen]=-np.inf
  index=int(np.argmax(score));assert np.isfinite(score[index]);chosen.append(index)
  column=res[:,index]/np.sqrt(res[index,index]);target=cross[index].copy()/np.sqrt(res[index,index])
  cross-=column[:,None]*target[None,:];res-=np.outer(column,column);res=(res+res.T)/2
 chosen=np.asarray(chosen);selected=C[np.ix_(chosen,chosen)];ridge=1e-6*float(np.trace(selected))/width
 weights=np.linalg.solve(selected+ridge*np.eye(width),B[chosen]);bias=s-h[chosen]@weights
 return dict(chosen=chosen,weights=weights.astype('f4').astype('f8'),bias=bias.astype('f4').astype('f8'),ridge=ridge)
def errors(r,model,P):
 indices=model['chosen'];B=model['weights'];b=model['bias'];G=r['G'][np.ix_(indices,indices)];T=r['T'][indices];h=r['h'][indices]
 q=B.T@h;E=B.T@G@B-B.T@T-T.T@B+r['Y']+np.outer(q,b)+np.outer(b,q)+np.outer(b,b)-np.outer(r['s'],b)-np.outer(b,r['s'])
 state=float(np.trace(E));state_energy=float(np.trace(r['Y']));M=P['msg_w'].astype('f8');mb=P['msg_b'].astype('f8')
 message=float(np.trace(M@E@M.T));message_energy=float(np.trace(M@r['Y']@M.T)+2*(r['s']@M.T)@mb+mb@mb)
 head=P['head'][r['layer']].astype('f8');c=B@head;bias=float(b@head)
 head_error=float(c@r['wg'][np.ix_(indices,indices)]@c+2*bias*(c@r['wh'][indices])+bias*bias*r['w']-2*c@r['wtu'][indices]-2*bias*r['wu']+r['wuu'])
 assert state>=-1e-9*max(state_energy,1e-30) and message>=-1e-9*max(message_energy,1e-30) and head_error>=-1e-9*max(r['delta2'],1e-30)
 return dict(state_error2=max(0.,state),state_energy=state_energy,message_error2=max(0.,message),message_energy=message_energy,head_error2=max(0.,head_error),head_energy=r['delta2'])
def summarize(rows):
 return dict(n_network_layers=len(rows),state_RMS_relative=float(np.sqrt(sum(r['state_error2'] for r in rows)/sum(r['state_energy'] for r in rows))),
  physical_head_RMS_relative=float(np.sqrt(sum(r['head_error2'] for r in rows)/sum(r['head_energy'] for r in rows))),
  message_source_RMS_relative=float(np.sqrt(sum(r['message_error2'] for r in rows)/sum(r['message_energy'] for r in rows))))
def passing(r,multiplier=1.):
 return r['state_RMS_relative']<=.005*multiplier and r['physical_head_RMS_relative']<=.01*multiplier and r['message_source_RMS_relative']<=.005*multiplier
def selftest():
 rng=np.random.default_rng(202610210329);H=rng.normal(size=(79,9));S=rng.normal(size=(79,5));head=rng.normal(size=5);physical=rng.uniform(.2,.5,79);delta=rng.normal(size=79)*.1
 p=dict(msg_w=rng.normal(size=(3,5)),msg_b=rng.normal(size=3),head=np.stack([head]));r=stats(H,S,physical,head,delta)
 model=fit([r],width=4);err=H[:,model['chosen']]@model['weights']+model['bias']-S;got=errors(r,model,p)
 np.testing.assert_allclose(got['state_error2'],np.mean(np.sum(err*err,axis=1)),rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(got['head_error2'],np.mean((physical*(err@head))**2),rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(got['message_error2'],np.mean(np.sum((err@p['msg_w'].T)**2,axis=1)),rtol=1e-12,atol=1e-12)
 X=H[:,model['chosen']];xc=X-X.mean(0);yc=S-S.mean(0);ridge=1e-6*np.mean(np.var(X,axis=0));B=np.linalg.solve(xc.T@xc/len(X)+ridge*np.eye(4),xc.T@yc/len(X))
 np.testing.assert_array_equal(model['weights'],B.astype('f4').astype('f8'))
 return dict(moment_losses_match_direct_arrays=True,affine_ridge_matches_direct_centered_solver=True,fold_selects_and_fits_only_training_statistics=True)
def main():
 checks=selftest()
 if '--self-test' in sys.argv:print(json.dumps(checks));return
 assert sys.platform=='linux' and sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 pr=read(HERE/'PREREG_v1.json');assert sha(HERE/'PREREG_v1.json')=='97c868a0a6ee0a12850c97af7a041f049d929b0c950df920d65fbf246425cb86'
 pp=RUNS/'hidden2_prune90_price_v1/PRICE_v1.json';price=read(pp);assert price['successful'] and price['passes_price_gate'] and price['no_captures_or_truth_read'] and price['prereg_sha256']==sha(HERE/'PREREG_v1.json')
 g=read(RUNS/'hidden2_prune90_price_v1_GUARD.json');assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['guard_sha256']==GUARD
 design=read(D/'DESIGN_LOCAL_v1.json');assert sha(D/'DESIGN_LOCAL_v1.json')==pr['design_sha256'] and design['train']==pr['teacher_seeds'] and design['weight_hash_disjointness']
 assert not set(pr['teacher_seeds'])&set(design['protected_seed_ids'])
 ready=read(D/'CAPTURE_READY_v1.json');cp=read(ready['plan']);assert sha(ready['plan'])==ready['plan_sha256'];completion=read(CAP/'COMPLETE.json');assert completion['successful'] and completion['plan_sha256']==ready['plan_sha256']
 source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(source)==SCI
 literal,_=f.literals_and_functions(source.read_text());P=f.decode(literal['_GNN_LITERAL']);OUT.mkdir(exist_ok=False)
 records=[];provenance=[]
 for gi,group in enumerate(cp['groups'],1):
  part='part%02d'%gi;folder=CAP/part;gp=CAP/(part+'_GUARD.json');guard=read(gp)
  assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors'] and guard['guard_sha256']==GUARD
  cap=folder/'gnn_full_state_v1';meta=read(cap/'METADATA_v1.json');done=read(cap/'CAPTURE_COMPLETE_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']==ready['hook_sha256'] and meta['target_free'] and done['complete'] and done['target_free']
  assert all(np.array_equal(v,f.decode(meta['gnn_literal'])[k]) for k,v in P.items())
  for seed in group:
   paths=[]
   for layer in range(1,16):
    row=next(r for r in done['files'] if r['rank']==seed and r['phase']=='scored' and r['layer']==layer);p=cap/row['path'];assert sha(p)==row['sha256']
    with gzip.open(p,'rt',encoding='utf8') as stream:r=json.load(stream)
    def array(k):
     a=np.asarray(r[k]['data'],'f8');assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all();return a
    H=array('hidden2');S=array('state');sigma=array('sigma');delta=array('delta')
    assert H.shape==(1024,128) and S.shape==(1024,64) and sigma.shape==(1024,)
    replay=H@P['enc4_w'].astype('f8').T+P['enc4_b'];err=float(np.max(np.abs(replay-S)));assert err<3e-5
    physical=sigma*float(P['rms'][layer])*(1+.25*np.tanh(float(P['gain'][layer])))
    records.append(stats(H,S,physical,P['head'][layer].astype('f8'),delta,seed,layer));paths.append(dict(layer=layer,sha256=row['sha256'],state_replay_max_abs=err))
   provenance.append(dict(seed=seed,guard_sha256=sha(gp),captures=paths));print(json.dumps(dict(utc=f.utc(),loaded=len(provenance),seed=seed)),flush=True)
 assert len(records)==1350 and [r['seed'] for r in provenance]==pr['teacher_seeds']
 folds=[];all_errors=[]
 for index,held in enumerate(pr['folds']):
  train=[r for r in records if r['seed'] not in held];valid=[r for r in records if r['seed'] in held];assert len(train)==72*15 and len(valid)==18*15
  model=fit(train);errors_=[errors(r,model,P) for r in valid];score=summarize(errors_);folds.append(dict(fold=index,validation_seeds=held,chosen=model['chosen'].tolist(),metrics=score,passed=passing(score)));all_errors+=errors_
 aggregate=summarize(all_errors);passed=passing(aggregate) and sum(r['passed'] for r in folds)>=4 and all(passing(r['metrics'],2.) for r in folds)
 result=dict(utc=f.utc(),status='TEACHER_FIDELITY_PASS_NATIVE_REQUIRED' if passed else 'CLOSED_TEACHER_FIDELITY_FAIL',passed=bool(passed),source_sha256=SCI,script_sha256=sha(__file__),prereg_sha256=sha(HERE/'PREREG_v1.json'),price_sha256=sha(pp),selftests=checks,folds=folds,aggregate=aggregate,provenance=provenance,no_network_truth_access=True,no_candidate_exported=True,no_native_score_or_adjusted_gain_claim=True)
 if passed:
  model=fit(records);np.savez(OUT/'PRUNE_MODEL_v1.npz',chosen=model['chosen'],enc2_w=P['enc2_w'][model['chosen']].astype('f4'),enc2_b=P['enc2_b'][model['chosen']].astype('f4'),enc4_w=model['weights'].T.astype('f4'),enc4_b=model['bias'].astype('f4'))
  result['model_sha256']=sha(OUT/'PRUNE_MODEL_v1.npz');result['chosen']=model['chosen'].tolist()
 write(OUT/'FITTED_v1.json',result);print(json.dumps(dict(status=result['status'],aggregate=aggregate,folds=[r['metrics'] for r in folds])),flush=True)
if __name__=='__main__':main()
