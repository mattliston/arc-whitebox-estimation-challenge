"""Frozen TRAIN72/full90 first-stage parents for clean on-policy second-step CV.

Only source construction; no capture/native launch and no second-stage targets.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE.parent;D=B/'truth_response95_v1';C=B/'hidden_complement90_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs');CAP=RUNS/'full_state_response_train90_v1'
V=B.parent/'astra_oct20_vm_frontier_v1';KO=B.parent/'fable_oct19_kc_v1'
sys.path[:0]=[str(C),str(V)]
import fit_hidden_complement_v1 as cf
import fit_frontier_v1 as f
sha=f.sha;utc=f.utc;read=lambda p:json.loads(Path(p).read_text())
CF='e6f33d68de379965c887cac3e43fbc31766625e66dead02d8bf0308cea2e5021'
PREREG='0c94b0dc84bd17f0542df28e9ae56d7486c8944fe3cd08db74e6c91a76c24443'
SCI=cf.SCI;REL=cf.REL;DESIGN=cf.DESIGN;HOOK=cf.HOOK;PLAN=cf.PLAN;GUARD=cf.GUARD;N=1024
arr=cf.arr
def write(p,x):
 with Path(p).open('x') as out:json.dump(x,out,indent=2,allow_nan=False);out.write('\n')
def rec(p):
 return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),effective_sha256=sha(p))
def definitions(text):
 return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def load_original_records(d,cp,P):
 records=[];provenance=[];train=d['train']
 for gi,group in enumerate(cp['groups'],1):
  part='part%02d'%gi;folder=CAP/part;gp=CAP/(part+'_GUARD.json');guard=read(gp)
  assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors'] and guard['guard_sha256']==GUARD
  assert guard['max_owned_swap_bytes']==0
  manifest=read(folder/'manifest.json')
  assert manifest['sha256']==SCI and manifest['diagnostic_only'] and manifest['environment']['thread_provenance']['blas_threads']==4
  capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']==HOOK and meta['target_free']
  capturedP=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],capturedP[k]) for k in P)
  completion=read(capdir/'CAPTURE_COMPLETE_v1.json');assert completion['complete'] and completion['target_free']
  for seed in group:
   assert seed in train and seed not in d['protected_seed_ids']
   raw=folder/('diagnostic_pred_%d.bin'%seed);assert raw.stat().st_size==65536
   pred=np.fromfile(raw,dtype='<f4').reshape(16,N).astype('f8');assert np.isfinite(pred).all()
   wr=cp['rows'][str(seed)]['weights'];wp=ROOT/wr['path']
   assert wp.stat().st_size==wr['bytes'] and sha(wp)==wr['sha256']
   W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,N,N)
   tr=d['rows'][str(seed)];tp=ROOT/tr['truth']['path']
   assert tp.stat().st_size==tr['truth']['bytes'] and sha(tp)==tr['truth']['sha256']
   assert tr['raw_float32_sha256']==wr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
   with np.load(tp,allow_pickle=False) as z:
    assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256']
    A=z['mc_post_A'].astype('f8')-pred;Bv=z['mc_post_B'].astype('f8')-pred
   assert A.shape==Bv.shape==(16,N) and np.isfinite(A).all() and np.isfinite(Bv).all()
   rr=dict(seed=seed,G=[],b=[],u0_sum=[],physical_gram=[],head_rms2=[]);caps=[]
   for l in range(1,16):
    matches=[x for x in completion['files'] if x['rank']==seed and x['phase']=='scored' and x['layer']==l];assert len(matches)==1
    item=matches[0];path=capdir/item['path'];assert sha(path)==item['sha256']
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
    X=np.column_stack([state,np.ones(N),hidden])*mask[:,None]
    inherited=np.asarray(W[l],dtype='f8').T@np.stack([A[l-1],Bv[l-1]],axis=1)
    yA=(A[l]-gate*inherited[:,0])/physical;yB=(Bv[l]-gate*inherited[:,1])/physical
    rr['G'].append(X.T@X);rr['b'].append(X.T@(.5*(yA+yB)));rr['u0_sum'].append(float(yA@yB))
    direct=physical[:,None]*X;rr['physical_gram'].append(direct.T@direct/N);rr['head_rms2'].append(float(np.mean((saved*mask)**2)))
    caps.append(dict(layer=l,capture_sha256=sha(path),head_replay_max_abs=replay,state_replay_max_abs=state_replay,clipped=int(np.sum(mask==0))))
   for k in ('G','b','u0_sum','physical_gram','head_rms2'):rr[k]=np.asarray(rr[k])
   assert float(np.sum(rr['u0_sum']))>0
   records.append(rr);provenance.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp),prediction_sha256=sha(raw),guard_sha256=sha(gp),captures=caps))
   print(json.dumps(dict(utc=utc(),loaded=seed,completed=len(records))),flush=True)
 assert [r['seed'] for r in records]==train
 return records,provenance
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 assert sha(cf.__file__)==CF and sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 assert sha(HERE/'PREREG_v1.json')==PREREG
 checks=cf.selftest();prereg=read(HERE/'PREREG_v1.json')
 assert sha(D/'DESIGN_LOCAL_v1.json')==DESIGN;d=read(D/'DESIGN_LOCAL_v1.json');train=d['train']
 assert train==prereg['train'] and len(train)==90 and d['weight_hash_disjointness'] and not set(train)&set(d['protected_seed_ids'])
 for j,fold in enumerate(prereg['folds']):
  assert fold['key']=='fold%d'%j and fold['validation']==train[j::5]
  assert fold['train']==[x for x in train if x not in fold['validation']] and len(fold['train'])==72
  assert not set(fold['train'])&set(fold['validation'])
 if args.self_test:
  print(json.dumps(dict(successful=True,selftests=checks,explicit_fold_exclusion=True)));return
 assert sys.platform=='linux'
 ready=read(D/'CAPTURE_READY_v1.json');assert ready['successful'] and ready['design_sha256']==DESIGN
 assert ready['hook_sha256']==HOOK and ready['plan_sha256']==PLAN and sha(ready['plan'])==PLAN and sha(ready['recipe'])==ready['recipe_sha256']
 cp=read(ready['plan']);assert [x for g in cp['groups'] for x in g]==train
 done=read(CAP/'COMPLETE.json');assert done['successful'] and done['plan_sha256']==PLAN
 rm=read(CAP/'RUN_MANIFEST.json');assert rm['candidate_sha256']==SCI and rm['blas_threads']==4
 original=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(original)==SCI
 pp,_=f.literals_and_functions(original.read_text());P=f.decode(pp['_GNN_LITERAL']);U=cf.nullspace(P['enc4_w'])
 parents={'science':KO/'candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py','release':KO/'candidate_rel_astra_o21_edge8_hidden_complement90_v1.py'}
 for role,path in parents.items():assert sha(path)==prereg['parent_'+role+'_sha256']
 bp,_=f.literals_and_functions(parents['science'].read_text());parent=f.decode(bp['_GNN_LITERAL'])
 assert set(parent)==set(P)|{'hidden_head'}
 assert all(np.array_equal(P[k],parent[k]) for k in P if k not in ('head','head_b'))
 oldfitpath=RUNS/'hidden_complement90_fit_v1/FITTED_v1.json';assert sha(oldfitpath)==prereg['first_stage']['full90_c_receipt_sha256']
 oldfit=read(oldfitpath);assert oldfit['passed'] and oldfit['source_sha256']==SCI
 records,provenance=load_original_records(d,cp,P)
 assert len(oldfit['provenance'])==len(provenance)==90
 for before,now in zip(oldfit['provenance'],provenance):assert before==now
 out=HERE/'parents_v1';out.mkdir(exist_ok=False);families=[]
 for definition in prereg['folds']+[dict(key='full90',train=train,validation=[])]:
  key=definition['key'];selected=[r for r in records if r['seed'] in definition['train']]
  assert [r['seed'] for r in selected]==definition['train'] and not set(definition['train'])&set(definition['validation'])
  # Deliberately sum selected TRAIN records only: never full-minus-held aggregation.
  G=sum(r['G'] for r in selected);b=sum(r['b'] for r in selected)
  _,coef=cf.coefficients(G,b,len(selected),P,U)
  Q={k:v.copy() for k,v in parent.items()}
  Q['head'][1:]=(P['head'][1:].astype('f8')+coef[:,:64]).astype('f4')
  Q['head_b'][1:]=(P['head_b'][1:].astype('f8')+coef[:,64]).astype('f4')
  Q['hidden_head'][1:]=coef[:,65:].astype('f4')
  if key=='full90':
   np.testing.assert_array_equal(coef,np.asarray(oldfit['coef_by_layer']))
   assert all(np.array_equal(Q[k],parent[k]) for k in Q)
  folder=out/key;folder.mkdir();coefpath=folder/'COEFFICIENTS_v1.json'
  write(coefpath,dict(utc=utc(),key=key,train=definition['train'],validation=definition['validation'],coef_by_layer=coef.tolist(),prereg_sha256=PREREG,original_source_sha256=SCI,original_fit_sha256=sha(oldfitpath),cf_sha256=CF,explicit_train_sum_only=True))
  outputs={};suffix='\n# Frozen independent TRAIN72 first-stage diagnostic parent; tensor-only.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\n_GNN_LITERAL = '+repr(f.encode(Q))+'\n_FAST_RESCUE_FRAC = {}\n'
  for role,path in parents.items():
   if key=='full90':target=path
   else:
    text=path.read_text();source=text+suffix;target=folder/('candidate_'+key+'_'+role+'_v1.py');compile(source,str(target),'exec')
    assert definitions(source)==definitions(text)
    with target.open('x') as stream:stream.write(source)
   lit,_=f.literals_and_functions(target.read_text());actual=f.decode(lit['_GNN_LITERAL'])
   assert all(np.array_equal(Q[k],actual[k]) for k in Q)
   outputs[role]=rec(target)
  write(folder/'BUILD_v1.json',dict(utc=utc(),key=key,outputs={r:dict(path=str(ROOT/x['path']),sha256=x['sha256'],parent_path=str(parents[r]),parent_sha256=sha(parents[r])) for r,x in outputs.items()},coefficient_sha256=sha(coefpath),all_live_functions_unchanged=True,only_existing_head_tensors_changed=True,no_new_features=True,no_native_launch=True))
  families.append(dict(key=key,train=definition['train'],validation=definition['validation'],science=outputs['science'],release=outputs['release'],coef_path=str(coefpath.relative_to(ROOT)),coef_sha256=sha(coefpath),build_path=str((folder/'BUILD_v1.json').relative_to(ROOT)),build_sha256=sha(folder/'BUILD_v1.json'),parent_fit_target_seed_ids=definition['train'],parent_fit_excludes_validation=True,full90_exact_b54=(key=='full90')))
 result=dict(utc=utc(),successful=True,prereg_sha256=PREREG,design_sha256=DESIGN,script_sha256=sha(__file__),original_fit_sha256=sha(oldfitpath),cf_sha256=CF,families=families,fold_disjointness_proof={'per_fold_parent_target_ids_equal_explicit_train72':True,'no_validation_in_parent_targets':True,'each_validation_seed_occurs_once':True,'correction_fit_must_repeat_identical_exclusion':True},full90_exact_b54_tensor_parity=True,original_provenance=provenance,selftests=checks,no_native_launch=True,no_adjusted_gain_claim=True)
 write(HERE/'PARENT_SOURCES_v1.json',result)
 print(json.dumps(dict(utc=utc(),successful=True,parent_sources_sha256=sha(HERE/'PARENT_SOURCES_v1.json'),families=6,no_native_launch=True)),flush=True)
if __name__=='__main__':main()
