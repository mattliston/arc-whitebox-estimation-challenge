"""One frozen four-coefficient fit to actual native TRAIN30 differences.

Whole-network LOO, fold-local scaling and fixed ridge/damping. No held truth,
candidate export, validation grid, amplitude search or adjusted-gain claim.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'truth_response90_direction4_native_fit_v1'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
CONTROLLER='315bc5227127b2fa7102fa937cb008edb7040c228527474826cd50a8d1f96aa2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def normal(r):
 return r['J'].T@r['J']/r['u0'],r['J'].T@(.5*(r['A']+r['B']))/r['u0']
def solve(G,b,n):
 scale=np.maximum(np.sqrt(np.diag(G)/n),1e-14)
 coef=.5*np.linalg.solve(G/(scale[:,None]*scale[None,:])+.1*n*np.eye(4),b/scale)/scale
 coef/=max(1.,float(np.max(np.abs(coef))));assert np.isfinite(coef).all() and np.max(np.abs(coef))<=1.+1e-14
 return coef
def metric(r,c):
 change=r['J']@c;u1=float(np.mean((r['A']-change)*(r['B']-change)));assert u1>0
 return dict(seed=r['seed'],u0=r['u0'],u1=u1,relative_pct=100*(u1/r['u0']-1),correction_rms=float(np.sqrt(np.mean(change**2))))
def summary(rows):
 values=np.array([r['relative_pct'] for r in rows]);return dict(n=len(rows),mean_relative_pct=float(values.mean()),
  geometric_pct=float(100*np.expm1(np.log1p(values/100).mean())),pooled_pct=float(100*(sum(r['u1'] for r in rows)/sum(r['u0'] for r in rows)-1)),
  lower=int(np.sum(values<0)),se_relative_pct=float(values.std(ddof=1)/np.sqrt(len(values))))
def selftest():
 rng=np.random.default_rng(202610210304);records=[]
 for i in range(7):
  J=rng.normal(size=(53,4));c=rng.normal(size=4)*.03
  records.append(dict(J=J,A=J@c+rng.normal(size=53)*.2,B=J@c+rng.normal(size=53)*.2,u0=.1+i*.02))
 def stacked(rr):
  X=np.concatenate([r['J']/np.sqrt(r['u0']) for r in rr]);y=np.concatenate([.5*(r['A']+r['B'])/np.sqrt(r['u0']) for r in rr]);s=np.sqrt(np.mean(X*X,axis=0))
  z=X/s;coef=.5*np.linalg.solve(z.T@z+.1*len(X)*np.eye(4),z.T@y)/s
  return coef/max(1.,float(np.max(np.abs(coef))))
 G=sum(normal(r)[0] for r in records);b=sum(normal(r)[1] for r in records)
 np.testing.assert_allclose(solve(G,b,53*7),stacked(records),atol=1e-13,rtol=1e-13)
 for i,r in enumerate(records):
  gi,bi=normal(r);np.testing.assert_allclose(solve(G-gi,b-bi,53*6),stacked(records[:i]+records[i+1:]),atol=1e-13,rtol=1e-13)
 base=rng.normal(size=53);linear=rng.normal(size=53);quadratic=rng.normal(size=53)*.01
 plus=base+linear+quadratic;minus=base-linear+quadratic
 np.testing.assert_allclose((plus-minus)/2,linear,atol=1e-14);np.testing.assert_allclose((plus+minus-2*base)/2,quadratic,atol=1e-14)
 a=rng.normal(size=53);b=rng.normal(size=53);change=rng.normal(size=53)
 np.testing.assert_allclose(np.mean((a-change)*(b-change))-np.mean(a*b),np.mean(change*change-change*(a+b)),atol=1e-14)
 return dict(weighted_normal_equations_equal_stacked_fit=True,whole_network_LOO_scaling_equal_stacked_fit=True,
  quadratic_central_difference_and_curvature_exact=True,crosshalf_objective_expansion_exact=True)
def native_prediction(folder,gi,seed,source,plan_sha):
 part='part%02d'%gi;guard=read(folder/(part+'_GUARD.json'))
 assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors'] and guard['guard_sha256']==GUARD
 d=folder/part;m=read(d/'manifest.json');assert m['sha256']==source and m['ranks']==[seed] and m['plan_sha256']==plan_sha
 assert m['environment']['thread_provenance']['blas_threads']==4 and not m['warmup_each']
 rows=[json.loads(line) for line in (d/'rows.jsonl').read_text().splitlines()];assert len(rows)==1
 row=rows[0];assert row['rank']==seed and row['phase']=='scored' and row['nonfinite']==0 and row['flops']>0
 p=d/f'pred_{seed}.bin';assert p.stat().st_size==65536 and sha(p)==row['prediction_sha256']
 pred=np.fromfile(p,dtype='<f4').reshape(16,1024).astype('f8');assert np.isfinite(pred).all()
 return pred,dict(prediction_sha256=sha(p),guard_sha256=sha(folder/(part+'_GUARD.json')),manifest_sha256=sha(d/'manifest.json'),flops=row['flops'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args();checks=selftest()
 if args.self_test:print(json.dumps(checks));return
 assert sys.platform=='linux'
 dp=HERE/'DESIGN_LOCAL_v1.json';pr=HERE/'DIRECTION4_PREREG_v1.json';tp=HERE/'DIRECTION4_NATIVE_TOPOLOGY_v1.json'
 assert sha(dp)=='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
 assert sha(pr)=='03bd92249d5cbe2915b5f8d102c7f66105efd041bfa7dd235ad0372ac3e539bb'
 assert sha(tp)=='7155d6016bd7629a6fe589519697eb702bfe47d9b47068d4bdbeb4c54f003ce0'
 design=read(dp);prereg=read(pr);seeds=prereg['fd_train_seeds'];assert seeds==design['train'][::3] and len(seeds)==30
 assert not set(seeds)&set(design['protected_seed_ids']) and design['weight_hash_disjointness']
 final=read(HERE/'DIRECTION4_NATIVE_PIPELINE_FINAL_v1.json');assert final['successful'] and not final['truth_arrays_opened']
 lp=HERE/'DIRECTION4_ENDPOINT_LAUNCHES_v1.json';launch=read(lp);assert sha(lp)==final['launches_sha256']
 anchorp=HERE/'DIRECTION4_ANCHORS_BYTE_EQUAL_v1.json';anchors=read(anchorp);assert anchors['successful'] and anchors['source_sha256']==SCI and sha(anchorp)==launch['anchors_sha256']
 ready=read(HERE/'direction4_endpoints_v1/FD_READY_v1.json');plan=read(ready['plan']);assert sha(ready['plan'])==ready['plan_sha256']
 assert plan['groups']==[[s] for s in seeds] and plan['training_only'] and plan['no_submission']
 geometryp=RUNS/'truth_response90_direction4_geometry_v1/DIRECTIONS_v1.json';geometry=read(geometryp)
 assert sha(geometryp)==ready['geometry_sha256'] and geometry['no_truth_arrays_or_fit_outputs_read'] and len(geometry['directions'])==4
 cp=read(HERE/'PLAN_CAPTURE_v1.json');capdone=read(CAP/'COMPLETE.json');assert capdone['successful']
 for row in launch['launches']:
  folder=Path(row['out']);complete=read(folder/'COMPLETE.json');manifest=read(folder/'RUN_MANIFEST.json')
  assert complete['successful'] and complete['plan_sha256']==ready['plan_sha256'] and not complete['pending'] and set(complete['results'].values())=={0}
  assert manifest['candidate_sha256']==row['source_sha256'] and manifest['blas_threads']==4 and manifest['max_active']==36
  assert manifest['scheduler_sha256']==CONTROLLER and not manifest['warmup_each']
 endpoints={(r['direction'],r['sign']):r for r in launch['launches']};assert set(endpoints)=={(i,s) for i in range(4) for s in (-1,1)}
 OUT.mkdir(exist_ok=False);records=[];prov=[];curvature=[];jitter=[]
 for gi,seed in enumerate(seeds,1):
  capgi=next(i for i,g in enumerate(cp['groups'],1) if seed in g);assert cp['groups'][capgi-1][0]==seed
  pp=CAP/('part%02d'%capgi)/f'diagnostic_pred_{seed}.bin';assert pp.stat().st_size==65536
  base=np.fromfile(pp,dtype='<f4').reshape(16,1024).astype('f8');J=[];Q=[];endpoint_prov=[]
  for axis in range(4):
   predictions={}
   for sign in (-1,1):
    endpoint=endpoints[axis,sign];pred,p=native_prediction(Path(endpoint['out']),gi,seed,endpoint['source_sha256'],ready['plan_sha256'])
    np.testing.assert_array_equal(pred[0],base[0]);predictions[sign]=pred[15];endpoint_prov.append(dict(axis=axis,sign=sign,**p))
   linear=(predictions[1]-predictions[-1])/2;curve=(predictions[1]+predictions[-1]-2*base[15])/2
   norm=float(np.linalg.norm(linear));assert norm>0
   ratio=float(np.linalg.norm(curve)/norm);spacing=np.spacing(base[15].astype('f4')).astype('f8')
   curvature.append(dict(seed=seed,axis=axis,curvature_over_central_L2=ratio,central_rms=float(np.sqrt(np.mean(linear**2))),curvature_rms=float(np.sqrt(np.mean(curve**2))),central_over_float32_spacing_L2=float(norm/max(np.linalg.norm(spacing),1e-30))))
   J.append(linear);Q.append(curve)
  Q=np.column_stack(Q);norms=np.linalg.norm(Q,axis=0);cos=(Q.T@Q)/np.maximum(norms[:,None]*norms[None,:],1e-100);values=np.linalg.svd(Q,compute_uv=False)
  jitter.append(dict(seed=seed,curvature_pairwise_cosines=[float(cos[i,j]) for i in range(4) for j in range(i+1,4)],curvature_rank1_energy_fraction=float(values[0]**2/max(float(values@values),1e-100))))
  records.append(dict(seed=seed,J=np.column_stack(J),base=base[15]))
  prov.append(dict(seed=seed,capture_prediction_sha256=sha(pp),endpoints=endpoint_prov))
 per_axis=[]
 for axis in range(4):
  ratios=np.array([r['curvature_over_central_L2'] for r in curvature if r['axis']==axis]);median=float(np.median(ratios));low=int(np.sum(ratios<=.5))
  per_axis.append(dict(axis=axis,median_curvature_over_central_L2=median,count_at_most_point5=low,passed=bool(median<=.25 and low>=24)))
 base_result=dict(utc=utc(),source_sha256=SCI,design_sha256=sha(dp),prereg_sha256=sha(pr),topology_sha256=sha(tp),script_sha256=sha(__file__),selftests=checks,
  source_anchors_sha256=sha(anchorp),geometry_sha256=sha(geometryp),endpoint_launches_sha256=sha(lp),curvature=per_axis,curvature_per_network=curvature,
  descriptive_jitter=jitter,provenance=prov,no_adjusted_gain_claim=True,no_candidate_exported=True,no_held_or_confirmation_truth=True)
 if not all(r['passed'] for r in per_axis):
  result=dict(base_result,status='REJECT_FIXED_STEP_CURVATURE_BEFORE_TARGETS',passed=False,truth_arrays_opened=False)
  write(OUT/'FITTED_v1.json',result);print(json.dumps(dict(status=result['status'],curvature=per_axis)),flush=True);return
 for r,p in zip(records,prov):
  seed=r['seed'];tr=design['rows'][str(seed)];truth=ROOT/tr['truth']['path'];wr=plan['rows'][str(seed)]['weights']
  assert sha(truth)==tr['truth']['sha256'] and truth.stat().st_size==tr['truth']['bytes']
  assert wr['raw_float32_sha256']==tr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in design['protected_raw_weight_hashes']
  with np.load(truth,allow_pickle=False) as z:
   assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256']
   r['A']=z['mc_post_A'][15].astype('f8')-r['base'];r['B']=z['mc_post_B'][15].astype('f8')-r['base']
  r['u0']=float(np.mean(r['A']*r['B']));assert r['u0']>0
  r['G'],r['b']=normal(r);p['truth_sha256']=sha(truth)
 G=sum(r['G'] for r in records);b=sum(r['b'] for r in records);coef=solve(G,b,1024*30)
 train_rows=[metric(r,coef) for r in records];loo_rows=[];loo_coef=[]
 for r in records:
  c=solve(G-r['G'],b-r['b'],1024*29);loo_rows.append(metric(r,c));loo_coef.append(dict(seed=r['seed'],coef=c.tolist()))
 train=summary(train_rows);loo=summary(loo_rows);blocks=[summary([r for r in loo_rows if r['seed'] in seeds[i::5]]) for i in range(5)]
 passed=bool(loo['mean_relative_pct']<=-.25 and loo['lower']>=18 and sum(r['mean_relative_pct']<0 for r in blocks)>=4)
 directions=np.asarray([r['head_delta'] for r in geometry['directions']],dtype='f8');assert directions.shape==(4,64)
 result=dict(base_result,utc=utc(),status='NATIVE_DIRECTION_CV_PASS_ACTUAL_TRAIN_VERIFY_REQUIRED' if passed else 'REJECT_NETWORK_LOO_BEFORE_NATIVE_EXPORT',passed=passed,
  truth_arrays_opened=True,train=train,loo=loo,train_per_network=train_rows,loo_per_network=loo_rows,descriptive_blocks=blocks,
  coef=coef.tolist(),loo_coef=loo_coef,head_delta=(coef@directions).tolist(),ridge=.1,damping=.5,max_abs_coefficient=float(np.max(np.abs(coef))))
 write(OUT/'FITTED_v1.json',result);print(json.dumps(dict(status=result['status'],train=train,loo=loo,blocks=blocks,coef=coef.tolist())),flush=True)
if __name__=='__main__':main()
