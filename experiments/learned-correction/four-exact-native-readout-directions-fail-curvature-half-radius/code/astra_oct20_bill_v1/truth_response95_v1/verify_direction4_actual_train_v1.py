"""Measure the single frozen fit on its native TRAIN30 check; never refit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
sys.path.insert(0,str(HERE));import fit_direction4_native_v1 as f
assert f.sha(f.__file__)=='a9bc4b22fcc3359dd474d8237fb829503a4d0b280b4fe7e166ea09ef1efe1efe'
sha=f.sha;read=f.read;write=f.write;utc=f.utc
fitp=RUNS/'truth_response90_direction4_native_fit_v1/FITTED_v1.json';fit=read(fitp);assert fit['passed']
qualified=read(HERE/'native_direction4_fit_v1/QUALIFIED_v1.json');assert len(qualified['qualified'])==1
ready=qualified['qualified'][0];assert ready['fit_sha256']==sha(fitp);plan=read(ready['plan']);assert sha(ready['plan'])==ready['plan_sha256']
out=RUNS/'truth_response90_direction4_fitted_train30_v1';done=read(out/'COMPLETE.json');manifest=read(out/'RUN_MANIFEST.json')
assert done['successful'] and done['plan_sha256']==ready['plan_sha256'] and not done['pending'] and set(done['results'].values())=={0}
assert manifest['candidate_sha256']==ready['source_sha256'] and manifest['blas_threads']==4 and manifest['max_active']==36 and manifest['scheduler_sha256']==f.CONTROLLER
design=read(HERE/'DESIGN_LOCAL_v1.json');assert sha(HERE/'DESIGN_LOCAL_v1.json')=='3ade08b96c06461c7cc9d455957b87a8c858d6500ef1227a03e6eff3a769ed44'
seeds=read(HERE/'DIRECTION4_PREREG_v1.json')['fd_train_seeds'];assert plan['groups']==[[s] for s in seeds] and not set(seeds)&set(design['protected_seed_ids'])
cp=read(HERE/'PLAN_CAPTURE_v1.json');launches=read(HERE/'DIRECTION4_ENDPOINT_LAUNCHES_v1.json')['launches'];endpoints={(r['direction'],r['sign']):r for r in launches}
anchors={r['seed']:r for r in read(HERE/'DIRECTION4_ANCHORS_BYTE_EQUAL_v1.json')['anchors']};coef=np.asarray(fit['coef']);rows=[];alignment=[];provenance=[];billing=[]
for gi,seed in enumerate(seeds,1):
 pred,pr=f.native_prediction(out,gi,seed,ready['source_sha256'],ready['plan_sha256'])
 capgi=next(i for i,g in enumerate(cp['groups'],1) if seed in g);assert cp['groups'][capgi-1][0]==seed
 p=f.CAP/('part%02d'%capgi)/f'diagnostic_pred_{seed}.bin';base=np.fromfile(p,dtype='<f4').reshape(16,1024).astype('f8');np.testing.assert_array_equal(pred[0],base[0])
 tp=ROOT/design['rows'][str(seed)]['truth']['path'];assert sha(tp)==design['rows'][str(seed)]['truth']['sha256']
 with np.load(tp,allow_pickle=False) as z:
  assert int(z['seed'])==seed and str(z['weights_sha256'])==design['rows'][str(seed)]['raw_float32_sha256']
  A=z['mc_post_A'][15].astype('f8')-base[15];B=z['mc_post_B'][15].astype('f8')-base[15]
 u0=float(np.mean(A*B));actual=pred[15]-base[15];u1=float(np.mean((A-actual)*(B-actual)));assert min(u0,u1)>0
 rows.append(dict(seed=seed,u0=u0,u1=u1,relative_pct=100*(u1/u0-1),flops=pr['flops']))
 predicted=np.zeros(1024)
 for axis in range(4):
  minus=np.fromfile(Path(endpoints[axis,-1]['out'])/('part%02d'%gi)/f'pred_{seed}.bin',dtype='<f4').reshape(16,1024)[15].astype('f8')
  plus=np.fromfile(Path(endpoints[axis,1]['out'])/('part%02d'%gi)/f'pred_{seed}.bin',dtype='<f4').reshape(16,1024)[15].astype('f8')
  predicted+=coef[axis]*(plus-minus)/2
 err=np.linalg.norm(actual-predicted);den=np.linalg.norm(actual);cos=float(actual@predicted/max(den*np.linalg.norm(predicted),1e-100))
 alignment.append(dict(seed=seed,relative_L2=float(err/max(den,1e-100)),cosine=cos,actual_norm=float(den),error_norm=float(err)))
 if seed in anchors:
  billing.append(dict(seed=seed,baseline_flops=anchors[seed]['flops'],candidate_flops=pr['flops'],bill_pct=100*(pr['flops']/anchors[seed]['flops']-1)))
 provenance.append(dict(seed=seed,truth_sha256=sha(tp),base_prediction_sha256=sha(p),**pr))
stats=f.summary(rows);blocks=[f.summary([r for r in rows if r['seed'] in seeds[i::5]]) for i in range(5)]
# This screen authorizes only a separate original32 evaluation; it is not bank.
supported=bool(stats['mean_relative_pct']<0 and stats['lower']>=18 and sum(b['mean_relative_pct']<0 for b in blocks)>=4 and all(r['bill_pct']==0 for r in billing))
result=dict(utc=utc(),status='ACTUAL_TRAIN30_SUPPORTS_FULL32' if supported else 'CLOSED_ACTUAL_TRAIN30_FAILED',supports_full32=supported,
 fit_sha256=sha(fitp),candidate=ready['candidate'],source_sha256=ready['source_sha256'],script_sha256=sha(__file__),training_MSE=stats,per_network=rows,descriptive_blocks=blocks,
 measured_anchor_billing=billing,nonanchor_baseline_bill_not_measured=True,response_alignment=alignment,
 relative_response_L2=float(np.sqrt(sum(r['error_norm']**2 for r in alignment)/max(sum(r['actual_norm']**2 for r in alignment),1e-100))),
 provenance=provenance,no_refit=True,no_adjusted_gain_claim=True,no_held_or_confirmation_truth=True,no_submission=True)
target=HERE/'native_direction4_fit_v1/ACTUAL_TRAIN30_v1.json';write(target,result)
if supported:
 A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1';basep=A/'cloud_linux_markline_v1/PLAN_MSG8_LOADERFIX_CONTROL_v1.json'
 assert sha(basep)=='9276f1d7d0dfc8b9d5a51c768a98e3eb2e1f2d152f6b038240db0a43b0d075a6'
 full=read(basep);full.update(utc=utc(),kind='FROZEN_NATIVE_DIRECTION4_ORIGINAL32',fit_sha256=sha(fitp),actual_train_receipt_sha256=sha(target),reference_submission=337422)
 full['candidates']={'msg8_loaderfix337422_control':full['candidates']['msg8_loaderfix337422_control'],ready['candidate']:plan['candidates'][ready['candidate']]}
 pp=HERE/'native_direction4_fit_v1/PLAN_FULL32_v1.json';write(pp,full)
 write(HERE/'native_direction4_fit_v1/FULL32_READY_v1.json',dict(utc=utc(),successful=True,candidate=ready['candidate'],source_sha256=ready['source_sha256'],outputs=ready['outputs'],plan=str(pp),plan_sha256=sha(pp),actual_train_sha256=sha(target),requested_jobs=12,shared_max_active=36,blas_threads=4,no_native_full32_launch=True,no_adjusted_gain_claim=True))
print(json.dumps(dict(status=result['status'],training_MSE=stats,billing=billing,response_relative_L2=result['relative_response_L2'])),flush=True)
