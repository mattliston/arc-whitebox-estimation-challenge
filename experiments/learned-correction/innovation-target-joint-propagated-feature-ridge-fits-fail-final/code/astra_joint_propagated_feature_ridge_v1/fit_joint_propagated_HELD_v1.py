"""One frozen final-objective joint propagated-feature ridge, saved TRAIN32 only."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,time
import numpy as np
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-authorized',action='store_true');a=ap.parse_args();assert a.run_authorized
 assert all(os.environ.get(k)=='1'for k in THREADS)and os.getpriority(os.PRIO_PROCESS,0)==10
 rulep=D/'FIT_RULE_FROZEN_v1.json';rule=json.loads(rulep.read_text());assert rule['fit_script_sha256']==sha(__file__)
 assert rule['ridge']==10. and rule['shrink']==.5 and rule['layers']==list(range(7,15))and rule['RMS_floor']==1e-6
 assert rule['folds']==[list(range(4*i,4*i+4))for i in range(8)]
 for p,h in rule['input_pins'].items():assert sha(p)==h,(p,'hash')
 local=json.loads(Path(rule['local_retirement']).read_text());term=json.loads(Path(rule['terminal_retirement']).read_text())
 for ret in (local,term):assert ret['approved']and ret['reviewed_by']=='root'and ret['complete_calls']==32 and ret['all_owned_PID_PPID_PGID_absent']and ret['all_guards_clean']
 ls={x['seed']:x for x in local['evaluations']};ts={x['seed']:x for x in term['evaluations']};assert sorted(ls)==sorted(ts)==list(range(32))
 meta=json.loads(Path(rule['capture_protocol']).read_text())['network_metadata'];meta={x['rank']:x for x in meta}
 start=time.monotonic();nets=[];pins={};models={};folds=[];rows=[]
 def limits():
  assert time.monotonic()-start<115,'Fixed fit time ceiling approached.'
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<7_000_000_000,'Fixed memory ceiling approached.'
 for rank in range(32):
  cp=Path(ls[rank]['capture_path']);tp=Path(ts[rank]['capture_path']);wp=Path(meta[rank]['weights_path'])
  assert sha(cp)==ls[rank]['capture_sha256']and sha(tp)==ts[rank]['capture_sha256']and sha(wp)==meta[rank]['weights_sha256']
  pins.update({str(cp):ls[rank]['capture_sha256'],str(tp):ts[rank]['capture_sha256'],str(wp):meta[rank]['weights_sha256']})
  W=np.load(wp,mmap_mode='r',allow_pickle=False);assert W.shape==(16,1024,1024)
  with np.load(cp,allow_pickle=False)as z,np.load(tp,allow_pickle=False)as t:
   baseline=z['predictions_float32'];assert baseline.tobytes()==t['predictions_float32'].tobytes()
   F=None
   for l in range(7,15):
    X=np.column_stack([z[f'L{l}_features30'],z[f'L{l}_state64']]).astype(float);sigma=z[f'L{l}_sigma'].astype(float);gate=X[:,5]
    assert X.shape==(1024,94)and np.all(sigma>0)and np.all((gate>=0)&(gate<=1))
    if F is not None:F=gate[:,None]*(W[l].astype(float).T@F)
    B=sigma[:,None]*np.column_stack([X,np.ones(1024)])
    F=B if F is None else np.concatenate([F,B],axis=1)
    limits()
   gate15=t['features30'][:,5].astype(float);F=gate15[:,None]*(W[15].astype(float).T@F)
   assert F.shape==(1024,760)and np.isfinite(F).all()
   raw=np.maximum(t['raw_final_float64'].astype(float),0);truth=t['target_final_float64'].astype(float)
   assert raw.astype(np.float32).tobytes()==baseline[15].tobytes()
   error=truth-raw;bm=float(np.mean((baseline[15].astype(float)-truth)**2));assert bm>0
   G=F.T@F/1024.;rhs=F.T@error/1024.;mu=F.mean(0)
   nets.append(dict(rank=rank,F=F,G=G,rhs=rhs,mu=mu,error_mean=float(error.mean()),baseline_mse=bm,raw=raw,truth=truth))
  del W
  if rank in (0,7,15,23,31):print(json.dumps(dict(stage='basis_ready',completed=rank+1,seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
  limits()
 for fi,fold in enumerate(rule['folds']):
  train=[x for x in nets if x['rank']not in fold]
  scale=np.maximum(np.sqrt(np.maximum(np.mean([np.diag(x['G'])for x in train],axis=0),0)),1e-6)
  weights=np.asarray([1/x['baseline_mse']for x in train]);weights/=weights.sum()
  G=np.zeros((761,761));rhs=np.zeros(761)
  for w,x in zip(weights,train):
   G[:760,:760]+=w*x['G'];G[:760,760]+=w*x['mu'];rhs[:760]+=w*x['rhs'];rhs[760]+=w*x['error_mean']
  G[:760,:760]/=scale[:,None]*scale[None,:];G[:760,760]/=scale;G[760,:760]=G[:760,760];G[760,760]=1.
  G[np.arange(760),np.arange(760)]+=10.;rhs[:760]/=scale
  beta=np.linalg.solve(G,rhs);coef=beta[:760]/scale;intercept=float(beta[760]);assert np.isfinite(beta).all()
  models[f'fold{fi}_coefficient760']=coef;models[f'fold{fi}_intercept']=np.asarray(intercept);models[f'fold{fi}_train_RMS760']=scale
  for rank in fold:
   x=nets[rank];shift=.5*(x['F']@coef+intercept);pred=np.maximum(x['raw']+shift,0).astype(np.float32)
   mse=float(np.mean((pred.astype(float)-x['truth'])**2));assert mse>0 and np.isfinite(pred).all()
   rows.append(dict(rank=rank,fold=fi,baseline_mse=x['baseline_mse'],mse=mse,log_mse_ratio=math.log(mse/x['baseline_mse']),predicted_shift_rms=float(np.sqrt(np.mean(shift**2)))))
  lm=float(np.mean([r['log_mse_ratio']for r in rows if r['fold']==fi]));folds.append(dict(fold=fi,heldout=fold,mean_log_mse_ratio=lm,geometric_final_MSE_change_pct=100*math.expm1(lm)))
  print(json.dumps(dict(stage='fold_complete',fold=fi,seconds=time.monotonic()-start)),flush=True);limits()
 logs=np.asarray([r['log_mse_ratio']for r in rows]);mean=float(logs.mean());negative=sum(r['mean_log_mse_ratio']<0 for r in folds);eligible=mean<=-.003 and negative>=6
 mp=D/'FOLD_JOINT_PROPAGATED_MODELS_v1.npz';assert not mp.exists();np.savez(mp,**models)
 result=dict(status='JOINT_FORECAST_REQUIRES_ROOT_NATIVE_REVIEW'if eligible else'FIXED_JOINT_FORECAST_CLOSED',eligible_for_native_review=eligible,mean_final_log_MSE_ratio=mean,geometric_final_MSE_change_pct=100*math.expm1(mean),SE_log_pct=100*float(logs.std(ddof=1))/np.sqrt(32),networks_lower=int((logs<0).sum()),folds_negative=negative,folds=folds,per_network=rows,zero_baseline_selected=mean>=0,forecast_only=True,no_bank_claim=True,no_native_models=True,no_final_all32_fit=True,rule_sha256=sha(rulep),fit_script_sha256=sha(__file__),data_pins=pins,fold_models_sha256=sha(mp),seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,transport_FLOPs=229512314880,Gram_FLOPs=32*2*1024*760*760,feature_columns=760,terminal_intercept=True,ridge=10.,shrink=.5,approximation='Baseline fixed-variance analytic gate Jacobian only; ignores learned-head/covariance/cumulant response and changing feature values. Final clamp/F32 scored; no intermediate clipping of propagated basis. No holdout labels used for fit/scaling/weights.')
 p=D/'JOINT_PROPAGATED_BLOCK_CV_RESULT_v1.json';p.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['status','eligible_for_native_review','mean_final_log_MSE_ratio','geometric_final_MSE_change_pct','SE_log_pct','networks_lower','folds_negative','folds','seconds','max_rss_bytes']}),flush=True)
if __name__=='__main__':main()
