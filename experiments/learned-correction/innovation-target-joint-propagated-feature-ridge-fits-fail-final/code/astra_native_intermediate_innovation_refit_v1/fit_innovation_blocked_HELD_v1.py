"""One fixed innovation-target fit and linearized held-network rollout forecast."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,resource,time
import numpy as np
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-authorized',action='store_true');a=ap.parse_args();assert a.run_authorized
 assert all(os.environ.get(k)=='1'for k in THREADS)and os.getpriority(os.PRIO_PROCESS,0)==10
 rulep=D/'FIT_RULE_FROZEN_v1.json';rule=json.loads(rulep.read_text());assert rule['fit_script_sha256']==sha(__file__)
 assert rule['ridge']==10. and rule['shrink']==.5 and rule['layers']==list(range(7,15)) and rule['folds']==[list(range(4*i,4*i+4))for i in range(8)]
 for p,h in rule['input_pins'].items():assert sha(p)==h,(p,'hash')
 local=json.loads(Path(rule['local_retirement']).read_text());terminal=json.loads(Path(rule['terminal_retirement']).read_text())
 for ret in (local,terminal):assert ret['approved']and ret['reviewed_by']=='root'and ret['complete_calls']==32 and ret['all_owned_PID_PPID_PGID_absent']and ret['all_guards_clean']
 ls={x['seed']:x for x in local['evaluations']};ts={x['seed']:x for x in terminal['evaluations']};assert sorted(ls)==sorted(ts)==list(range(32))
 meta=json.loads(Path(rule['capture_protocol']).read_text())['network_metadata'];meta={x['rank']:x for x in meta};assert sorted(meta)==list(range(32))
 fitp=Path(rule['unchanged_ridge_source']);sp=importlib.util.spec_from_file_location('_unchanged_local_fit',fitp);fitmod=importlib.util.module_from_spec(sp);sp.loader.exec_module(fitmod)
 start=time.monotonic();records={l:[]for l in range(7,15)};nets={};pins={}
 for rank in range(32):
  cp=Path(ls[rank]['capture_path']);tp=Path(ts[rank]['capture_path']);assert sha(cp)==ls[rank]['capture_sha256'];assert sha(tp)==ts[rank]['capture_sha256']
  wp=Path(meta[rank]['weights_path']);assert sha(wp)==meta[rank]['weights_sha256'];W=np.load(wp,mmap_mode='r',allow_pickle=False);assert W.shape==(16,1024,1024)
  pins.update({str(cp):sha(cp),str(tp):sha(tp),str(wp):meta[rank]['weights_sha256']})
  with np.load(cp,allow_pickle=False)as z,np.load(tp,allow_pickle=False)as t:
   pred=z['predictions_float32'];assert pred.tobytes()==t['predictions_float32'].tobytes()
   truth=z['target_means_float64'];raw=z['raw_means_float64'];assert raw.shape==truth.shape==(8,1024)
   gates={};prev_error=None
   for i,l in enumerate(range(7,15)):
    X=np.column_stack([z[f'L{l}_features30'],z[f'L{l}_state64']]).astype(float);sigma=z[f'L{l}_sigma'].astype(float)
    gate=X[:,5].copy();assert np.all((gate>=0)&(gate<=1))and np.all(sigma>0)
    error=truth[i]-raw[i];innovation=error.copy()if l==7 else error-gate*(W[l].astype(float).T@prev_error)
    assert np.maximum(raw[i],0).astype(np.float32).tobytes()==pred[l].tobytes()
    bm=float(np.mean((pred[l].astype(float)-truth[i])**2));assert bm>0
    records[l].append(dict(rank=rank,layer=l,x=X,sigma=sigma,raw=raw[i].copy(),target=raw[i]+innovation,truth=truth[i].copy(),baseline_mse=bm,gate=gate,innovation=innovation))
    gates[l]=gate;prev_error=error
   terminal_gate=t['features30'][:,5].astype(float);assert np.all((terminal_gate>=0)&(terminal_gate<=1))
   terminal_raw=t['raw_final_float64'].copy();terminal_truth=t['target_final_float64'].copy()
   assert np.maximum(terminal_raw,0).astype(np.float32).tobytes()==pred[15].tobytes()
   bm=float(np.mean((pred[15].astype(float)-terminal_truth)**2));assert bm>0
   nets[rank]=dict(weights_path=str(wp),gate15=terminal_gate,raw15=terminal_raw,truth15=terminal_truth,baseline_mse=bm)
  del W
 print(json.dumps(dict(stage='inputs_and_fixed_innovations_ready',seconds=time.monotonic()-start)),flush=True)
 models={};rows=[];localrows=[];folds=[]
 for fi,fold in enumerate(rule['folds']):
  fitmaps={l:fitmod.fit([r for r in records[l]if r['rank']not in fold])for l in range(7,15)}
  for l,m in fitmaps.items():models[f'fold{fi}_L{l}_coefficient94']=m['coef'];models[f'fold{fi}_L{l}_bias']=np.asarray(m['bias'])
  for rank in fold:
   net=nets[rank];W=np.load(net['weights_path'],mmap_mode='r',allow_pickle=False);shift=None
   for l in range(7,15):
    r=records[l][rank];assert r['rank']==rank;m=fitmaps[l]
    normalized=(r['x'][:,:30]@m['coef'][:30]+r['x'][:,30:]@m['coef'][30:])+m['bias']
    predicted_innovation=r['sigma']*normalized
    shift=.5*predicted_innovation if l==7 else r['gate']*(W[l].astype(float).T@shift)+.5*predicted_innovation
    pred=np.maximum(r['raw']+shift,0).astype(np.float32);mse=float(np.mean((pred.astype(float)-r['truth'])**2))
    localrows.append(dict(rank=rank,layer=l,log_mse_ratio=math.log(mse/r['baseline_mse'])))
   finalshift=net['gate15']*(W[15].astype(float).T@shift)
   predicted=np.maximum(net['raw15']+finalshift,0).astype(np.float32)
   mse=float(np.mean((predicted.astype(float)-net['truth15'])**2));assert mse>0 and np.isfinite(predicted).all()
   rows.append(dict(rank=rank,fold=fi,baseline_mse=net['baseline_mse'],mse=mse,log_mse_ratio=math.log(mse/net['baseline_mse']),predicted_final_shift_rms=float(np.sqrt(np.mean(finalshift**2)))))
   del W
  fm=float(np.mean([r['log_mse_ratio']for r in rows if r['fold']==fi]));folds.append(dict(fold=fi,heldout=fold,mean_log_mse_ratio=fm,geometric_final_MSE_change_pct=100*math.expm1(fm)))
  print(json.dumps(dict(stage='fold_complete',fold=fi,seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<175
 logs=np.asarray([r['log_mse_ratio']for r in rows]);mean=float(logs.mean());negative=sum(r['mean_log_mse_ratio']<0 for r in folds)
 eligible=mean<=-.003 and negative>=6
 output=D/'FOLD_INNOVATION_RIDGE_MODELS_v1.npz';assert not output.exists();np.savez(output,**models)
 result=dict(status='LINEARIZED_FORECAST_REQUIRES_ROOT_NATIVE_REVIEW'if eligible else'FIXED_INNOVATION_FORECAST_CLOSED',eligible_for_native_review=eligible,mean_final_log_MSE_ratio=mean,geometric_final_MSE_change_pct=100*math.expm1(mean),SE_log_pct=100*float(logs.std(ddof=1))/np.sqrt(32),networks_lower=int((logs<0).sum()),folds_negative=negative,folds=folds,per_network=rows,per_layer_linearized=[dict(layer=l,geometric_MSE_change_pct=100*math.expm1(float(np.mean([r['log_mse_ratio']for r in localrows if r['layer']==l]))))for l in range(7,15)],old_total_target_local_CV_pct=-.39502228101888526,forecast_only=True,no_bank_claim=True,no_native_models=True,no_final_all32_fit=True,zero_baseline_selected=mean>=0,rule_sha256=sha(rulep),fit_script_sha256=sha(__file__),data_pins=pins,fold_models_sha256=sha(output),seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,approximation='Fixed baseline analytic gate Jacobian only; omits learned-head and covariance/cumulant response. Intermediate propagation is the frozen linear recurrence; scoring applies nonnegative clamp/F32. No holdout target enters predicted shifts.')
 out=D/'INNOVATION_BLOCK_CV_RESULT_v1.json';out.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['status','eligible_for_native_review','mean_final_log_MSE_ratio','geometric_final_MSE_change_pct','SE_log_pct','networks_lower','folds_negative','folds','per_layer_linearized','seconds','max_rss_bytes']}),flush=True)
if __name__=='__main__':main()
