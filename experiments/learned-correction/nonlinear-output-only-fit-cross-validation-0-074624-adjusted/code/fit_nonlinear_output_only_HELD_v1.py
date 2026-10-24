"""Frozen saved-TRAIN32 nonlinear output-only mean-response CV; no native models."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,time
import numpy as np
import torch
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-authorized',action='store_true');a=ap.parse_args();assert a.run_authorized
 assert all(os.environ.get(k)=='1'for k in THREADS)and os.getpriority(os.PRIO_PROCESS,0)==10
 rp=D/'FIT_RULE_FROZEN_v1.json';rule=json.loads(rp.read_text());assert rule['fit_script_sha256']==sha(__file__)
 assert rule['layers']==list(range(7,16))and rule['steps']==500 and rule['shrink']==.5
 for p,h in rule['input_pins'].items():assert sha(p)==h,(p,'hash')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();nets=[];pins={};models={};rows=[];folds=[]
 (D/'FIT_PROCESS_ID_v1.json').open('x').write(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),rule_sha256=sha(rp)))+'\n')
 def limits():
  assert time.monotonic()-start<115,'Fixed fit time ceiling approached'
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<7_000_000_000,'Fixed memory ceiling approached'
 local=json.loads(Path(rule['local_retirement']).read_text());term=json.loads(Path(rule['terminal_retirement']).read_text())
 for ret in (local,term):assert ret['approved']and ret['reviewed_by']=='root'and ret['complete_calls']==32 and ret['all_owned_PID_PPID_PGID_absent']and ret['all_guards_clean']
 ls={x['seed']:x for x in local['evaluations']};ts={x['seed']:x for x in term['evaluations']};assert sorted(ls)==sorted(ts)==list(range(32))
 meta={x['rank']:x for x in json.loads(Path(rule['capture_protocol']).read_text())['network_metadata']}
 for rank in range(32):
  cp=Path(ls[rank]['capture_path']);tp=Path(ts[rank]['capture_path']);wp=Path(meta[rank]['weights_path']);certp=Path(ts[rank]['certificate'])
  for p,h in ((cp,ls[rank]['capture_sha256']),(tp,ts[rank]['capture_sha256']),(wp,meta[rank]['weights_sha256']),(certp,ts[rank]['certificate_sha256'])):
   assert sha(p)==h,(str(p),'hash');pins[str(p)]=h
  W=np.load(wp,mmap_mode='r',allow_pickle=False);assert W.shape==(16,1024,1024)
  ws=torch.from_numpy(np.array(W[8:16],dtype=np.float32,copy=True));del W
  with np.load(cp,allow_pickle=False)as z,np.load(tp,allow_pickle=False)as t:
   baseline=z['predictions_float32'];assert baseline.tobytes()==t['predictions_float32'].tobytes()
   xs=[];ss=[];rs=[];ys=[]
   for layer in range(7,15):
    xs.append(np.column_stack([z[f'L{layer}_features30'],z[f'L{layer}_state64']]))
    ss.append(z[f'L{layer}_sigma']);rs.append(z['raw_means_float64'][layer-7]);ys.append(z['target_means_float64'][layer-7])
   xs.append(np.column_stack([t['features30'],t['state64']]));ss.append(t['sigma']);rs.append(t['raw_final_float64']);ys.append(t['target_final_float64'])
   X=np.asarray(xs,dtype=np.float32);sigma=np.asarray(ss,dtype=np.float64);raw=np.asarray(rs,dtype=np.float64);truth=np.asarray(ys,dtype=np.float64)
   assert X.shape==(9,1024,94)and np.isfinite(X).all()and np.all(sigma>0)
   assert np.maximum(raw,0).astype(np.float32).tobytes()==baseline[7:16].tobytes()
   gates=X[:,:,5].copy();assert np.all((gates>=0)&(gates<=1))
   mse=float(np.mean((baseline[15].astype(float)-truth[-1])**2));assert mse>0
   flop=int(json.loads(certp.read_text())['flops'])
   nets.append(dict(rank=rank,X=X,sigma=sigma,W=ws,gate=torch.from_numpy(gates),raw=torch.from_numpy(raw[-1].copy()),truth=torch.from_numpy(truth[-1].copy()),local_residual=(truth-np.maximum(raw,0))/sigma,baseline_mse=mse,flops=flop))
  limits()
 X=np.stack([x['X']for x in nets]);sig=torch.from_numpy(np.stack([x['sigma']for x in nets]).astype(np.float32))
 sums=np.sum(X.astype(np.float64),axis=2);sums2=np.sum(X.astype(np.float64)**2,axis=2)
 residual2=np.asarray([np.mean(x['local_residual']**2,axis=1)for x in nets])
 eye=torch.eye(9)[None,:,None,:].expand(32,9,1024,9)
 print(json.dumps(dict(stage='data_ready',seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
 for fi,held in enumerate(rule['folds']):
  train=[r for r in range(32)if r not in held]
  mean=sums[train].sum(0)/(28*1024);var=sums2[train].sum(0)/(28*1024)-mean**2;std=np.maximum(np.sqrt(np.maximum(var,0)),1e-6)
  scale=np.sqrt(residual2[train].mean(0));assert np.isfinite(scale).all()
  inputs=torch.cat([torch.from_numpy(((X-mean[None,:,None,:])/std[None,:,None,:]).astype(np.float32)),eye],dim=-1)
  scales=torch.from_numpy(scale.astype(np.float32))
  torch.manual_seed(20261015+fi);rng=np.random.default_rng(20261015+fi)
  model=torch.nn.Sequential(torch.nn.Linear(103,16),torch.nn.SiLU(),torch.nn.Linear(16,1))
  torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  def forward(batch):
   local=model(inputs[batch]).squeeze(-1)*sig[batch]*scales[None,:,None]*.5
   predictions=[]
   for j,rank in enumerate(batch):
    n=nets[rank];delta=local[j,0]
    for li in range(1,9):delta=n['gate'][li]*(n['W'][li-1].T@delta)+local[j,li]
    predictions.append(torch.clamp(n['raw']+delta.double(),min=0).float().double())
   return predictions
  permutation=None
  for step in range(500):
   if step%14==0:permutation=rng.permutation(train)
   batch=permutation[2*(step%14):2*(step%14)+2].tolist()
   optimizer.zero_grad(set_to_none=True);preds=forward(batch)
   loss=sum(torch.mean((p-nets[r]['truth'])**2)/nets[r]['baseline_mse']for p,r in zip(preds,batch))/2
   assert torch.isfinite(loss);loss.backward();optimizer.step()
   if step%25==0:limits()
  with torch.no_grad():
   for rank,pred in zip(held,forward(held)):
    n=nets[rank];mse=float(torch.mean((pred-n['truth'])**2));assert mse>0 and torch.isfinite(pred).all()
    logm=math.log(mse/n['baseline_mse']);added=rule['added_network_FLOPs'];bill0=max(n['flops']/2**41,.1);bill1=max((n['flops']+added)/2**41,.1);logb=math.log(bill1/bill0)
    rows.append(dict(rank=rank,fold=fi,baseline_mse=n['baseline_mse'],mse=mse,log_mse_ratio=logm,log_bill_ratio=logb,log_adjusted_ratio=logm+logb,baseline_flops=n['flops'],projected_added_flops=added))
  for name,p in model.state_dict().items():models[f'fold{fi}_{name}']=p.detach().numpy().copy()
  models[f'fold{fi}_input_mean']=mean;models[f'fold{fi}_input_std']=std;models[f'fold{fi}_output_scale']=scale
  subset=[r for r in rows if r['fold']==fi];lm=float(np.mean([r['log_mse_ratio']for r in subset]));ln=float(np.mean([r['log_adjusted_ratio']for r in subset]))
  folds.append(dict(fold=fi,heldout=held,mean_log_MSE_ratio=lm,mean_log_adjusted_ratio=ln,geometric_MSE_change_pct=100*math.expm1(lm),geometric_adjusted_change_pct=100*math.expm1(ln)))
  print(json.dumps(dict(stage='fold_complete',fold=fi,seconds=time.monotonic()-start)),flush=True);limits()
  del model,optimizer,inputs
 logs=np.asarray([r['log_mse_ratio']for r in rows]);net=np.asarray([r['log_adjusted_ratio']for r in rows]);mean=float(logs.mean());netmean=float(net.mean());negative=sum(f['mean_log_adjusted_ratio']<0 for f in folds);eligible=netmean<=-.003 and negative>=6
 mp=D/'FOLD_NONLINEAR_OUTPUT_ONLY_MODELS_v1.npz';assert not mp.exists();np.savez(mp,**models)
 result=dict(status='NONLINEAR_FORECAST_REQUIRES_ROOT_NATIVE_REVIEW'if eligible else'FIXED_NONLINEAR_FORECAST_CLOSED',eligible_for_native_review=eligible,mean_final_log_MSE_ratio=mean,geometric_final_MSE_change_pct=100*math.expm1(mean),mean_final_log_adjusted_ratio=netmean,geometric_final_adjusted_change_pct=100*math.expm1(netmean),SE_log_pct=100*float(logs.std(ddof=1))/np.sqrt(32),networks_MSE_lower=int((logs<0).sum()),networks_adjusted_lower=int((net<0).sum()),folds_negative=negative,folds=folds,per_network=rows,zero_baseline_selected=netmean>=0,forecast_only=True,no_native_calls=True,no_all32_finalfit=True,rule_sha256=sha(rp),fit_script_sha256=sha(__file__),data_pins=pins,fold_models_sha256=sha(mp),seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,added_network_FLOPs=rule['added_network_FLOPs'],added_network_charged_ops=rule['added_network_charged_ops'],scope='Physical QR unchanged. Output-only exact fixed baseline mean-response recurrence; float32 auxiliary and final float64 add/clamp then float32 prediction. Native arithmetic remains to validate.')
 p=D/'NONLINEAR_OUTPUT_ONLY_BLOCK_CV_RESULT_v1.json';p.open('x').write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k]for k in ['status','geometric_final_MSE_change_pct','geometric_final_adjusted_change_pct','SE_log_pct','networks_MSE_lower','networks_adjusted_lower','folds_negative','folds','seconds','max_rss_bytes']}),flush=True)
if __name__=='__main__':main()
