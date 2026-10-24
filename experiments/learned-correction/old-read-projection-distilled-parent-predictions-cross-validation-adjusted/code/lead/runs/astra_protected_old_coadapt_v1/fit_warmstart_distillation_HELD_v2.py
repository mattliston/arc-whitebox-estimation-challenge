"""One frozen QRteacher distillation on protected-old native TRAIN32 captures."""
from pathlib import Path
import ast,base64,hashlib,json,math,os,resource,time,zlib
import numpy as np
import psutil
import torch
D=Path(__file__).resolve().parent;R=D.parent;START=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
PLAN=json.loads((D/'FROZEN_PLAN_v2.json').read_text())
assert PLAN['source_sha256']==sha(__file__) and PLAN['approved_by']=='root'
assert all(os.environ.get(k)=='1'for k in PLAN['thread_variables']) and os.getpriority(os.PRIO_PROCESS,0)==10
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
write(D/'PROCESS_v1.json',dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),source_sha256=sha(__file__),plan_sha256=sha(D/'FROZEN_PLAN_v2.json')))
PEAK=0
def guard():
 global PEAK
 PEAK=max(PEAK,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 assert PEAK<7_500_000_000 and psutil.virtual_memory().available/psutil.virtual_memory().total>=.50
 assert time.monotonic()-START<300,'Fixed 300s ceiling: no checkpoint selection/retry.'
def load():
 source=Path(PLAN['parent_source']);assert sha(source)==PLAN['parent_sha256']
 tree=ast.parse(source.read_text());literal=[ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_GNN_LITERAL'for t in n.targets)]
 assert len(literal)==1
 params={k:np.frombuffer(zlib.decompress(base64.b64decode(v[1])),dtype='<f4').reshape(v[0]).copy()for k,v in literal[0].items()}
 receipt=json.loads((D/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text());protocol=json.loads((D/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text())
 assert receipt['approved']and receipt['all_owned_PID_PPID_PGID_absent']and receipt['complete_calls']==32 and receipt['actual_managed_exits_observed']
 assert sha(D/'CAPTURE_PROTOCOL_REGISTERED_v1.json')==PLAN['capture_protocol_sha256']
 data=[]
 for e in sorted(receipt['evaluations'],key=lambda x:x['seed']):
  r=e['seed'];assert sha(e['capture_path'])==e['capture_sha256'];ref=protocol['network_metadata'][r]['reference']
  cert=json.loads(Path(e['certificate']).read_text());assert sha(e['certificate'])==e['certificate_sha256']
  old=json.loads(Path(ref['certificate']).read_text());assert sha(ref['certificate'])==ref['certificate_sha256']
  with np.load(e['capture_path'],allow_pickle=False)as z:
   h=z['hidden1_128'].copy();sigma=z['sigma'].copy();raw=z['raw_final_float64'].copy();truth=z['target_final_float64'].copy();base=z['predictions_float32'][15].copy();teacher=z['teacher_final_float32'].astype('d')
   assert np.array_equal(z['head'],params['head'][15])and np.array_equal(z['head_b'],params['head_b'][15]);rms=np.asarray(z['rms'],dtype='f4').reshape(());gain=np.asarray(z['gain'],dtype='f4').reshape(())
  assert h.shape==(1024,128)and np.array_equal(np.maximum(raw,0).astype('f4'),base)
  scale0=np.multiply(sigma.astype('f4'),rms);mse=float(np.mean((base.astype('d')-truth)**2));qr_mse=float(np.mean((teacher-truth)**2))
  data.append(dict(rank=r,h=h,raw=raw,truth=truth,teacher=teacher,base=base,scale0=scale0,gain=float(gain),baseline_mse=mse,qr_mse=qr_mse,cheap_flops=int(cert['flops']),qr_flops=int(old['flops'])))
 assert [x['rank']for x in data]==list(range(32));return params,data
class Adapt(torch.nn.Module):
 def __init__(self,p):
  super().__init__();self.names=('w2','b2','head','bias');values=(p['enc2_w'],p['enc2_b'],p['head'][15],p['head_b'][15:16])
  self.delta=torch.nn.ParameterList()
  for i,v in enumerate(values):
   self.register_buffer('initial'+str(i),torch.from_numpy(v.copy()));self.register_buffer('scale'+str(i),torch.tensor(max(float(np.sqrt(np.mean(v.astype('d')**2))),1e-3),dtype=torch.float32));self.delta.append(torch.nn.Parameter(torch.zeros_like(getattr(self,'initial'+str(i)))))
  self.register_buffer('w4',torch.from_numpy(p['enc4_w'].copy()));self.register_buffer('b4',torch.from_numpy(p['enc4_b'].copy()))
 def parameters_at(self):return [getattr(self,'initial'+str(i))+getattr(self,'scale'+str(i))*d for i,d in enumerate(self.delta)]
 def forward(self,h):
  w,b,head,bias=self.parameters_at();v=h@w.T+b;v=v/(1+torch.exp(-v));return ((v@self.w4.T+self.b4)@head+bias).reshape(-1)
 def prior(self):return torch.stack([torch.mean(d*d)for d in self.delta]).mean()
def fit(data,p,label):
 guard();begin=time.monotonic();torch.manual_seed(20261016)
 model=Adapt(p);H=torch.from_numpy(np.concatenate([d['h']for d in data]));RAW=torch.from_numpy(np.concatenate([d['raw']for d in data]));Y=torch.from_numpy(np.concatenate([d['teacher']for d in data]));S=torch.from_numpy(np.concatenate([d['scale0']for d in data]));G=torch.from_numpy(np.concatenate([np.full(1024,d['gain'],dtype='f4')for d in data]));normalizer=max(float(np.mean(np.concatenate([(d['teacher']-d['raw'])**2 for d in data]))),1e-24);IW=torch.full((len(H),),1/normalizer,dtype=torch.float64)
 with torch.no_grad():BASE=model(H).detach();assert torch.equal(model(H)-BASE,torch.zeros_like(BASE))
 opt=torch.optim.Adam(model.parameters(),lr=.001);rng=torch.Generator().manual_seed(20261016)
 for step in range(500):
  if step%25==0:guard()
  b=torch.randint(len(H),(4096,),generator=rng);opt.zero_grad(set_to_none=True)
  correction=(S[b]*(model(H[b])-BASE[b]))*G[b];pred=torch.clamp_min(RAW[b]+correction.double(),0)
  # Forward clamp/cast matches production; straight-through float32 cast derivative.
  pred=pred.float().double();loss=torch.mean((pred-Y[b])**2*IW[b])+model.prior()
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 model.eval();vals=model.parameters_at();path=D/(label+'_weights_v1.npz')
 np.savez(path,**{n:v.detach().numpy().copy()for n,v in zip(model.names,vals)})
 norms={n:float(torch.sqrt(torch.mean(d.detach()**2)))for n,d in zip(model.names,model.delta)}
 meta=dict(label=label,train_ranks=[d['rank']for d in data],steps=500,wall_seconds=time.monotonic()-begin,weights_path=str(path),weights_sha256=sha(path),relative_RMS_updates=norms,teacher_delta_foldtrain_RMS=math.sqrt(normalizer))
 write(D/(label+'_FREEZE_v1.json'),meta);return model,meta
def evaluate(model,baseline,d):
 with torch.no_grad():
  h=torch.from_numpy(d['h']);difference=model(h)-baseline(h);correction=(torch.from_numpy(d['scale0'])*difference)*d['gain']
 pred=np.maximum(d['raw']+correction.numpy().astype('d'),0).astype('f4').astype('d');mse=float(np.mean((pred-d['truth'])**2))
 teacher_before=float(np.mean((d['base'].astype('d')-d['teacher'])**2));teacher_after=float(np.mean((pred-d['teacher'])**2))
 extra=int(PLAN['extra_branch_flops']);bill=lambda f:max(f/2**41,.1)
 return dict(rank=d['rank'],cheap_mse=d['baseline_mse'],qr_mse=d['qr_mse'],adapted_mse=mse,teacher_mse_before=teacher_before,teacher_mse_after=teacher_after,teacher_recovery=1-teacher_after/teacher_before,log_MSE_vs_QR=math.log(mse/d['qr_mse']),log_MSE_vs_cheap=math.log(mse/d['baseline_mse']),cheap_flops=d['cheap_flops'],qr_flops=d['qr_flops'],extra_branch_flops=extra,log_bill_vs_QR=math.log(bill(d['cheap_flops']+extra)/bill(d['qr_flops'])),cheap_log_bill_vs_QR=math.log(bill(d['cheap_flops'])/bill(d['qr_flops'])),log_adjusted_vs_QR=math.log(mse/d['qr_mse'])+math.log(bill(d['cheap_flops']+extra)/bill(d['qr_flops'])),log_adjusted_vs_cheap=math.log(mse/d['baseline_mse'])+math.log(bill(d['cheap_flops']+extra)/bill(d['cheap_flops'])),cheap_log_MSE_vs_QR=math.log(d['baseline_mse']/d['qr_mse']),cheap_log_adjusted_vs_QR=math.log(d['baseline_mse']/d['qr_mse'])+math.log(bill(d['cheap_flops'])/bill(d['qr_flops'])))
def main():
 p,data=load();baseline=Adapt(p).eval();scores=[];folds=[]
 assert sum(t.numel()for t in baseline.parameters())==16577
 for f in range(8):
  held=set(range(4*f,4*f+4));m,meta=fit([d for d in data if d['rank']not in held],p,'fold'+str(f));rows=[evaluate(m,baseline,d)for d in data if d['rank']in held];scores.extend(rows)
  meta['mean_log_adjusted_vs_QR']=float(np.mean([x['log_adjusted_vs_QR']for x in rows]));meta['network_wins']=sum(x['log_adjusted_vs_QR']<0 for x in rows);meta['teacher_error_recovery']=1-sum(x['teacher_mse_after']for x in rows)/sum(x['teacher_mse_before']for x in rows);folds.append(meta)
  print(json.dumps(dict(fold_complete=f+1,fit_seconds=meta['wall_seconds'],elapsed_seconds=time.monotonic()-START,estimated_remaining_seconds=(7-f)*meta['wall_seconds'],no_partial_scores_reported=True)),flush=True)
 keys=['log_MSE_vs_QR','log_MSE_vs_cheap','log_bill_vs_QR','log_adjusted_vs_QR','log_adjusted_vs_cheap','cheap_log_MSE_vs_QR','cheap_log_bill_vs_QR','cheap_log_adjusted_vs_QR']
 means={k:float(np.mean([x[k]for x in scores]))for k in keys};pct={k:100*math.expm1(v)for k,v in means.items()}
 recovery=1-sum(x['teacher_mse_after']for x in scores)/sum(x['teacher_mse_before']for x in scores)
 eligible=pct['log_adjusted_vs_QR']<=-.3 and recovery>0;final=None
 if eligible:_,final=fit(data,p,'all32')
 result=dict(status='ELIGIBLE_CV_DISTILLATION_HELD_FOR_ROOT'if eligible else'CLOSE_BELOW_FIXED_TOTAL_ADJUSTED_GATE',mean_logs=means,geometric_percent=pct,teacher_error_recovery=recovery,network_wins_vs_QR=sum(x['log_adjusted_vs_QR']<0 for x in scores),network_count=32,log_SE_vs_QR=float(np.std([x['log_adjusted_vs_QR']for x in scores],ddof=1)/np.sqrt(32)),folds=folds,per_network=scores,final_fit=final,elapsed_seconds=time.monotonic()-START,peak_rss_bytes=PEAK,source_sha256=sha(__file__),plan_sha256=sha(D/'FROZEN_PLAN_v2.json'),training_target='cachedQRfinalfloat32 only',MC_truth_evaluation_only=True,no_independent_gain_claim=True)
 write(D/'BLOCKED_CV_RESULT_v1.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('folds','per_network')}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(D/'STOPPED_v1.json',dict(error=repr(e),elapsed_seconds=time.monotonic()-START));raise
