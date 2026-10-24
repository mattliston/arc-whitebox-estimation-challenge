"""One pretrained final-stage adaptation on immutable native TRAIN32 captures."""
from pathlib import Path
import ast,base64,hashlib,json,math,os,resource,time,zlib
import numpy as np
import psutil
import torch
D=Path(__file__).resolve().parent;R=D.parent;START=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
PLAN=json.loads((D/'FROZEN_PLAN_v1.json').read_text())
assert PLAN['source_sha256']==sha(__file__) and PLAN['approved_by']=='root'
assert all(os.environ.get(k)=='1'for k in PLAN['thread_variables']) and os.getpriority(os.PRIO_PROCESS,0)==10
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
write(D/'PROCESS_v1.json',dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),source_sha256=sha(__file__),plan_sha256=sha(D/'FROZEN_PLAN_v1.json')))
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
 hp=R/'astra_native_terminal_hidden_refit_v1';lp=R/'astra_native_terminal_linear_refit_v1'
 hr=json.loads((hp/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text());lr=json.loads((lp/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text())
 assert sha(hp/'ROOT_CAPTURE_RETIREMENT_v1.json')==PLAN['hidden_retirement_sha256'] and sha(lp/'ROOT_CAPTURE_RETIREMENT_v1.json')==PLAN['reference_retirement_sha256']
 assert hr['approved']and lr['approved']and hr['all_owned_PID_PPID_PGID_absent']and lr['all_owned_PID_PPID_PGID_absent']
 refs={e['seed']:e for e in lr['evaluations']};data=[]
 for e in sorted(hr['evaluations'],key=lambda x:x['seed']):
  r=e['seed'];ref=refs[r];assert sha(e['capture_path'])==e['capture_sha256']and sha(ref['capture_path'])==ref['capture_sha256']
  with np.load(e['capture_path'],allow_pickle=False)as z:h=z['hidden1_128'].copy()
  with np.load(ref['capture_path'],allow_pickle=False)as z:
   sigma=z['sigma'].copy();raw=z['raw_final_float64'].copy();truth=z['target_final_float64'].copy();base=z['predictions_float32'][15].copy()
   assert np.array_equal(z['head'],params['head'][15])and np.array_equal(z['head_b'],params['head_b'][15])
   rms=np.asarray(z['rms'],dtype='f4').reshape(());gain=np.asarray(z['gain'],dtype='f4').reshape(())
  assert h.shape==(1024,128)and np.array_equal(np.maximum(raw,0).astype('f4'),base)
  # Preserve native sigma->float32 and multiplication order for the residual branch.
  scale0=np.multiply(sigma.astype('f4'),rms);mse=float(np.mean((base.astype('d')-truth)**2))
  data.append(dict(rank=r,h=h,raw=raw,truth=truth,base=base,scale0=scale0,gain=float(gain),baseline_mse=mse))
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
 model=Adapt(p);H=torch.from_numpy(np.concatenate([d['h']for d in data]));RAW=torch.from_numpy(np.concatenate([d['raw']for d in data]));Y=torch.from_numpy(np.concatenate([d['truth']for d in data]));S=torch.from_numpy(np.concatenate([d['scale0']for d in data]));G=torch.from_numpy(np.concatenate([np.full(1024,d['gain'],dtype='f4')for d in data]));IW=torch.from_numpy(np.concatenate([np.full(1024,1/d['baseline_mse'],dtype='d')for d in data]))
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
 meta=dict(label=label,train_ranks=[d['rank']for d in data],steps=500,wall_seconds=time.monotonic()-begin,weights_path=str(path),weights_sha256=sha(path),relative_RMS_updates=norms)
 write(D/(label+'_FREEZE_v1.json'),meta);return model,meta
def evaluate(model,baseline,d):
 with torch.no_grad():
  h=torch.from_numpy(d['h']);difference=model(h)-baseline(h);correction=(torch.from_numpy(d['scale0'])*difference)*d['gain']
 pred=np.maximum(d['raw']+correction.numpy().astype('d'),0).astype('f4').astype('d');mse=float(np.mean((pred-d['truth'])**2))
 return dict(rank=d['rank'],baseline_mse=d['baseline_mse'],mse=mse,log_ratio=math.log(mse/d['baseline_mse']))
def main():
 p,data=load();baseline=Adapt(p).eval();scores=[];folds=[]
 assert sum(t.numel()for t in baseline.parameters())==16577
 for f in range(8):
  held=set(range(4*f,4*f+4));m,meta=fit([d for d in data if d['rank']not in held],p,'fold'+str(f));rows=[evaluate(m,baseline,d)for d in data if d['rank']in held];scores.extend(rows)
  meta['mean_log_ratio']=float(np.mean([x['log_ratio']for x in rows]));meta['network_wins']=sum(x['log_ratio']<0 for x in rows);folds.append(meta)
  print(json.dumps(dict(fold_complete=f+1,fit_seconds=meta['wall_seconds'],elapsed_seconds=time.monotonic()-START,estimated_remaining_seconds=(7-f)*meta['wall_seconds'],no_partial_scores_reported=True)),flush=True)
 mean=float(np.mean([x['log_ratio']for x in scores]));final=None
 # Only the frozen 500-step model or zero. No shrink/epoch/ridge grid.
 if mean<0:
  _,final=fit(data,p,'all32')
 result=dict(status='NEGATIVE_CV_FIT_HELD_FOR_ROOT'if mean<0 else'BASELINE_ZERO_SELECTED_CLOSE',candidate_mean_log_MSE=mean,candidate_geometric_MSE_percent=100*math.expm1(mean),selected='warmstart'if mean<0 else'zero',network_wins=sum(x['log_ratio']<0 for x in scores),network_count=32,log_SE=float(np.std([x['log_ratio']for x in scores],ddof=1)/np.sqrt(32)),folds=folds,per_network=scores,final_fit=final,elapsed_seconds=time.monotonic()-START,peak_rss_bytes=PEAK,source_sha256=sha(__file__),plan_sha256=sha(D/'FROZEN_PLAN_v1.json'),no_new_models_or_captures=True,no_independent_gain_claim=True)
 write(D/'BLOCKED_CV_RESULT_v1.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('folds','per_network')}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(D/'STOPPED_v1.json',dict(error=repr(e),elapsed_seconds=time.monotonic()-START));raise
