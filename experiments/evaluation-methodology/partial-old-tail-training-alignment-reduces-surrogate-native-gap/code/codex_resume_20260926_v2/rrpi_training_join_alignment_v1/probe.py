"""One target-free old/patched Torch comparison against fixed RRPI predictions."""
from pathlib import Path
import ast,ctypes,gc,hashlib,importlib.util,inspect,json,os,sys,time
D=Path(__file__).resolve().parent; H=D.parent; P=H.parent
def pin(p):
 p=Path(p).resolve(); raw=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
protocol=json.loads((D/'protocol.json').read_text())
assert len(sys.argv)==2 and pin(D/'protocol.json')['sha256']==sys.argv[1]
for r in protocol['pins']: assert pin(r['path'])==r,r['path']
for item in protocol['data']:
 for k in ('weights','native'): assert pin(item[k]['path'])==item[k]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):assert os.environ[k]=='1'
os.environ['JOIN_SCHED']='384-320-384'
for rel in ('fable_analyst_v1','fable_covhunter_v1','fable_engineer_v1','fable_physicist_v1/refit6b','fable_porter_v1'):
 sys.path.insert(0,str(P/rel))
sys.path.insert(0,str(H/'hill_climb_rrpi_extra32_warm_v1'))
import numpy as np
import torch
from native_parameters import load_s33_arrays
sys.path.insert(0,str(D))
import join_adapter
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
spec=importlib.util.spec_from_file_location('join_alignment_card',P/'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC=importlib.util.module_from_spec(spec);spec.loader.exec_module(LC)
torch.set_num_threads(1);torch.set_grad_enabled(False)
bd=inspect.getsource(LC.JT.make_bd);assert bd.count('for j in range(4):')==1
exec(compile(ast.parse(bd.replace('for j in range(4):','for j in range(3):')),'j3cut','exec'),LC.JT.__dict__)
trainer=H/'hill_climb_rrpi_extra32_warm_v1/train.py'
node=next(n for n in ast.parse(trainer.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install_star')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(trainer)+':star-only','exec'),globals())
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
def collect():gc.collect();relief(None,0)
arrays,dials=load_s33_arrays(np)
T={k:torch.from_numpy(np.array(v,copy=True)) for k,v in arrays.items()}
p={k[2:]:v for k,v in T.items() if k.startswith('p.')}
e={k[2:]:v for k,v in T.items() if k.startswith('e.')}
DL={k:torch.tensor(v,dtype=torch.float64) for k,v in dials.items()}
results=[]
for item in protocol['data']:
 weights=np.load(item['weights']['path'],allow_pickle=False)
 native=np.load(item['native']['path'],allow_pickle=False)
 assert weights.shape==(16,1024,1024) and native.shape==(16,1024)
 preds={}
 for arm in ('old','patched'):
  collect(); LC.JT.DIALS_T=DL
  cd=LC.card(torch,weights,p,T['a'],T['c'],e,fine_ckpt=False)
  cd.hooks['k4_15']=lambda layer,value,ctx:value
  install_star(cd)
  if arm=='patched':join_adapter.install(cd,LC.s2card.EVERY3)
  state=cd.initial();out=[];start=time.monotonic()
  for layer in range(16):
   state=cd.step(layer,state)
   out.append(state[0].clamp_min(0).to(torch.float32).cpu().numpy().astype(np.float64))
   collect()
  pred=np.stack(out);assert np.isfinite(pred).all()
  with (D/(arm+'_'+str(item['rank'])+'.npy')).open('xb') as f:np.save(f,pred,allow_pickle=False)
  preds[arm]=pred
  print(json.dumps(dict(rank=item['rank'],arm=arm,seconds=time.monotonic()-start,final_native_mse=float(np.mean((pred[15]-native[15])**2)))),flush=True)
  del cd,state,out,pred;collect()
 old=np.mean((preds['old']-native)**2,axis=1);new=np.mean((preds['patched']-native)**2,axis=1)
 assert preds['old'][:7].tobytes()==preds['patched'][:7].tobytes()
 results.append(dict(rank=item['rank'],old_native_mse_by_layer=old.tolist(),patched_native_mse_by_layer=new.tolist(),first7_bytes_equal=True))
 del weights,native,preds;collect()
ratio=sum(r['patched_native_mse_by_layer'][15] for r in results)/sum(r['old_native_mse_by_layer'][15] for r in results)
passed=ratio<=.8 and all(r['patched_native_mse_by_layer'][15]<r['old_native_mse_by_layer'][15] for r in results)
with (D/'result.json').open('x') as f:json.dump(dict(completed=True,rows=results,pooled_final_gap_ratio=ratio,admit_fixed_warm2=passed,targets_accessed=False,trained=False,protocol=pin(D/'protocol.json')),f,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,pooled_final_gap_ratio=ratio,admit_fixed_warm2=passed)),flush=True)
