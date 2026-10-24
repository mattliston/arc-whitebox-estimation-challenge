"""Fixed saved-gradient geometry diagnostic; no model or target-file reads."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
OLD=HERE/'information_v1'
OLD_SHA='73bec2488604faebd40ecd2024990969a0983df8fe6b8c9b9004bb04333cf3f9'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
 began=time.monotonic()
 def expired(*_):raise TimeoutError('15s saved-gradient geometry')
 signal.signal(signal.SIGALRM,expired);signal.alarm(15)
 require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'nice10/BLAS1')
 require(sha(OLD/'result.json')==OLD_SHA,'Exact original failed pilot')
 old=json.loads((OLD/'result.json').read_text())
 require(old['passed'] is False and old['completed'] is False and old['gradient_calls']==20 and old['nonlinear_calls']==0,'Preserved failure/20 gradients')
 out=HERE/'gradient_geometry_diagnostic_v1';out.mkdir(exist_ok=False)
 result=dict(passed=False,source_sha256=sha(__file__),original_failed_result_sha256=OLD_SHA,
   model_calls=0,target_files_read=False,old_pilot_still_failed=True,native_derivative_claim=False,
   direction_rules_evaluated=1,parameters_saved=False,raw_or_adjusted_score_claim=False)
 try:
  import numpy as np
  require(np.__version__=='1.26.4' and not any(x in sys.modules for x in ('torch','flopscope','threadpoolctl')),'Saved NumPy arrays only')
  rows=[];bindings={};calls=old['calls'][:20]
  for index,call in enumerate(calls):
   require(call['kind']=='backward' and call['call']==index and call['passed'] and call['completed'],'Complete reverse call')
   item=call['gradient'];p=Path(item['path'])
   require(p.is_relative_to(OLD) and sha(p)==item['sha256']==old['artifacts'][str(p)],'Frozen gradient pin')
   g=np.load(p,allow_pickle=False)
   require(g.dtype==np.float32 and g.shape==(64,64) and np.isfinite(g).all()
      and hashlib.sha256(g.tobytes()).hexdigest()==item['raw_sha256'],'Gradient bytes/type')
   rows.append(g.astype(np.float64).reshape(-1));bindings[str(p)]=item['sha256']
  G=np.stack(rows);K=G@G.T;ev=np.linalg.eigvalsh(K)
  trace=float(np.trace(K));require(trace>0,'Nonzero gradient metric')
  damping=1e-3*trace/20
  regularized=K+damping*np.eye(20)
  z=np.linalg.solve(regularized,np.ones(20))
  candidate=-(G.T@z);require(np.isfinite(candidate).all() and np.linalg.norm(candidate)>0,'Finite fixed diagnostic direction')
  candidate*=64/np.linalg.norm(candidate)
  pooled=G.mean(0);pooled*= -64/np.linalg.norm(pooled)
  dc,dp=G@candidate,G@pooled
  a=math.fsum(x['scalar']*2e-8 for x in calls)/20
  cosine=float(candidate@pooled/(np.linalg.norm(candidate)*np.linalg.norm(pooled)))
  linear_fraction=-2e-8*float(dc.mean())/a
  info=bool((dc<0).all() and abs(cosine)<=.9 and linear_fraction>=.10)
  result.update(passed=True,gradient_bindings=bindings,ranks=[x['rank'] for x in calls],
   gram_eigenvalues=ev.tolist(),gram_trace=trace,regularized_condition=float(np.linalg.cond(regularized)),
   damping=damping,gradient_coherence=float(np.sum(G.mean(0)**2)/(trace/20)),
   mean_gradient_norm=float(np.linalg.norm(G.mean(0))),
   per_net_pooled_direction=dp.tolist(),per_net_common_direction=dc.tolist(),
   old_negative_count=int((dp<0).sum()),common_negative_count=int((dc<0).sum()),
   common_to_pooled_cosine=cosine,proposal_loss_baseline=a,
   common_unit_step_first_order_fraction=linear_fraction,information_premise=info,
   information_premise_definition='All20 recorded proposal dots negative, abs cosine<=.9, unit-step first-order fraction>=.10; not an endpoint/finite-step gain',
   no_gauss_newton_or_fisher_claim=True,no_direction_or_model_payload_saved=True)
 except BaseException as exc:result.update(passed=False,error=f'{type(exc).__name__}: {exc}')
 result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 require(result['elapsed_s']<15 and result['peak_rss_bytes']<2_000_000_000,'15s/2GB bound')
 require(sha(OLD/'result.json')==OLD_SHA,'Old pilot unchanged')
 write(out/'result.json',result);write(out/'completion.json',dict(passed=result['passed'],result_sha256=sha(out/'result.json')))
 signal.alarm(0);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
