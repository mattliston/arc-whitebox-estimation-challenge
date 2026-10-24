"""Saved-artifact-only failure diagnosis; no model or target-file access."""
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
PINS={
 OLD/'result.json':'73bec2488604faebd40ecd2024990969a0983df8fe6b8c9b9004bb04333cf3f9',
 OLD/'completion.json':'302d0b2d5e3a2337a9133543467c11fe504b866ab79721ab15abae9945af49d2',
 OLD/'finalization.json':'ec6dc2ed5f96930259e2b8c7e6599adbc7c2b4ea92c729ec295049a74d006602',
 HERE/'information_watch_v1/result.json':'5cd65c4fbcd69f0a6daf0a3f82307edfa06bdf818e31e5345b283d230139478c',
}
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
 began=time.monotonic()
 def expired(*_):raise TimeoutError('15s saved-artifact diagnosis')
 signal.signal(signal.SIGALRM,expired);signal.alarm(15)
 require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'BLAS1')
 for p,h in PINS.items():require(sha(p)==h,'Frozen failed receipt')
 old=json.loads((OLD/'result.json').read_text())
 require(old['passed'] is False and old['completed'] is False and old['gradient_calls']==20 and old['jvp_calls']==13 and old['nonlinear_calls']==0,'Original failure unchanged')
 out=HERE/'information_failure_diagnosis_v1';out.mkdir(exist_ok=False)
 result=dict(passed=False,original_pilot_still_failed=True,model_calls=0,target_files_read=False,
   thresholds_changed=False,native_score_measured=False,full20_magnitude_verdict=False,
   pins={str(p):h for p,h in PINS.items()},source_sha256=sha(__file__))
 try:
  import numpy as np
  require(np.__version__=='1.26.4' and not any(k in sys.modules for k in ('torch','flopscope','threadpoolctl')),'NumPy-only saved arrays')
  def read(p):
   p=Path(p);require(p.is_relative_to(OLD) and sha(p)==old['artifacts'][str(p)],'Same-run sealed artifact')
   return json.loads(p.read_text())
  def load(item):
   p=Path(item['path']);require(p.is_relative_to(OLD) and sha(p)==item['sha256']==old['artifacts'][str(p)],'Same-run array pin')
   a=np.load(p,allow_pickle=False)
   require(list(a.shape)==item['shape'] and str(a.dtype)==item['dtype'] and hashlib.sha256(a.tobytes()).hexdigest()==item['raw_sha256'],'Full array metadata')
   require(np.isfinite(a).all(),'Finite saved values');return a.astype(np.float64)
  direction=load(read(old['direction_seal']['path'])['direction'])
  rows=[]
  for i in range(14):
   base=old['calls'][i];attempt=read(OLD/f'call_{20+i:02d}_attempt.json')
   require(base['rank']==attempt['rank'] and base['kind']=='backward' and attempt['kind']=='jvp','Fixed matched observed roster')
   grad=load(base['gradient']);products=grad*direction
   reverse=float(np.sum(products));forward=attempt['tangent'];absolute=abs(reverse-forward)
   scale=max(abs(reverse),abs(forward));tolerance=1e-10+1e-4*scale
   j=load(attempt['served_jvp'])[-1]
   rows.append(dict(rank=base['rank'],reverse_dot=reverse,forward_readout=forward,
      absolute_error=absolute,relative_error=absolute/scale,unchanged_tolerance=tolerance,
      tolerance_multiple=absolute/tolerance,
      reverse_dot_cancellation=float(np.sum(np.abs(products)))/abs(reverse),
      proposal_a=base['scalar']*2e-8,proposal_b=forward*1e-8,proposal_c=float(np.mean(j*j))))
  require(all(x['tolerance_multiple']<=1 for x in rows[:-1]) and rows[-1]['tolerance_multiple']>1 and rows[-1]['rank']==52,'Reproduce original exact failed gate')
  a,b,c=(math.fsum(x['proposal_'+k] for x in rows)/14 for k in ('a','b','c'))
  step=max(0.,min(1.,-b/c));ratio=(a+2*step*b+step*step*c)/a
  result.update(passed=True,rows=rows,observed_roster_count=14,
   partial_failed_run_proposal_quadratic=dict(a=a,b=b,c=c,step=step,ratio=ratio,
      label='Post-hoc incomplete FP32-target proposal diagnostic only; neither original-target raw nor full20/native score'),
   no_parameter_proposal_or_candidate_created=True)
 except BaseException as exc:result.update(passed=False,error=f'{type(exc).__name__}: {exc}')
 result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 require(result['elapsed_s']<15 and result['peak_rss_bytes']<2_000_000_000,'15s/2GB bound')
 for p,h in PINS.items():require(sha(p)==h,'Old failures unchanged')
 write(out/'result.json',result);write(out/'completion.json',dict(passed=result['passed'],result_sha256=sha(out/'result.json')))
 signal.alarm(0);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
