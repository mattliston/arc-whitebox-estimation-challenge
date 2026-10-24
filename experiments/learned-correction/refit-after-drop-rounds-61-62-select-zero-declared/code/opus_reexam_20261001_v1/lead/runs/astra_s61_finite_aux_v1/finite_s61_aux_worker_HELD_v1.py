"""Finite queue adapter; worker17 imports and serve() execute byte-for-byte unchanged.
Only fixed existing eval requests are eligible. Atomic original rename; max24 claims.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];T=O/'trainer'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--home',type=Path,required=True);a=ap.parse_args()
 assert sha(a.plan)==a.plan_sha256;c=json.loads(a.plan.read_text());assert a.index in c['worker_indices'] and c['max_claims_per_worker']==24
 assert os.getpriority(os.PRIO_PROCESS,0)==10
 assert all(os.environ.get(k)==v for k,v in c['environment'].items())
 for p,h in c['pins'].items():assert sha(p)==h
 run=Path(c['shared_run']);tag='w%d'%a.index;assert not list((run/'claimed').glob('*.%s.json'%tag))
 worker=Path(c['worker']);source=worker.read_text();cut="say(event='start', run=str(RUN)";assert source.count(cut)==1
 prefix=source[:source.index(cut)];assert hashlib.sha256(prefix.encode()).hexdigest()==c['worker_prefix_sha256']
 sys.path.insert(0,str(T));sys.argv=[str(worker),str(run),str(a.index)]
 w={'__name__':'_finite_s61_exact_worker17','__file__':str(worker)}
 exec(compile(prefix,str(worker),'exec'),w)
 assert w['E'].THREADS==w['torch'].get_num_threads()==w['torch'].get_num_interop_threads()==1
 assert Path(w['E'].M.__file__).resolve()==T/'otr_m_v8.py'
 write(a.home/'CHILD_OWNER_v1.json',dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),nice=os.getpriority(os.PRIO_PROCESS,0),worker_index=a.index,worker_sha256=sha(worker),worker_prefix_sha256=c['worker_prefix_sha256'],shared_run=str(run)))
 eligible={x['filename']:x for x in c['frozen_queue']};claims=[]
 while len(claims)<24 and not(run/'STOP').exists():
  claim=None;request_data=None
  for request in sorted((run/'queue').glob('*.json')):
   if request.name not in eligible:continue
   try:raw=request.read_bytes()
   except FileNotFoundError:continue
   item=eligible[request.name];assert hashlib.sha256(raw).hexdigest()==item['sha256']
   queued=json.loads(raw);assert queued['mode']=='eval' and queued['id']==item['id'] and queued['rank']==item['rank']
   seed=int(queued['rank'])
   if seed>=7_000_000 and seed in w['allowed_seeds']:
    ready=w['truth_plan'].ready(seed)if w['truth_plan']else(O/'analyst/fresh'/('truth_seed%d.npz'%seed)).exists()
    if not ready:continue
   target=run/'claimed'/('%s.%s.json'%(request.stem,tag));assert not target.exists()
   try:os.rename(request,target)
   except FileNotFoundError:continue
   claim=target;request_data=queued;break
  if claim is None:break # Finite existing queue only; never wait for a later experiment.
  record=dict(claim=str(claim),claim_sha256=sha(claim),id=request_data['id'],rank=request_data['rank'],mode='eval')
  with (a.home/'claims.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
  w['serve'](claim) # Exact worker17 function, request contents untouched.
  out,err=w['result_path'](run,request_data),w['error_path'](run,request_data)
  assert out.exists()or err.exists(),'Claim did not publish a terminal result'
  record.update(result_path=str(out)if out.exists()else None,result_sha256=sha(out)if out.exists()else None,error_path=str(err)if err.exists()else None,error_sha256=sha(err)if err.exists()else None)
  claims.append(record);write(a.home/('COUNT_%02d_v1.json'%len(claims)),dict(completed_claims=len(claims),max_claims=24,worker_index=a.index))
  if w['process'].memory_info().rss>w['RSS_LIMIT']:
   w['say'](event='recycle',rss=w['process'].memory_info().rss);break
 write(a.home/'WORKER_COMPLETE_v1.json',dict(status='finite_existing_queue_complete',worker_index=a.index,claims=claims,completed_claims=len(claims),max_claims=24,source_unchanged=True,request_bytes_unchanged=True,no_new_requests=True,no_score_selection=True))
if __name__=='__main__':main()
