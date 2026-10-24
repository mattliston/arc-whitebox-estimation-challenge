"""Once-only launch of the exact reviewed metadata preparer; no capture dispatch."""
import os,subprocess,sys,json
from pathlib import Path
from dispatch_support_once_v1 import ROOT,HERE,RUNS,L,sha,read,utc,flags,guardcheck
assert sys.platform=='linux'
rp=HERE/'ROOT_PREPARE_REVIEW_v1.json';op=L/'OP1_METADATA_PREPARER_CLEARANCE_v1.json';pp=HERE/'FULL_EXPERIMENT_SOURCE_RECIPE_v1.json'
assert sha(rp)=='3699890f1bb6eb2728fc0a443d11df91d167e46fbcfc97e6749b69980a3c080c'
assert sha(op)=='271aac4ad48a9805abc1f42614a72ddc8fdf41004de4b3f999702eac7f3b0400'
review,ops,packet=read(rp),read(op),read(pp)
assert review['successful'] and review['one_prepare_authorized'] and ops['root_review_sha256']==sha(rp) and ops['recipe_sha256']==sha(pp)
item=packet['support_sequence'][3];assert ops['command']==item and ops['max_one_small_guard']
assert sha(HERE/item['script'])==review['script_sha256'] and sha(HERE/'parent_qualification_v1/QUALIFIED_v1.json')==review['qualification_sha256']
for section in ('source_pins','support_result_pins'):
 for path,digest in review[section].items():assert sha(ROOT/path)==digest,path
for itemguard in review['support_guards']:assert guardcheck(itemguard['path'])==itemguard['sha256']
receipt,log=Path(item['guard_receipt']),Path(item['log']);launch=HERE/'PREPARE_LAUNCH_v1.json'
assert not any(p.exists() for p in (receipt,log,launch,HERE/'capture_plans_v1'))
lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();guards=[];truth=[]
for line in lines:
 words=line.split()[1:];names={Path(w).name for w in words}
 assert not names&{'server_linux_v7.py','run_onpolicy_recovery_v1.py','run_capture_v1.py','run_group_capture_v1.py','audit_capture_v1.py','fit_onpolicy_second_step_recovery_v1.py','build_parent_sources_v1.py','check_observer_synthetic_v1.py','fit_second_step_v1.py','qualify_parent_sources_v1.py','prepare_capture_v1.py'},line
 if 'guard_linux_v7.py' in names:
  child=words[words.index('--')+1:];assert any(Path(w).name=='truth_chunk_cpu_additional12_v1.py' for w in child),line
  receiptpath=words[words.index('--receipt')+1];assert any(receiptpath.startswith(str(RUNS/name)+'/seed') for name in ('fresh12_additional_initial4_cpu_chunks_v1','fresh12_additional_middle4_cpu_chunks_v1')),line
  guards.append(line)
 elif 'truth_chunk_cpu_additional12_v1.py' in names:truth.append(line)
assert len(guards)<=8 and len(truth)<=8
cg=Path('/sys/fs/cgroup/arc-estimator-v5');ev,local=flags(cg/'memory.events'),flags(cg/'memory.events.local')
assert ev['max']==8033 and local['max']==128 and all(ev[k]==0 for k in ('oom','oom_kill','oom_group_kill')) and int((cg/'memory.swap.current').read_text())==0
live={}
for c in cg.glob('arc-*'):
 try:
  if not (c/'cgroup.procs').read_text().strip():continue
  events=flags(c/'memory.events');assert all(v==0 for v in events.values()) and int((c/'memory.swap.current').read_text())==0;live[c.name]=events
 except FileNotFoundError:pass
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:'))};assert memory['MemAvailable']>25500000000 and memory['SwapTotal']==0
env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','ARC_BLAS_THREADS')}
preflight=dict(utc=utc(),guards=guards,truth_workers=truth,parent_events=ev,parent_local_events=local,live_child_events=live,memory=memory,parent_current=int((cg/'memory.current').read_text()))
with log.open('x') as stream:process=subprocess.Popen(item['command'],cwd=ROOT,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
raw=Path('/proc/%d/stat'%process.pid).read_text();start=int(raw[raw.rfind(')')+2:].split()[19])
result=dict(utc=utc(),pid=process.pid,startticks=start,command=item['command'],environment=env,script_sha256=sha(HERE/item['script']),launcher_sha256=sha(__file__),root_review_sha256=sha(rp),ops_clearance_sha256=sha(op),preflight=preflight,no_capture_native_model_prediction_truth_fit_or_retry=True)
with launch.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(utc=result['utc'],pid=process.pid,startticks=start,launch_sha256=sha(launch))),flush=True)
