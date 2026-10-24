"""Once-only dispatch of root-approved 36 own-source anchors; no remaining/fit."""
import argparse,os,subprocess,sys,json
from pathlib import Path
from dispatch_support_once_v1 import ROOT,HERE,RUNS,L,sha,read,utc,flags,guardcheck

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ops-clearance',type=Path,required=True);ap.add_argument('--ops-sha256',required=True);args=ap.parse_args()
 assert sys.platform=='linux'
 rp=HERE/'ROOT_ANCHORS_REVIEW_v1.json';assert sha(rp)=='8bbf9e4b74b986336bc2d644f43bac2fe77d58a57a9cb89ea59f9b2438817f7c'
 assert sha(args.ops_clearance)==args.ops_sha256
 review,ops=read(rp),read(args.ops_clearance)
 assert review['successful'] and review['capture_stage_authorized']=='anchors' and not review['one_fixed_fit_authorized'] and not review['remaining_stage_authorized']
 assert ops['successful'] and ops['root_review_sha256']==sha(rp)
 for p,h in review['checked_source_and_plan_pins'].items():assert sha(ROOT/p)==h,p
 assert sha(HERE/'SECOND_FIT_RECIPE_v1.json')==review['second_fit_recipe_sha256']
 for item in review['required_retired_guards']:assert guardcheck(item['path'])==item['sha256']
 for f in read(HERE/'PARENT_SOURCES_v1.json')['families']:assert sha(ROOT/f['science']['path'])==f['science']['sha256']
 ready=read(HERE/'capture_plans_v1/CAPTURE_READY_v1.json')
 for family in ready['outputs']:
  for stage,item in family['outputs'].items():
   assert not Path(item['out']).exists(),item['out']
   if 'audit_guard_path' in item:assert not Path(item['audit_guard_path']).exists(),item['audit_guard_path']
 launch=HERE/'ANCHORS_LAUNCH_v1.json';log=HERE/'ANCHORS_CONTROLLER_v1.log'
 for path in (launch,log,HERE/'ANCHORS_FINAL_v1.json',HERE/'CAPTURE_FINAL_v1.json',HERE/'pipeline_launches_anchors_v1',HERE/'pipeline_launches_remaining_v1'):assert not path.exists(),path
 # Named runtime qualifier has independently retired cleanly before this dispatch.
 qp=HERE.parent/'runtime_b54_onpolicy_paid_v1/QUALIFICATION_GUARD_v1.json';qualifier_guard=guardcheck(qp)
 guards=[];truth=[];lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
 for line in lines:
  words=line.split()[1:];names={Path(w).name for w in words}
  assert not names&{'server_linux_v7.py','run_capture_pipeline_v1.py','run_onpolicy_recovery_v1.py','run_capture_v1.py','run_group_capture_v1.py','audit_capture_v1.py','fit_onpolicy_second_step_recovery_v1.py','build_parent_sources_v1.py','check_observer_synthetic_v1.py','fit_second_step_v1.py','qualify_parent_sources_v1.py','prepare_capture_v1.py','run_b54_onpolicy_qualification_v1.py'},line
  if 'guard_linux_v7.py' in names:
   child=words[words.index('--')+1:];assert any(Path(w).name=='truth_chunk_cpu_additional12_v1.py' for w in child),line
   receipt=words[words.index('--receipt')+1]
   assert any(receipt.startswith(str(RUNS/n)+'/seed') for n in ('fresh12_additional_initial4_cpu_chunks_v1','fresh12_additional_middle4_cpu_chunks_v1')),line
   guards.append(line)
  elif 'truth_chunk_cpu_additional12_v1.py' in names:truth.append(line)
 assert len(guards)<=8 and len(truth)<=8
 cg=Path('/sys/fs/cgroup/arc-estimator-v5');ev,local=flags(cg/'memory.events'),flags(cg/'memory.events.local')
 assert ev['max']==8033 and local['max']==128 and all(ev[k]==0 for k in ('oom','oom_kill','oom_group_kill')) and int((cg/'memory.swap.current').read_text())==0
 live={};owned=0
 for c in cg.glob('arc-*'):
  try:
   if not (c/'cgroup.procs').read_text().strip():continue
   events=flags(c/'memory.events');assert all(v==0 for v in events.values()) and int((c/'memory.swap.current').read_text())==0
   current=int((c/'memory.current').read_text());owned+=current;live[c.name]=dict(events=events,current_bytes=current)
  except FileNotFoundError:pass
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:'))}
 assert memory['SwapTotal']==0
 parentcurrent=int((cg/'memory.current').read_text());outside=max(0,parentcurrent-owned)
 worst=outside+21*6500000000;physical_margin=memory['MemAvailable']+owned-21*6500000000-19000000000
 assert worst<160*1024**3 and physical_margin>0,(worst,physical_margin)
 preflight=dict(utc=utc(),guards=guards,truth_workers=truth,parent_events=ev,parent_local_events=local,live_children=live,memory=memory,parent_current=parentcurrent,outside_live_children_bytes=outside,conservative21_full_caps_bytes=worst,physical_reserve_margin_bytes=physical_margin,named_runtime_qualifier_retired_guard_sha256=qualifier_guard)
 command=['/home/arcworker/arc-envs/judge/bin/python','-B',str(HERE/'run_capture_pipeline_v1.py'),'--stage','anchors','--root-review',str(rp),'--root-review-sha256',sha(rp)]
 env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','ARC_BLAS_THREADS')}
 with log.open('x') as stream:process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 raw=Path('/proc/%d/stat'%process.pid).read_text();start=int(raw[raw.rfind(')')+2:].split()[19])
 result=dict(utc=utc(),pid=process.pid,startticks=start,command=command,environment=env,controller_sha256=sha(HERE/'run_capture_pipeline_v1.py'),launcher_sha256=sha(__file__),root_review_sha256=sha(rp),ops_clearance_sha256=sha(args.ops_clearance),preflight=preflight,exact36_predictions=True,no_remaining522_or_fit_or_retry=True)
 with launch.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(utc=result['utc'],pid=process.pid,startticks=start,launch_sha256=sha(launch))),flush=True)
if __name__=='__main__':main()
