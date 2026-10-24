"""One exact reviewed remaining522 controller with conditional fixed second fit."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dispatch_support_once_v1 import ROOT,HERE,RUNS,L,sha,read,utc,flags
from run_capture_pipeline_v1 import guardcheck

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ops-sha256',required=True);args=ap.parse_args();assert sys.platform=='linux'
 rp=HERE/'ROOT_REMAINING_REVIEW_v1.json';fp=HERE/'ROOT_SECOND_FIT_REVIEW_v1.json';sp=HERE/'REMAINING_AND_FIT_STAGE_READY_v2.json';op=HERE/'OPS_REMAINING522_AND_FIXEDFIT_CLEARANCE_v1.json'
 assert sha(rp)=='41bfc944df1b04ae8ebffff9632fcf86ee3c744c3f777863d6408c18d1eb157e'
 assert sha(fp)=='0ad39c6f73b908a98a3640845acd8d228313e1ea927eacc62f7c1468f42918bf'
 assert sha(sp)=='2351c9d52800d4541f168142ae92b12017e83e9d2a1e3bbe397dd3a40b12e7f5' and sha(op)==args.ops_sha256
 review,fit,stage,ops=read(rp),read(fp),read(sp),read(op)
 assert ops['status']=='CLEAR_ONE_OP1_REMAINING522_PLUS_ROOT_CONDITIONAL_FIXEDFIT' and ops['root_review_sha256']==sha(rp)
 assert review['successful'] and review['one_fixed_fit_authorized'] and review['capture_stage_authorized']=='remaining'
 assert fit['successful'] and fit['one_fixed_fit_authorized'] and review['second_fit_root_review_sha256']==sha(fp)
 assert review['stage_ready_sha256']==sha(sp) and review['mandatory_spare21cap_preflight']
 for section in ('pins','closed_prior_branch_pins'):
  for p,h in stage[section].items():assert sha(ROOT/p)==h,p
 assert sha(HERE.parent/'runtime_b54_onpolicy_paid_v1/CLOSURE_v1.json')==review['oldop_closure_sha256']
 for item in review['required_retired_guards']:assert guardcheck(Path(item['path']),item['blas_threads'])==item['sha256']
 for family in read(HERE/'PARENT_SOURCES_v1.json')['families']:assert sha(ROOT/family['science']['path'])==family['science']['sha256']
 assert sha(HERE/'ANCHORS_FINAL_v1.json')==review['anchor_final_sha256'] and read(HERE/'ANCHORS_FINAL_v1.json')['successful']
 for family in read(HERE/'capture_plans_v1/CAPTURE_READY_v1.json')['outputs']:
  item=family['outputs']['remaining']
  for path in (Path(item['out']),Path(item['audit_guard_path']),RUNS/('onpolicy_first128_'+family['key']+'_remaining_v1.log')):assert not path.exists(),path
 launch=HERE/'REMAINING_LAUNCH_v1.json';log=HERE/'REMAINING_CONTROLLER_v1.log'
 for path in (launch,log,HERE/'pipeline_launches_remaining_v1',HERE/'CAPTURE_FINAL_v1.json',HERE/'PIPELINE_FINAL_v1.json',RUNS/'onpolicy_first128_second_fit_v1',RUNS/'onpolicy_first128_second_fit_v1_GUARD_v1.json',RUNS/'onpolicy_first128_second_fit_v1.log'):assert not path.exists(),path
 lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();guards=[];truth=[]
 for line in lines:
  words=line.split()[1:];names={Path(w).name for w in words}
  assert not names&{'server_linux_v7.py','run_capture_pipeline_v1.py','run_onpolicy_recovery_v1.py','run_capture_v1.py','run_group_capture_v1.py','audit_capture_v1.py','fit_onpolicy_second_step_recovery_v1.py','fit_second_step_v1.py','fit_occupancy_v1.py','price_occupancy_v1.py','price_occupancy_v2.py','run_b54_onpolicy_qualification_v1.py'},line
  if 'guard_linux_v7.py' in names:
   child=words[words.index('--')+1:];assert any(Path(w).name=='truth_chunk_cpu_additional12_v1.py' for w in child),line
   receipt=words[words.index('--receipt')+1];assert any(receipt.startswith(str(RUNS/n)+'/seed') for n in ('fresh12_additional_initial4_cpu_chunks_v1','fresh12_additional_middle4_cpu_chunks_v1')),line
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
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:'))};assert memory['SwapTotal']==0
 current=int((cg/'memory.current').read_text());outside=max(0,current-owned);bound=outside+21*6500000000;reserve=memory['MemAvailable']+owned-21*6500000000-19000000000
 assert bound<160*1024**3 and reserve>0,(bound,reserve)
 preflight=dict(utc=utc(),guards=guards,truth_workers=truth,parent_events=ev,parent_local_events=local,live_children=live,memory=memory,parent_current=current,outside_live_children_bytes=outside,conservative21_full_caps_bytes=bound,physical_reserve_margin_bytes=reserve,no_external_small=True)
 command=['/home/arcworker/arc-envs/judge/bin/python','-B',str(HERE/'run_capture_pipeline_v1.py'),'--stage','remaining','--root-review',str(rp),'--root-review-sha256',sha(rp),'--fit-after-capture']
 env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','ARC_BLAS_THREADS')}
 with log.open('x') as stream:process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 raw=Path('/proc/%d/stat'%process.pid).read_text();start=int(raw[raw.rfind(')')+2:].split()[19])
 result=dict(utc=utc(),pid=process.pid,startticks=start,command=command,environment=env,controller_sha256=sha(HERE/'run_capture_pipeline_v1.py'),launcher_sha256=sha(__file__),root_review_sha256=sha(rp),fit_root_review_sha256=sha(fp),ops_clearance_sha256=sha(op),preflight=preflight,remaining522_plus_only_one_authorized_fixed_fit=True,no_export_candidate_native_fire_or_retry=True)
 with launch.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(utc=result['utc'],pid=process.pid,startticks=start,launch_sha256=sha(launch))),flush=True)
if __name__=='__main__':main()
