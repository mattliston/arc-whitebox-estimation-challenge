"""Two explicitly reviewed stages: six own anchors, then two fixed capture waves.

No retry. Original native-empty boundaries, guard, groups3, BLAS4 and shared36.
Maximum12 native workers. Remaining522 require a separate root anchor review.
"""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1';PORT=A/'cloud_linux_capacity_v7';L=A/'cloud_linux_markline_v1'
sys.path.insert(0,str(PORT));from common_linux_v7 import thread_env
PY='/home/arcworker/arc-envs/judge/bin/python'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
PREREG='42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def write(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def guardcheck(path,threads):
 g=read(path)
 assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==threads
 assert g['max_owned_swap_bytes']==0 and g['cap_bytes']==6500000000 and g['reserve_bytes']==19000000000 and g['max_active']==36
 assert all(v==0 for v in g['cgroup_events_final'].values())
 return sha(path)
def native_empty():
 lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
 assert not [line for line in lines if 'server_linux_v7.py' in [Path(word).name for word in line.split()[1:]]]
def start(command,tag,threads,evidence):
 log=RUNS/(tag+'.log');launch=LAUNCHES/(tag+'_LAUNCH_v1.json')
 assert not log.exists() and not launch.exists()
 with log.open('x') as stream:
  child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT,
   env=dict(os.environ,**thread_env(threads),ARC_BLAS_THREADS=str(threads)))
 stat=Path('/proc/%d/stat'%child.pid).read_text();startticks=int(stat[stat.rfind(')')+2:].split()[19])
 write(launch,dict(utc=utc(),pid=child.pid,startticks=startticks,command=command,log=str(log),evidence=evidence,
  controller_sha256=sha(__file__),root_review_sha256=ROOT_REVIEW_SHA,no_auto_retry_or_fire=True))
 print(json.dumps(dict(utc=utc(),tag=tag,pid=child.pid,startticks=startticks)),flush=True)
 return child

def native(family,stage,jobs):
 item=family['outputs'][stage];out=Path(item['out']);assert not out.exists()
 plan=ROOT/item['plan']['path'];assert sha(plan)==item['plan']['sha256']
 runner=L/'run_science_linux_markline_v1.py' if stage=='control' else HERE/'run_capture_v1.py'
 cmd=[PY,'-B',str(runner)]
 if stage!='control':
  recipe=ROOT/item['recipe']['path'];assert sha(recipe)==item['recipe']['sha256'];cmd+=['--recipe',str(recipe)]
 cmd+=['--root',str(ROOT),'--plan',str(plan),'--candidate','teacher','--out',str(out),
  '--state-dir',str(RUNS/'admission'),'--client-python','/home/arcworker/arc-envs/client/bin/python',
  '--server-python','/home/arcworker/arc-envs/server/bin/python','--jobs',str(jobs),'--max-active','36',
  '--blas-threads','4','--setup','paced','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5']
 assert '--warmup-each' not in cmd
 return start(cmd,'onpolicy_first128_'+family['key']+'_'+stage+'_v1',4,
  dict(plan_sha256=sha(plan),runner_sha256=sha(runner),source_sha256=family['teacher']['sha256']))
def audit(family,stage):
 item=family['outputs'][stage];receipt=Path(item['audit_guard_path']);assert not receipt.exists()
 auditor=HERE/'audit_capture_v1.py';assert sha(auditor)==RECIPE['capture_auditor_sha256']
 cmd=[PY,'-B',str(PORT/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
  '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',PY,'-B',str(auditor),
  '--family',family['key'],'--stage',stage,'--ready-sha256',READY]
 child=start(cmd,'onpolicy_first128_'+family['key']+'_'+stage+'_audit_v1',1,dict(auditor_sha256=sha(auditor),ready_sha256=READY))
 assert child.wait()==0
 guardsha=guardcheck(receipt,1);path=Path(item['out'])/'AUDIT_v1.json';report=read(path)
 assert report['successful'] and report['family']==family['key'] and report['stage']==stage
 assert report['source_sha256']==family['teacher']['sha256'] and report['no_truth_arrays_read']
 if stage=='anchor':assert report['all3_anchor_predictions_byte_identical'] and report['all3_anchor_charged_invoices_exact']
 return dict(audit_sha256=sha(path),audit_guard_sha256=guardsha,audit_path=str(path),guard_path=str(receipt))
def one_anchor(family):
 control=native(family,'control',1);anchor=native(family,'anchor',1)
 exits=[control.wait(),anchor.wait()];assert exits==[0,0],(family['key'],exits)
 proof=audit(family,'anchor')
 write(LAUNCHES/(family['key']+'_ANCHOR_COMPLETE_v1.json'),proof)
 return family['key'],proof
def one_remaining(family):
 child=native(family,'remaining',4);assert child.wait()==0
 proof=audit(family,'remaining')
 write(LAUNCHES/(family['key']+'_REMAINING_COMPLETE_v1.json'),proof)
 return family['key'],proof

def batch(families,workers,function):
 results,failures={},[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(function,family):family['key'] for family in families}
  for future in concurrent.futures.as_completed(futures):
   try:
    key,result=future.result();results[key]=result
   except BaseException as error:
    failures.append(dict(family=futures[future],error=repr(error)))
    print(json.dumps(dict(utc=utc(),failed=failures[-1])),flush=True)
 return results,failures

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('anchors','remaining'),required=True)
 ap.add_argument('--root-review',type=Path,required=True);ap.add_argument('--root-review-sha256',required=True)
 ap.add_argument('--fit-after-capture',action='store_true');args=ap.parse_args()
 global ROOT_REVIEW_SHA,READY,MANIFEST,LAUNCHES,RECIPE
 assert sys.platform=='linux'
 ROOT_REVIEW_SHA=args.root_review_sha256;assert sha(args.root_review)==ROOT_REVIEW_SHA;review=read(args.root_review)
 assert review['successful'] and review['capture_stage_authorized']==args.stage and review['controller_sha256']==sha(__file__)
 assert review['max_native_workers']==12 and review['actual_prediction_ceiling']==558 and review['extra_warmup_calls']==0
 assert sha(PORT/'guard_linux_v7.py')==GUARD
 for item in review['required_retired_guards']:
  assert guardcheck(Path(item['path']),item['blas_threads'])==item['sha256']
 ready_path=HERE/'capture_plans_v1/CAPTURE_READY_v1.json';READY=review['capture_ready_sha256'];assert sha(ready_path)==READY
 ready=read(ready_path);MANIFEST=ready['source_manifest']['sha256']
 assert ready['successful'] and ready['prereg_sha256']==PREREG
 assert sha(HERE/'PREREG_v1.json')==PREREG and sha(HERE/'PARENT_SOURCES_v1.json')==MANIFEST
 assert ready['actual_full_predict_calls']==ready['ceiling_predictions']==558 and ready['extra_warmup_calls']==0
 assert ready['max_capture_native']==12 and ready['jobs_per_family']==4
 qualified=HERE/'parent_qualification_v1/QUALIFIED_v1.json';assert sha(qualified)==review['parent_qualification_sha256']
 assert read(qualified)['successful'] and read(qualified)['all_six_official_cold_warm_arithmetic_identical']
 for row in ready['harness']+ready['control_harness']+[ready['hook']]:assert sha(ROOT/row['path'])==row['sha256']
 RECIPE=read(HERE/'SECOND_FIT_RECIPE_v1.json');assert sha(HERE/'SECOND_FIT_RECIPE_v1.json')==review['second_fit_recipe_sha256']
 assert RECIPE['source_manifest_sha256']==MANIFEST and RECIPE['capture_ready_sha256']==READY
 assert sha(HERE/'audit_capture_v1.py')==RECIPE['capture_auditor_sha256']
 assert sha(HERE/'fit_second_step_v1.py')==RECIPE['fitter_sha256']
 assert not (HERE/'CAPTURE_FINAL_v1.json').exists()
 families=ready['outputs'];assert [r['key'] for r in families]==['fold0','fold1','fold2','fold3','fold4','full90']
 LAUNCHES=HERE/('pipeline_launches_'+args.stage+'_v1');LAUNCHES.mkdir(exist_ok=False)
 native_empty()
 if args.stage=='anchors':
  assert not args.fit_after_capture and not (HERE/'ANCHORS_FINAL_v1.json').exists()
  results,failures=batch(families,6,one_anchor)
  native_empty()
  write(HERE/'ANCHORS_FINAL_v1.json',dict(utc=utc(),successful=not failures,families=results,failures=failures,
   actual_scored_predictions=36 if not failures else None,extra_warmup_calls=0,ready_sha256=READY,source_manifest_sha256=MANIFEST,
   prereg_sha256=PREREG,controller_sha256=sha(__file__),root_review_sha256=ROOT_REVIEW_SHA,
   all_six_prediction_and_charged_invoice_anchors_exact=not failures,no_truth_arrays_read=True,remaining522_not_launched=True))
  assert not failures,failures
  return
 anchors_path=HERE/'ANCHORS_FINAL_v1.json';assert sha(anchors_path)==review['anchor_final_sha256'];anchors=read(anchors_path)
 assert anchors['successful'] and anchors['all_six_prediction_and_charged_invoice_anchors_exact']
 assert anchors['source_manifest_sha256']==MANIFEST and anchors['ready_sha256']==READY
 assert review['root_reviewed_all_six_anchor_custody']
 for family in families:
  key=family['key'];proof=anchors['families'][key];item=family['outputs']['anchor']
  assert sha(Path(item['out'])/'AUDIT_v1.json')==proof['audit_sha256']
  assert guardcheck(Path(item['audit_guard_path']),1)==proof['audit_guard_sha256']
 results={key:dict(anchor=proof) for key,proof in anchors['families'].items()};failures=[]
 for wave,names in enumerate(ready['waves']):
  assert names==(['fold0','fold1','fold2'] if wave==0 else ['fold3','fold4','full90'])
  native_empty()
  selected=[family for family in families if family['key'] in names];assert len(selected)==3
  done,errors=batch(selected,3,one_remaining)
  for key,proof in done.items():results[key]['remaining']=proof
  failures.extend(errors)
  write(LAUNCHES/('WAVE%d_FINAL_v1.json'%wave),dict(utc=utc(),successful=not errors,families=done,failures=errors))
  if errors:break
 native_empty()
 final=dict(utc=utc(),successful=not failures,families=results,failures=failures,captured_predictions=540 if not failures else None,
  unhooked_controls=18 if not failures else None,ceiling_predictions=558,actual_full_predict_calls=558 if not failures else None,
  extra_warmup_calls=0,ready_sha256=READY,source_manifest_sha256=MANIFEST,prereg_sha256=PREREG,
  all_six_prediction_and_charged_invoice_anchors_exact=True,controller_sha256=sha(__file__),root_review_sha256=ROOT_REVIEW_SHA,
  anchor_final_sha256=sha(anchors_path),no_truth_arrays_read=True,no_candidate_export=True,no_auto_retry_or_fire=True)
 write(HERE/'CAPTURE_FINAL_v1.json',final);assert not failures,failures
 if args.fit_after_capture:
  assert review['one_fixed_fit_authorized']
  fit_review=HERE/'ROOT_SECOND_FIT_REVIEW_v1.json';assert sha(fit_review)==review['second_fit_root_review_sha256']
  assert read(fit_review)['successful'] and read(fit_review)['one_fixed_fit_authorized']
  native_empty();receipt=RUNS/'onpolicy_first128_second_fit_v1_GUARD_v1.json';assert not receipt.exists()
  script=HERE/'fit_second_step_v1.py'
  cmd=[PY,'-B',str(PORT/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
   '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',PY,'-B',str(script)]
  child=start(cmd,'onpolicy_first128_second_fit_v1',1,dict(fitter_sha256=sha(script),capture_final_sha256=sha(HERE/'CAPTURE_FINAL_v1.json')))
  assert child.wait()==0;guardsha=guardcheck(receipt,1);fitted=RUNS/'onpolicy_first128_second_fit_v1/FITTED_v1.json';report=read(fitted)
  write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status=report['status'],passed=report['passed'],
   fitted_sha256=sha(fitted),fit_guard_sha256=guardsha,no_candidate_export=True,no_native_launch=True,no_auto_fire=True))
 print(json.dumps(dict(utc=utc(),capture_successful=True,optional_fit_completed=args.fit_after_capture)),flush=True)
if __name__=='__main__':main()
