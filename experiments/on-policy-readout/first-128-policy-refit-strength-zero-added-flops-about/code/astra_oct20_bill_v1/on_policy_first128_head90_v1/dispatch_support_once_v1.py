"""Dispatch one of the three exact reviewed support commands, never a sequence retry."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/home/arcworker/arc-workspace');HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs')
L=HERE.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
flags=lambda p:{line.split()[0]:int(line.split()[1]) for line in p.read_text().splitlines()}
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'

def guardcheck(path):
 g=read(path);assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==1
 assert g['cap_bytes']==6500000000 and g['reserve_bytes']==19000000000 and g['max_active']==36
 assert g['max_owned_swap_bytes']==0 and all(v==0 for v in g['cgroup_events_final'].values());return sha(path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,choices=(0,1,2),required=True);args=ap.parse_args();assert sys.platform=='linux'
 rp=HERE/'ROOT_SUPPORT_SEQUENCE_REVIEW_v1.json';assert sha(rp)=='ea4c673bb1ff10d6f571213a6143c5d6cba1d2196b72387c4b91d7f8957959f9';review=read(rp)
 qp=HERE/'ROOT_QUALIFY_REVIEW_v1.json';assert sha(qp)=='02d3a3e47f7d7128f20483b7c653d205fef9d9f3d79d049e8b5ad42b93b3bc65'
 op=L/'OP1_THREE_SUPPORT_AFTER_OLDFIT_CLEARANCE_v1.json';assert sha(op)=='ef149ce188067c3ae1f163553f74f7a510b8ee18d4265e69ea5631f4dd92bbac';ops=read(op)
 pp=HERE/'FULL_EXPERIMENT_SOURCE_RECIPE_v1.json';assert sha(pp)==review['packet_sha256']==ops['recipe_sha256'];packet=read(pp)
 assert review['successful'] and review['max_simultaneous_support_guards']==1 and review['truth_reservations']==8
 assert ops['root_review_sha256']==sha(rp) and ops['max_one_support_guard'] and ops['each_successful_original_guard_and_expected_proof_before_next']
 assert review['commands']==ops['commands']==packet['support_sequence'][:3]
 for path,digest in review['source_pins'].items():assert sha(ROOT/path)==digest,path
 assert sha(HERE/'PARENT_SOURCES_v1.json')==review['source_manifest_sha256']
 manifest=read(HERE/'PARENT_SOURCES_v1.json')
 for family in manifest['families']:
  for role in ('science','release'):assert sha(ROOT/family[role]['path'])==family[role]['sha256']
  for role in ('coef','build'):assert sha(ROOT/family[role+'_path'])==family[role+'_sha256']
 mandatory=review['mandatory_retirement'];oldg=Path(mandatory['old_fit_guard']);oldp=Path(mandatory['old_pipeline_final']);oldf=Path(mandatory['old_fitted']);pf=read(oldp);ff=read(oldf)
 assert guardcheck(oldg)==pf['fit_guard_sha256']==ops['preflight']['oldfit_guard_sha256']
 assert pf['successful'] and pf['fitted_sha256']==sha(oldf)==ops['preflight']['old_fitted_sha256'] and sha(oldp)==ops['preflight']['old_pipeline_final_sha256']
 assert ff['successful'] and ff['script_sha256']==mandatory['old_fit_child_sha256']
 bg=RUNS/'onpolicy_first128_parent_build_v1_GUARD_v1.json';assert guardcheck(bg)==mandatory['parent_builder_already_clean_guard_sha256']
 prior=[]
 for i in range(args.index):
  item=review['commands'][i];gp=Path(item['guard_receipt']);final=HERE/('SUPPORT_%d_FINAL_v1.json'%i);v=read(final)
  assert v['successful'] and v['index']==i and v['guard_sha256']==guardcheck(gp) and v['log_sha256']==sha(item['log'])
  assert v['expected_proofs_all_pass'] and v['root_review_sha256']==sha(rp);prior.append(dict(path=str(final),sha256=sha(final)))
 item=review['commands'][args.index];receipt,log=Path(item['guard_receipt']),Path(item['log']);launch=HERE/('SUPPORT_%d_LAUNCH_v1.json'%args.index)
 assert not any(p.exists() for p in (receipt,log,launch,HERE/('SUPPORT_%d_FINAL_v1.json'%args.index)))
 assert not (HERE/'parent_qualification_v1').exists()
 lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();guards=[];truth=[]
 forbidden={'server_linux_v7.py','run_onpolicy_recovery_v1.py','run_capture_v1.py','run_group_capture_v1.py','audit_capture_v1.py','audit_onpolicy_capture_recovery_v1.py','fit_onpolicy_second_step_recovery_v1.py','build_parent_sources_v1.py','check_observer_synthetic_v1.py','fit_second_step_v1.py','qualify_parent_sources_v1.py'}
 for line in lines:
  words=line.split()[1:];names={Path(w).name for w in words}
  assert not names&forbidden,line
  if 'guard_linux_v7.py' in names:
   assert '--' in words;child=words[words.index('--')+1:];assert any(Path(w).name=='truth_chunk_cpu_additional12_v1.py' for w in child),line
   rp2=words[words.index('--receipt')+1]
   assert any(rp2.startswith(str(RUNS/name)+'/seed') for name in ('fresh12_additional_initial4_cpu_chunks_v1','fresh12_additional_middle4_cpu_chunks_v1')),line
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
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:'))}
 assert memory['MemAvailable']>25500000000 and memory['SwapTotal']==0
 env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','ARC_BLAS_THREADS')}
 preflight=dict(utc=utc(),truth_guards=guards,truth_workers=truth,parent_events=ev,parent_local_events=local,live_child_events=live,memory=memory,parent_current=int((cg/'memory.current').read_text()))
 with log.open('x') as stream:
  process=subprocess.Popen(item['command'],cwd=ROOT,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 raw=Path('/proc/%d/stat'%process.pid).read_text();start=int(raw[raw.rfind(')')+2:].split()[19])
 result=dict(utc=utc(),index=args.index,name=item['name'],pid=process.pid,startticks=start,command=item['command'],environment=env,script_sha256=sha(HERE/item['script']),launcher_sha256=sha(__file__),root_review_sha256=sha(rp),ops_clearance_sha256=sha(op),prior_supports=prior,preflight=preflight,no_preparer_capture_fullfit_model_prediction_or_truth=True,no_retry=True)
 with launch.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(utc=result['utc'],index=args.index,pid=process.pid,startticks=start,launch_sha256=sha(launch))),flush=True)
if __name__=='__main__':main()
