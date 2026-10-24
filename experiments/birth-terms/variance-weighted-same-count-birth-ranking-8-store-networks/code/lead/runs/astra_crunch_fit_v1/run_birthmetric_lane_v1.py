"""Root-assigned fixed stage2 ranks16/200/250; one guarded native worker at a time."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
D=Path(__file__).resolve().parent;R=D.parent;O=R.parents[1]
B=R/'astra_crunch_birthmetric_v1';G=R/'astra_crunch_physics_v1/guard_memory50_v1.py'
NPY=O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
VPY=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
C=B/'candidate_birthmetric_v2.py';J=B/'judge_v2.py';PLAN=B/'EXTEND8_PREREG_v1.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(C)=='afbc69ce09232157e3e845a4d0b79ebb6a47f7bd0f3d906809ccb9801a8e7361'
plan=json.loads(PLAN.read_text());assert plan['fixed_next_ranks'][:3]==[16,200,250]
env=dict(os.environ)
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[k]='1'
rows=[]
for rank in (16,200,250):
 pressure=subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=10)
 free=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1));assert free>=60,(rank,free)
 result=B/f'res{rank}_v2.json';guard=B/f'guard{rank}_v2.json';log=B/f'log{rank}_v2.txt'
 assert not result.exists()and not guard.exists()and not log.exists()
 entry=dict(rank=rank,free_percent=free,candidate_sha256=sha(C),judge_sha256=sha(J),guard_sha256=sha(G),prereg_sha256=sha(PLAN),pid=os.getpid(),threads=1,nice_child=10)
 with (B/f'entry{rank}_v2.json').open('x')as f:json.dump(entry,f,indent=2);f.write('\n')
 cmd=[str(VPY),str(G),'--record',str(guard),'--seconds','150','--footprint-limit','7500000000','--rss-backstop','7500000000',str(NPY),'-B',str(J),str(C),str(rank),'ctl',str(result)]
 print(json.dumps(dict(stage='native_start',rank=rank,free_percent=free)),flush=True)
 with log.open('x')as f:completed=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert completed.returncode==0,(rank,completed.returncode)
 g=json.loads(guard.read_text());assert g['completed']and g['whole_group_retired']and g['exit_code']==0 and g['remaining_rows']==[]
 a=json.loads(result.read_text());c=json.loads((R/'opus_kinkoracle_v1'/f'res_ctl_{rank}.json').read_text())
 assert a['birthmetric']['calls']==15 and a['rank']==rank
 row=dict(rank=rank,U=a['U'],control_U=c['U'],flops=a['flops'],control_flops=c['flops'],MSE_change_pct=100*(a['U']/c['U']-1),bill_change_pct=100*(a['flops']/c['flops']-1),adjusted_change_pct=100*(a['U']*a['flops']/(c['U']*c['flops'])-1),guard_elapsed_s=g['elapsed_s'],guard_max_rss_bytes=g['max_rss_bytes'],guard_min_free_percent=g['min_pressure_free_pct'],all_owned_processes_retired=True)
 rows.append(row);print(json.dumps(dict(stage='native_retired',**row)),flush=True)
out=dict(status='ROOT_FIXED_STAGE2_LANE_COMPLETE',ranks=[16,200,250],candidate_sha256=sha(C),rows=rows,no_candidate_changes=True,no_tuning=True)
with (D/'BIRTHMETRIC_LANE_RESULT_v1.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out),flush=True)
