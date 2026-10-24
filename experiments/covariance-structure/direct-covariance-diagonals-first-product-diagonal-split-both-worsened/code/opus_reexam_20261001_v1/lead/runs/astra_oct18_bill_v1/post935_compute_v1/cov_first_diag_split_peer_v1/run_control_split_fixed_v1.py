import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
plan=json.loads((D/'NATIVE_PLAN_v1.json').read_text())
for job in plan['jobs']:
    assert hashlib.sha256(Path(job['source']).read_bytes()).hexdigest()==job['sha256']
output=Path(plan['output']);output.mkdir(exist_ok=False)
(output/'CONTROLLER_v1.json').write_text(json.dumps(dict(pid=os.getpid(),jobs=plan['jobs'],no_endpoint_inspection_between=True),indent=2)+'\n')
for job in plan['jobs']:
    group=output/job['label']
    result=subprocess.run([sys.executable,'-B',str(D/'harness/run_guarded_fusion_v1.py'),job['source'],str(job['rank']),str(group)])
    if result.returncode:raise SystemExit(result.returncode)
    guard=json.loads((group/'GUARD_v1.json').read_text())
    assert guard.get('completed')and guard.get('whole_group_retired')and not guard.get('remaining_rows'),guard
    print('retired',job['label'],flush=True)
(output/'COMPLETE_v1.json').write_text(json.dumps(dict(jobs=[j['label']for j in plan['jobs']],all_groups_retired=True),indent=2)+'\n')
