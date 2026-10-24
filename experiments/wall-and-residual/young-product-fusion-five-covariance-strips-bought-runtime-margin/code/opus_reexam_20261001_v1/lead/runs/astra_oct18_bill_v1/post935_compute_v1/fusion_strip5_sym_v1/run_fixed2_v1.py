import json,os,subprocess,sys,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
candidate=D/'candidate_95c_fusion_strip5_sym_HELD_v1.py'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='a41895cf5d6186e299a30cec33b8792f3610aa96aef5d90f924d15acf1117851'
output=D/'native_fixed2_v1';output.mkdir(exist_ok=False)
(output/'CONTROLLER_v1.json').write_text(json.dumps(dict(pid=os.getpid(),ranks=[35,200],candidate=str(candidate),policy='Bothranks fixedbeforeexecution, freshprocesses, no endpointinspectionbetween'),indent=2)+'\n')
for rank in (35,200):
    group=output/('rank'+str(rank))
    result=subprocess.run([sys.executable,'-B',str(D/'harness/run_guarded_fusion_v1.py'),str(candidate),str(rank),str(group)])
    if result.returncode:raise SystemExit(result.returncode)
    guard=json.loads((group/'GUARD_v1.json').read_text())
    assert guard.get('completed') and guard.get('whole_group_retired') and not guard.get('remaining_rows'),guard
    print('retired',rank,flush=True)
(output/'COMPLETE_v1.json').write_text(json.dumps(dict(ranks=[35,200],all_groups_retired=True),indent=2)+'\n')
