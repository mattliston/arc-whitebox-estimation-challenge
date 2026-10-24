"""HELD: fixed fresh35 then fresh200, root must authorize isolated SDK slot."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
candidate=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/candidate_95c_tensor_membership_l13_HELD_v1.py')
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='8a168e46265215578609013a9eb5487da4e8e712b22c7bf8e62f592da5d37614'
output=args.output.resolve();output.mkdir()
with(output/'CONTROLLER_v1.json').open('x')as f:json.dump(dict(pid=os.getpid(),ranks=[35,200],candidate=str(candidate),sha256='8a168e46265215578609013a9eb5487da4e8e712b22c7bf8e62f592da5d37614',policy='Fixed two fresh processes; no endpoint inspection between. 7.5GB/free50 guard.'),f,indent=2)
for rank in(35,200):
 group=output/('rank'+str(rank))
 result=subprocess.run([sys.executable,'-B','/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/run_guarded_fusion_v1.py',str(candidate),str(rank),str(group)])
 if result.returncode:raise SystemExit(result.returncode)
 guard=json.loads((group/'GUARD_v1.json').read_text())
 assert guard.get('completed')and guard.get('whole_group_retired')and not guard.get('remaining_rows')
 print('retired',rank,flush=True)
with(output/'COMPLETE_v1.json').open('x')as f:json.dump(dict(ranks=[35,200],candidate_sha256='8a168e46265215578609013a9eb5487da4e8e712b22c7bf8e62f592da5d37614',all_groups_retired=True),f,indent=2)
