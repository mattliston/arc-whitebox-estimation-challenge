"""Stage one science candidate for paired measurement: adds it to copies of the store32 plan and the two fresh plans
(local + VM), copies the file to the VM at the same relative path, and prints the VM launch commands.
usage: stage_candidate_v1.py <science.py> <name>
"""
import hashlib, json, subprocess, sys
from pathlib import Path
R = Path('/Users/matt/arc-white-box-estimation'); O = 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
NC = O + '/lead/runs/astra_oct20_bill_v1/runtime_first128_frozen_census_v1'; F = O + '/lead/runs/fable_oct21_final_v1'
VM = '/private/tmp/claude-501/-Users-matt-arc-white-box-estimation/fa350431-f721-4021-9dc7-6a57a67ff7ed/scratchpad/vm.sh'
src, name = Path(sys.argv[1]).resolve(), sys.argv[2]; rel = str(src.relative_to(R))
b = src.read_bytes(); sha = hashlib.sha256(b).hexdigest()
entry = dict(path=rel, bytes=len(b), sha256=sha, effective_sha256=sha, role='frozen_candidate')
def run(*a): subprocess.run(a, check=True)
run(VM, 'ssh', 'mkdir -p /home/arcworker/arc-workspace/' + str(Path(rel).parent)); run(VM, 'put', str(src), '/home/arcworker/arc-workspace/' + rel)
out = R / F / ('stage_' + name); out.mkdir(parents=True, exist_ok=True)
plans = {'store32': R / NC / 'NATIVE_PLAN_LINUX_v1.json', 'original8': R / NC / 'fresh12_confirmation_v1/original8/PLAN_v1.json', 'extension4': R / NC / 'fresh12_confirmation_v1/extension4/PLAN_v1.json'}
cmds = []
for key, p in plans.items():
    d = json.loads(p.read_text()); d['candidates'][name] = entry
    target = out / (key + '_PLAN_v1.json'); target.write_text(json.dumps(d, indent=1) + '\n'); psha = hashlib.sha256(target.read_bytes()).hexdigest()
    vp = '/home/arcworker/arc-workspace/' + str(target.relative_to(R)); run(VM, 'ssh', 'mkdir -p ' + str(Path(vp).parent)); run(VM, 'put', str(target), vp)
    L = '/home/arcworker/arc-workspace/' + O + '/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
    if key == 'store32':
        cmds.append(f"/home/arcworker/arc-envs/judge/bin/python -B {L}/launch_frozen_jobs12_markline_v2.py --plan {vp} --plan-sha {psha} --candidate {name} --source-sha {sha} --tag {name}_native32_v1 --authorization 'fable lead 2026-10-21 paired measurement'")
    else:
        jobs = 8 if key == 'original8' else 4
        cmds.append(f"cd /home/arcworker/arc-runs && nohup /home/arcworker/arc-envs/judge/bin/python -B {L}/run_science_linux_markline_v1.py --root /home/arcworker/arc-workspace --plan {vp} --candidate {name} --out /home/arcworker/arc-runs/fresh12_{key}_{name}_confirmation_v1 --state-dir /home/arcworker/arc-runs/admission --client-python /home/arcworker/arc-envs/client/bin/python --server-python /home/arcworker/arc-envs/server/bin/python --jobs {jobs} --max-active 36 --blas-threads 4 --setup paced --cgroup-root /sys/fs/cgroup/arc-estimator-v5 > /home/arcworker/arc-runs/fresh12_{key}_{name}_confirmation_v1_controller.log 2>&1 &")
(out / 'LAUNCH_CMDS.sh').write_text('\n'.join(cmds) + '\n'); print(json.dumps(dict(name=name, sha256=sha, launch_script=str(out / 'LAUNCH_CMDS.sh'))))
