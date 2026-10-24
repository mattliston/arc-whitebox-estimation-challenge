"""Sequential native boundary run, pinned to the measured 0.12.3 environments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

O = Path(__file__).resolve().parents[3]
R = O / 'lead/runs'
candidate, ranks, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
assert candidate.is_file()
out.mkdir(exist_ok=False)
runtime = R / 'astra_crunch_runtime_v1/failure336631_v1'
versions = {}
code = "import json,flopscope,importlib.metadata as m; print(json.dumps({'version':flopscope.__version__,'packages':{x.metadata.get('Name'):x.version for x in m.distributions() if any(w in x.metadata.get('Name','').lower() for w in ('flop','numpy'))}}))"
for name in ('boundary310_v1', 'server310_openblas_v1'):
    versions[name] = json.loads(subprocess.check_output([str(runtime/name/'bin/python'), '-c', code], text=True))
    assert versions[name]['version'].split('+')[0] == '0.12.3'
plan = dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks, runtime=versions)
(out/'PLAN_v1.json').write_text(json.dumps(plan, indent=2)+'\n')
env = dict(os.environ)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    env[key] = '1'
cmd = [sys.executable, '-B', str(R/'astra_crunch_physics_v1/guard_memory50_v1.py'), '--record', str(out/'GUARD_v1.json'), '--seconds', '600', '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', sys.executable, '-B', str(R/'astra_oct18_lead_v1/boundary_run_census_v2.py'), str(candidate), ranks, str(out), '--short-setup']
with (out/'run.log').open('x') as log:
    result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
print('native_exit', result.returncode, flush=True)
sys.exit(result.returncode)
