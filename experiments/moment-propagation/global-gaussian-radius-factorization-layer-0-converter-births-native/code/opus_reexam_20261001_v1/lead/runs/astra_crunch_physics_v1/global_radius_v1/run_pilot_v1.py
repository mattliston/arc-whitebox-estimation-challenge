"""Fixed current-SDK 35,100 session; root-authorized after tiny/source review."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='1'
os.nice(10)
import hashlib,json,re,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
F=R/'astra_crunch_runtime_v1/failure336631_v1'
pre=json.loads((H/'PILOT_PREREG_v1.json').read_text());source=H/pre['candidate']
assert hashlib.sha256(source.read_bytes()).hexdigest()==pre['candidate_sha256']
free=int(re.search(r'System-wide memory free percentage: (\d+)%',subprocess.check_output(['/usr/bin/memory_pressure'],text=True)).group(1));assert free>=60,free
out=H/pre['candidate_output'];assert not out.exists();out.mkdir()
guard=H/'PILOT_GUARD_v1.json';assert not guard.exists()
cmd=[sys.executable,str(H.parent/'guard_memory50_v1.py'),'--record',str(guard),'--seconds','150','--footprint-limit','7500000000','--rss-backstop','7500000000',str(F/'env310_v1/bin/python'),'-B',str(F/'boundary_run_sdk_v1.py'),str(source),'35,100',str(out),'--setup-slowdown','1.3']
(H/'PILOT_LAUNCH_v1.json').write_text(json.dumps(dict(command=cmd,entry_free_percent=free,source_sha256=pre['candidate_sha256']),indent=2)+'\n')
with (H/'PILOT_LOG_v1.txt').open('x') as log:
    done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
g=json.loads(guard.read_text());assert done.returncode==0 and g['completed'] and g['whole_group_retired'],g
rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in rows]==[35,100]
print(json.dumps(dict(source_sha256=pre['candidate_sha256'],ranks=[x['rank'] for x in rows],bills=[x['flops'] for x in rows],requests=[x['requests'] for x in rows],guard_retired=True,maxrss=g['max_rss_bytes'],minfree=g['min_pressure_free_pct'])))
