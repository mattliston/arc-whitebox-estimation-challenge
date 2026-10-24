"""Parent-authorized fixed 259/277/287 lane; one guarded native at a time."""
from pathlib import Path
import hashlib, json, os, re, subprocess, sys, time
D=Path(__file__).resolve().parent
O=D.parents[2]
ROOT=O.parents[4]
B=D.parent/'astra_crunch_birthmetric_v1'
candidate=B/'candidate_birthmetric_v2.py'
expected='afbc69ce09232157e3e845a4d0b79ebb6a47f7bd0f3d906809ccb9801a8e7361'
runtime=O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
guard=D/'guard_memory50_v1.py'
env=dict(os.environ)
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    env[key]='1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected
records=[]
for rank in (259,277,287):
    out=B/f'res{rank}_v2.json';receipt=B/f'guard{rank}_v2.json';log=B/f'log{rank}_v2.txt'
    assert not any(p.exists() for p in (out,receipt,log))
    pressure=subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=10)
    free=int(re.search(r'System-wide memory free percentage:\s*(\d+)',pressure).group(1))
    assert free>=60,('entry memory gate',free)
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected
    command=[sys.executable,str(guard),'--record',str(receipt),'--seconds','150',
             '--footprint-limit','7500000000','--rss-backstop','7500000000',
             str(runtime),'-B',str(B/'judge_v2.py'),str(candidate),str(rank),'ctl',str(out)]
    print(json.dumps(dict(start=rank,entry_free_pct=free)),flush=True)
    with log.open('x') as stream:
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    g=json.loads(receipt.read_text())
    assert run.returncode==0 and g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and not g['remaining_rows'],g
    a=json.loads(out.read_text());c=json.loads((O/f'lead/runs/opus_kinkoracle_v1/res_ctl_{rank}.json').read_text())
    umse=100*(a['U']/c['U']-1);bill=100*(a['flops']/c['flops']-1)
    row=dict(rank=rank,delta_U_pct=umse,delta_bill_pct=bill,delta_adjusted_pct=umse+bill,
             guard_exit=g['exit_code'],whole_group_retired=True,max_rss_bytes=g['max_rss_bytes'],
             min_pressure_free_pct=g['min_pressure_free_pct'],birthmetric=a['birthmetric'])
    records.append(row)
    print(json.dumps(row),flush=True)
(D/'BIRTHMETRIC_LANE_RESULT_v1.json').open('x').write(json.dumps(dict(candidate_sha256=expected,records=records,completed=True),indent=2)+'\n')
