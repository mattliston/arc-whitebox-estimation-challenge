from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent;ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
jobs=[('prepare',LIGHT,'prepare.py',[],600),('zero',NATIVE,'run.py',['zero'],180)]
for d in (0,7):
    for amp,suffix in ((.125,'plus'),(-.125,'minus'),(.0625,'halfplus'),(-.0625,'halfminus')):
        jobs.append((f'd{d}_{suffix}',NATIVE,'run.py',[str(d),str(amp)],180))
jobs.append(('assessment',LIGHT,'assess.py',[],180))
for name,runtime,script,args,seconds in jobs:
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as f:
        r=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),'--seconds',str(seconds),str(runtime),str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=json.loads((HERE/(name+'_guard.json')).read_text())
    assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(name,g)
    print(json.dumps(dict(event='retired',name=name,peak_rss=g['max_rss_bytes'],seconds=g['elapsed_s'])),flush=True)
print('RESPONSE_QUALIFICATION_COMPLETE',flush=True)
