from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent;ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
def run(name,script,args,runtime=NATIVE):
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as f:
        r=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),'--seconds','180',str(runtime),str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=json.loads((HERE/(name+'_guard.json')).read_text())
    assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(name,g)
    print(json.dumps(dict(event='retired',name=name,peak_rss=g['max_rss_bytes'],seconds=g['elapsed_s'])),flush=True)
for name in ('disabled','zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus'):run(name,'run.py',[name])
run('assess_d0','assess.py',['d0'],LIGHT)
if json.loads((HERE/'direction0_result.json').read_text())['passed']:
    for suffix in ('plus','minus','halfplus','halfminus'):
        name='d7_'+suffix;run(name,'run.py',[name])
    run('assess_all','assess.py',['all'],LIGHT)
print('COVARIANCE_RESPONSE_COMPLETE',flush=True)
