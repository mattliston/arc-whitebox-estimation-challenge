from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent;ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
jobs=[('build_mm6',LIGHT,'build_mm6.py',[]),('unit_mm6',NATIVE,'unit_mm6.py',[]),
    ('disabled_mm6',NATIVE,'evaluate_mm6.py',[str(HERE/'disabled_mm6.py'),'35,100','global_pair_exact_disabled_mm6_train2_v1']),
    ('disabled_check_mm6',LIGHT,'assess_mm6.py',['disabled'])]
for name,runtime,script,args in jobs:
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as f:
        r=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),'--seconds','180',str(runtime),str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=json.loads((HERE/(name+'_guard.json')).read_text())
    assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(name,g)
    print(json.dumps(dict(event='retired',name=name,peak_rss=g['max_rss_bytes'],seconds=g['elapsed_s'])),flush=True)
print('NATIVE_HELPER_AND_DISABLED_COMPLETE_USE_GENERIC_SCREEN_NEXT',flush=True)
