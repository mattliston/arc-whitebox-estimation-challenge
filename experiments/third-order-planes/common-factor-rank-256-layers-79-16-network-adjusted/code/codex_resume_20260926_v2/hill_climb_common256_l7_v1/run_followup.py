from pathlib import Path
import hashlib,json,subprocess
H=Path(__file__).resolve().parent.parent
S=Path(__file__).resolve().parent
P2=H.parent
ROOT=P2.parents[3]
LIGHT=ROOT/'.venv/bin/python'
NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
p=json.loads((S/'train16_extension_protocol.json').read_text())
for name,ranks,seconds in [('train6',p['additional_blocks'][0],300),('extra8',p['additional_blocks'][1],360)]:
    c=p['candidate'];assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
    print(json.dumps(dict(event='starting',block=name)),flush=True)
    with (S/(name+'.log')).open('x') as f:
        run=subprocess.run([str(LIGHT),'-B',str(GUARD),'--record',str(S/(name+'_guard.json')),'--seconds',str(seconds),'--rss-limit','7500000000',str(NATIVE),'-B',str(H/'native_eval.py'),c['path'],','.join(map(str,ranks)),'hill_climb_common256_l7_'+name+'_v1'],stdout=f,stderr=subprocess.STDOUT)
    g=json.loads((S/(name+'_guard.json')).read_text())
    assert run.returncode==0 and g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['error'] is None and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7500000000
    print(json.dumps(dict(event='completed',block=name,seconds=g['elapsed_s'],rss=g['max_rss_bytes'])),flush=True)
print('FIXED_FOLLOWUP_COMPLETE',flush=True)
