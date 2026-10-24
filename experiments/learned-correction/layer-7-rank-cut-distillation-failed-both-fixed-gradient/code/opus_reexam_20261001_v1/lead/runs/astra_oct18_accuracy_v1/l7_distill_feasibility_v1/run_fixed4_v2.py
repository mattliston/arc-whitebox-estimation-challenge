"""Fixed four teacher/student gradients; no optimizer, point, or CHECK access."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

H=Path(__file__).resolve().parent
P=H.parent/'current_torch_rtt_port_v1'
ROOT=Path('/Users/matt/arc-white-box-estimation')
plan=json.loads((H/'PLAN_v3.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',
         TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):
    env[key]='1'
rows=[]

def pincheck():
    for path,digest in plan['source_pins'].items():
        assert sha(path)==digest,path
    assert sha(plan['decision_rule'])==plan['decision_rule_sha256']

def checked(path,guard_path):
    g=json.loads(guard_path.read_text())
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0
    assert not g['error'] and not g['remaining_rows']
    assert g['min_pressure_free_pct']>=50
    assert max(g['max_rss_bytes'],g['max_footprint_bytes'])<9_000_000_000
    c=json.loads((path/'COMPLETE_v1.json').read_text())
    assert c['peak_rss']<9_000_000_000 and not c['parameters_updated'] and c['no_optimizer']
    assert sha(c['result'])==c['result_sha256']
    return c,g

def run(seed,role,path,guard,teacher=None):
    pincheck()
    command=[str(ROOT/'.venv/bin/python'),'-B',str(P/'offline_train8_v1/guard_offline9gb_free50_v1.py'),
        '--record',str(guard),'--seconds','180','--footprint-limit','9000000000',
        '--rss-backstop','9000000000','/usr/bin/nice','-n','10','/Users/matt/anaconda3/bin/python',
        '-B',str(H/'run_one_v3.py'),'--rank',str(seed),'--role',role,'--output',str(path)]
    if teacher is not None:
        command.extend(['--teacher-receipt',str(teacher/'TEACHER_v1.json')])
    print(json.dumps(dict(event='START',rank=seed,role=role)),flush=True)
    with (H/f'LOG_{role}_{seed}_v1.txt').open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,(seed,role,result.returncode)
    return checked(path,guard)

try:
    assert plan['ranks']==list(range(7100268,7100272))
    for seed in plan['ranks']:
        pincheck()
        teacher=H/('teacher268_v3' if seed==7100268 else f'teacher{seed}_v1')
        tg=H/('GUARD_teacher268_v3.json' if seed==7100268 else f'GUARD_teacher{seed}_v1.json')
        tc,tguard=run(seed,'teacher',teacher,tg)
        student=H/f'student{seed}_v1';sg=H/f'GUARD_student{seed}_v1.json'
        sc,sguard=run(seed,'student-grad',student,sg,teacher)
        result=subprocess.run(['/Users/matt/anaconda3/bin/python','-B',str(H/'inspect_student_v1.py'),
                               str(student),str(teacher)],cwd=ROOT,env=env)
        assert result.returncode==0,(seed,'result check')
        tr=json.loads((teacher/'TEACHER_v1.json').read_text())
        row=dict(rank=seed,result=sc['result'],result_sha256=sc['result_sha256'],
                 teacher=str(teacher/'TEACHER_v1.json'),teacher_sha256=sha(teacher/'TEACHER_v1.json'),
                 teacher_loss=sc['distillation_mse'],true_mc_mse=sc['true_mc_mse'],
                 teacher_true_mc_mse=tc['true_mc_mse'],guard=str(sg),guard_sha256=sha(sg),
                 teacher_guard=str(tg),real_highwater=sc['peak_rss'],seconds=sc['seconds'],
                 data_pins=tr['data_pins'],prefix_L0_6_byte_exact=True)
        rows.append(row)
        with (H/f'RETIRED_{seed}_v1.json').open('x') as f:json.dump(row,f,indent=2)
        print(json.dumps(dict(event='GRADIENT_RETIRED',**row)),flush=True)
    pincheck()
    with (H/'GRADIENT_MANIFEST_v1.json').open('x') as f:
        json.dump(dict(status='ALL_FOUR_DISTILLATION_GRADIENTS_COMPLETE_AND_RETIRED',
                       rows=rows,source_pins=plan['source_pins'],parameters_updated=False,
                       no_optimizer=True,decision_rule=plan['decision_rule'],
                       decision_rule_sha256=plan['decision_rule_sha256']),f,indent=2)
    print('ALL_FOUR_COMPLETE',flush=True)
except BaseException as error:
    with (H/'STOP_v1.json').open('x') as f:
        json.dump(dict(error=repr(error),completed_ranks=[r['rank'] for r in rows],
                       parameters_updated=False),f,indent=2)
    raise
