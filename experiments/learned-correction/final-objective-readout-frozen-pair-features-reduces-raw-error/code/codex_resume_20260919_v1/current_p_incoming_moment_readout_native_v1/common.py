import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
F=D/'current_p_incoming_moment_readout_fit_v1'
QN=D/'current_p_incoming_moment_readout_native_qualification_v1'
SMALL=D/'current_p_incoming_moment_readout_design_v1'
ZERO=SMALL/'zero.py'
DISABLED=SMALL/'disabled.py'
PARENT=D/'current_p_message_covariance_release_v1/candidate.py'
PARENT_SHA='d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
EKEYS=('joint_theta',)
EXPORT_KEYS=('output',)
JKEYS=('left','right','edge','bias','feature_scale','rho_scale','third_scale','target_scale')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def resource_floor():
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<1_000_000_000
def verify(data=False):
    seal=read(HERE/'sources_sealed.json')
    for p in seal['pins']+(seal['data_pins'] if data else []):check(p)
    assert pin(PARENT)['sha256']==PARENT_SHA
    resource_floor()
def process_rows():
    return subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,command='],text=True,timeout=5).splitlines()
def numerical_workers():
    exes={str(RUNTIME),'/Users/matt/anaconda3/bin/python',str(D.parent/'data_gate/.client-venv/bin/python')}
    return [line for line in process_rows() if len(line.strip().split(None,3))==4 and line.strip().split(None,3)[3].split()[0] in exes]
def fit_success():
    verify()
    result=read(F/'controller_result.json')
    assert result['completed'] and result['passed'] and result['error'] is None
    check(result['fit']);check(result['sources'])
    fit=read(F/'fit_result.json')
    assert fit['completed'] and fit['passed'] and fit['training_graphs']==320 and fit['updates']==80
    assert fit['readout_trainable_values']==96 and fit['fixed_parameters_unchanged']
    assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
    check(fit['terminal_payload']);check(fit['source_seal'])
    for name in [f'batch_{i:02d}' for i in range(80)]+['verify_fit']:
        w=read(F/f'{name}_watch.json')
        assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
        assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    controller_pid=read(HERE/'inputs.json')['fit_controller_pid_at_preparation']
    owned=[]
    for line in process_rows():
        fields=line.strip().split(None,3)
        if len(fields)!=4:continue
        if F.name+'/' in fields[3] or int(fields[0])==controller_pid or int(fields[2])==controller_pid:owned.append(line)
    assert not owned,owned
    return fit
def admission():
    fit=fit_success();b=read(HERE/'build.json')
    assert b['completed'] and b['whole_source_inverses_exact'] and b['only_joint_output_value_changed']
    assert b['parent']==pin(PARENT) and b['terminal_payload']==fit['terminal_payload']
    assert b['zero_source']==pin(ZERO) and b['disabled_source']==pin(DISABLED)
    assert b['native_qualification']==pin(QN/'qualification.json')
    for key in ('parent','candidate','null_candidate'):check(b[key])
    return b
def done(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w
def stage_ready(stage):
    assert stage in ('control','validation')
    admission();done('build')
    if stage=='validation':
        done('qualify_controls');assert read(HERE/'control_qualification.json')['passed']
def guarded(name,seconds,script,*args):
    verify();assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=done(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
