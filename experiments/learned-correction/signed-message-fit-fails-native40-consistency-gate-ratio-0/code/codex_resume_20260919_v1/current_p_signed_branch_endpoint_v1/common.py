import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
ROOT = D.parents[4]
A = P2/'astra_phase2_reexamination_v1'
F = D/'current_p_signed_branch_fit_v1'
Q = D/'current_p_signed_branch_qualification_v2'
PARENT = D/'current_p_release_v1/candidate.py'
ZERO = Q/'candidate_zero.py'
BANK = ZERO  # Legacy serializer name; the scientific comparison uses PARENT.
NATIVE_RUNTIME = A/'openblas_runtime_env_v1/bin/python'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')
PKEYS = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain','ungated_w')
INACTIVE = ('head','head_b','gain')


def read(p):
    return json.loads(Path(p).read_text())


def pin(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20),b''):
            h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)


def check(p):
    assert pin(p['path']) == {k:p[k] for k in ('path','sha256','bytes')},p['path']


def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False)
        f.write('\n')


def verify():
    for p in read(HERE/'sources_sealed.json')['pins']:
        check(p)


def processes():
    result = []
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5).splitlines():
        s = line.strip().split(None,2)
        if len(s)==3:
            result.append(dict(pid=int(s[0]),pgid=int(s[1]),command=s[2]))
    return result


def numerical_workers():
    names = {str(NATIVE_RUNTIME),str(RUNTIME),str(P2/'data_gate/.client-venv/bin/python')}
    return [r for r in processes() if r['command'].split(None,1)[0] in names]


def fit_success():
    verify()
    result = read(F/'controller_result.json')
    assert result['completed'] and result['passed'] and result['error'] is None
    fit = read(F/'fit_result.json')
    assert fit['completed'] and fit['passed'] and fit['training_graphs']==160 and fit['updates']==40 and fit['epoch']==1
    assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
    assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
    assert 'ungated_w' in fit['changed_trainable_keys']
    watch = done('fit')
    fit_scripts = {(F/name).resolve() for name in ['run_fit.py','guard.py','train.py']}
    for process in processes():
        paths = [Path(x) for x in process['command'].split() if x.endswith(('/run_fit.py','/guard.py','/train.py'))]
        assert not any((p if p.is_absolute() else ROOT/p).resolve() in fit_scripts for p in paths), process
    assert fit['source_seal']==pin(F/'sources_sealed.json')
    for key in ['terminal_payload','training_order','source_seal']:
        check(fit[key])
    complete = read(F/'fit_complete.json')
    assert complete['completed'] and complete['result']==pin(F/'fit_result.json') and complete['watch']==pin(F/'fit_watch.json')
    return fit


def clean_watch(p):
    w = read(p)
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w


def done(name):
    return clean_watch((F if name=='fit' else HERE)/f'{name}_watch.json')


def admission():
    fit_success()
    b = read(HERE/'build.json')
    assert b['passed'] and b['only_gnn_literal_changed']
    assert b['parent']==pin(PARENT) and b['initial_source']==pin(ZERO)
    for key in ['candidate','null_candidate','parent','initial_source']:
        check(b[key])
    return b


def stage_ready(stage):
    assert stage in ('control','validation')
    admission()
    done('build')
    if stage=='validation':
        done('qualify_null')
        assert read(HERE/'native_null_qualification.json')['passed']


def qualify_retirements(name,records):
    pins = []
    for i,r in enumerate(records):
        path = HERE/name/f'call_{i:02d}_retirement.json'
        x = read(path)
        assert x['complete'] and x['index']==i and x['rank']==r['rank']
        assert x['all_tracked_call_objects_released'] and x['tracked_objects']==18 and not x['live_objects']
        assert x['after_closed_saved_bill'] and x['saved_bill_unchanged'] and x['allocator_called']
        assert not x['allocator_statistics_called'] and not x['candidate_state_reset']
        check(x['saved_bill'])
        assert x['saved_bill']==r['bill']
        pins.append(pin(path))
    return pins


def resource_floor():
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000


def guarded(name,seconds,script,*args):
    resource_floor()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc = subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),
            '--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),
            str(NATIVE_RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w = done(name)
    resource_floor()
    print(dict(completed=name,elapsed_s=w['elapsed_s'],peak_rss_bytes=w['max_rss_bytes']),flush=True)
