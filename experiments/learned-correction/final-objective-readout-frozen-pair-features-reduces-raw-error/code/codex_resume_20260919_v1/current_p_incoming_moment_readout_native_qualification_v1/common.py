import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
SMALL=D/'current_p_incoming_moment_readout_design_v1'
PARENT=D/'current_p_message_covariance_release_v1/candidate.py'
PARENT_SHA='d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'

def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(data=False):
    s=read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):check(p)
    assert pin(PARENT)['sha256']==PARENT_SHA
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def process_rows():
    return subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,command='],text=True,timeout=5).splitlines()
def numerical_workers():
    exes={str(RUNTIME),'/Users/matt/anaconda3/bin/python',str(D.parent/'data_gate/.client-venv/bin/python')}
    return [r for r in process_rows() if len(a:=r.strip().split(None,3))==4 and a[3].split()[0] in exes]
def admission():
    verify();b=read(HERE/'build.json')
    assert b==read(SMALL/'build.json')
    assert b['completed'] and b['whole_source_inverses_exact'] and b['helper_AST_rename_inverse']
    assert b['parent']==pin(PARENT) and b['trainable_values']==96 and not b['fit_initialized']
    assert b['positive_zero_bytes'] and b['all_parent_values_frozen']
    assert b['public_hidden_and_normalizer_values_byte_exact']
    for k in ('disabled','zero','sentinel','fixed_model','output_scale'):check(b[k])
    result=read(SMALL/'result.json');unit=read(SMALL/'unit_result.json');watch=read(SMALL/'unit_watch.json')
    assert result['completed'] and result['passed'] and result['error'] is None and result['small_only']
    assert unit['passed'] and unit['small_corrections']==16 and unit['small_feature_calls']==8 and unit['full_network_calls']==0
    assert unit['readout_values']==96 and 0<unit['conservative_seven_layer_flops']<=12_000_000_000
    assert not unit['network_arrays_opened'] and not unit['target_arrays_opened'] and not unit['fit_initialized']
    assert watch['completed'] and watch['exit_code']==0 and watch['error'] is None
    assert watch['rss_trigger_bytes']==1_000_000_000 and watch['max_rss_bytes']<1_000_000_000
    assert watch['max_swap_growth_bytes']==0 and watch['whole_group_retired'] and not watch['remaining_rows']
    receipt=read(SMALL/'orchestration_receipt.json')
    assert receipt['tool_terminal_observed'] and receipt['tool_terminal_exit_code']==0
    assert receipt['worker_group_retired'] and not receipt['controller_run_claim_used']
    assert receipt['worker_pid']==watch['pid']==43025 and receipt['tool_session_id']==70779
    rows=[r.strip().split(None,3) for r in process_rows()];rows=[r for r in rows if len(r)==4]
    assert not [r for r in rows if int(r[0])==watch['pid'] or int(r[2])==watch['pid']]
    return b
def done(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def stage_ready(stage):
    assert stage=='control';admission()
def guarded(name,seconds,script,*args):
    verify();assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=done(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
