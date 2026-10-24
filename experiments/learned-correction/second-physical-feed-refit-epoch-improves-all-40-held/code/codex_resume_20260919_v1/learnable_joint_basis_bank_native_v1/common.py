import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
SMALL=D/'learnable_joint_basis_implementation_v1'
FIXED=D/'joint_remainder_bank_native_qualification_v1'
UNIT=D/'learnable_joint_basis_native_component_v1'
PARENT=D/'current_p_message_covariance_release_v1/candidate.py'
PARENT_SHA='d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
K=D/'joint_remainder_native_component_v1';L=D/'joint_remainder_carrier_native_v1'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(data=False):
    s=read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):check(p)
    assert pin(PARENT)['sha256']==PARENT_SHA
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def process_rows():return subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,command='],text=True,timeout=5).splitlines()
def numerical_workers():
    exes={str(RUNTIME),'/Users/matt/anaconda3/bin/python',str(D.parent/'data_gate/.client-venv/bin/python')}
    return [r for r in process_rows() if len(a:=r.strip().split(None,3))==4 and a[3].split()[0] in exes]
def prior_retired():
    rows=[r.strip().split(None,3) for r in process_rows()];rows=[r for r in rows if len(r)==4]
    for folder in (FIXED,UNIT):
        claim=read(folder/'run_claim.json')
        assert not [r for r in rows if int(r[0])==claim['pid'] or int(r[2])==claim['pgid'] or folder.name+'/' in r[3]],str(folder)
def admission():
    verify();prior_retired()
    b=read(SMALL/'build.json')
    assert b['completed'] and b['total_parameters']==3169 and b['additional_parameters']==3072
    assert b['source_only'] and not b['bank_imported'] and b['helper_inverse_exact']
    assert all(r['whole_source_inverse_exact'] and r['old97_parameters_byte_exact'] and r['original_bank_literals_unchanged'] for r in b['roles'])
    bank=read(D/'official_best_331824.json');assert bank['candidate']==pin(PARENT) and bank['completed']==100 and bank['failed']==0
    f=read(FIXED/'closeout.json');assert f['passed'] and f['controller_and_all_groups_retired'] and f['full_native_calls']==12
    u=read(UNIT/'closeout.json');assert u['passed'] and u['controller_and_group_retired'] and u['tool_exit_code']==0
    q=read(UNIT/'qualification.json');assert q['passed'] and q['component_calls']==9 and q['full_network_calls']==0
    roles={x['role']:x['source'] for x in b['roles']}
    assert set(roles)=={'zero','disabled','sentinel','identity_sentinel','basis_zero','physical_only','head_only'}
    for p in roles.values():check(p)
    return roles
def stage_ready(stage):assert stage=='control';admission()
def done(name):
    w=read(HERE/(name+'_watch.json'))
    assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def guarded(name,seconds,script,*args):
    verify();assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/(name+'.log')).open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/(name+'_watch.json')),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=done(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
