import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
Q=D/'current_p_incoming_moment_readout_training_qualification_v1'
RUNTIME=Path('/Users/matt/anaconda3/bin/python')
EKEYS=('joint_theta',)
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
    s=read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):check(p)
    resource_floor()
def qualified_modules():
    assert 'common' not in sys.modules
    sys.path.insert(0,str(Q))
    import common
    import model_readout
    import allocator
    for m,n in ((common,'common.py'),(model_readout,'model_readout.py'),(allocator,'allocator.py')):
        assert Path(m.__file__).resolve()==Q/n
    common.verify()
    assert read(Q/'result.json')['passed'] and read(Q/'full_result.json')['passed']
    return model_readout,allocator
def numerical_workers():
    exes={str(RUNTIME),str(A/'openblas_runtime_env_v1/bin/python'),str(D.parent/'data_gate/.client-venv/bin/python')}
    rows=[]
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5).splitlines():
        f=line.strip().split(None,2)
        if len(f)==3 and f[2].split()[0] in exes:rows.append(line)
    return rows
def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def guarded(name,seconds,script,*args):
    verify();assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name);assert not numerical_workers();resource_floor()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
