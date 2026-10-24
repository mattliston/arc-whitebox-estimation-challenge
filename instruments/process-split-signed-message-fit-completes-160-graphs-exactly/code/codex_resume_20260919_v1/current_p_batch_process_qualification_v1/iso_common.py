import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2/'astra_phase2_reexamination_v1'
FIT = D/'current_p_signed_branch_fit_v1'
Q = D/'current_p_signed_branch_qualification_v2'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')


def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def resource_floor():
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000


def verify(data=False):
    s=read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):check(p)
    resource_floor()


def qualified_modules():
    assert 'common' not in sys.modules
    sys.path.insert(0,str(Q))
    import common
    import model_branch
    import allocator
    assert Path(common.__file__).resolve()==Q/'common.py'
    assert Path(model_branch.__file__).resolve()==Q/'model_branch.py'
    assert Path(allocator.__file__).resolve()==Q/'allocator.py'
    common.verify()
    q=read(Q/'result.json')
    assert q['completed'] and q['passed'] and q['error'] is None
    assert read(Q/'native_qualification.json')['passed'] and read(Q/'torch_qualification.json')['passed']
    return model_branch,allocator


def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w


def numerical_workers():
    exes={str(RUNTIME),str(A/'openblas_runtime_env_v1/bin/python'),str(P2/'data_gate/.client-venv/bin/python')}
    result=[]
    for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5).splitlines():
        a=line.strip().split(None,2)
        if len(a)==3 and a[2].split()[0] in exes:result.append(line)
    return result


def guarded(name,seconds,script,*args):
    verify()
    assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name)
    assert not numerical_workers()
    resource_floor()
    print(dict(completed=name,wall_s=w['elapsed_s'],peak_rss_bytes=w['max_rss_bytes']),flush=True)
