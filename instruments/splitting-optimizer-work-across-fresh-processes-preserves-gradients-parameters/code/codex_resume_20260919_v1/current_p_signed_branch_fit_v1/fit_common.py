import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2 / 'astra_phase2_reexamination_v1'
Q = D / 'current_p_signed_branch_qualification_v2'
OLD_FIT = D / 'current_full_lower_p_fit_v1'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')


def read(p):
    return json.loads(Path(p).read_text())


def pin(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''):
            h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)


def check(p):
    assert pin(p['path']) == {k:p[k] for k in ('path','sha256','bytes')},p['path']


def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False)
        f.write('\n')


def verify(data=False):
    s = read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):
        check(p)


def resource_floor():
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 1_000_000_000


def qualified_modules():
    assert 'common' not in sys.modules
    sys.path.insert(0,str(Q))
    import common
    import model_branch
    import allocator
    assert Path(common.__file__).resolve() == Q/'common.py'
    assert Path(model_branch.__file__).resolve() == Q/'model_branch.py'
    assert Path(allocator.__file__).resolve() == Q/'allocator.py'
    common.verify()
    q = read(Q/'result.json')
    assert q['completed'] and q['passed'] and q['error'] is None
    assert read(Q/'native_qualification.json')['passed'] and read(Q/'torch_qualification.json')['passed']
    return model_branch,allocator


def done(name):
    w = read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w


def guarded(name,seconds,runtime,script):
    resource_floor()
    with (HERE/f'{name}.log').open('x') as f:
        rc = subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),
            '--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),
            str(runtime),'-B',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    done(name)
    resource_floor()
