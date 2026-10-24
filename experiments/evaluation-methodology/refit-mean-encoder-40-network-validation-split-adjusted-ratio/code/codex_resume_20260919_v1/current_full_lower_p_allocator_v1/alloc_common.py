import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
OLD = D / 'current_full_lower_p_proposal_v1'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')


def read(p):
    return json.loads(Path(p).read_text())


def pin(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), sha256=h.hexdigest(), bytes=p.stat().st_size)


def check(p):
    assert pin(p['path']) == {k: p[k] for k in ('path', 'sha256', 'bytes')}, p['path']


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify():
    for p in read(HERE / 'sources_sealed.json')['pins']:
        check(p)


def resource_floor():
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= 1_000_000_000


def original_model():
    assert 'common' not in sys.modules
    sys.path.insert(0, str(OLD))
    import model_v2
    import common
    assert Path(common.__file__).resolve() == OLD / 'common.py'
    assert Path(model_v2.__file__).resolve() == OLD / 'model_v2.py'
    common.verify()
    common.verify('repair_v2_sealed.json')
    return model_v2


def guarded(name, seconds, script, *args):
    resource_floor()
    print(dict(starting=name), flush=True)
    with (HERE / f'{name}.log').open('x') as f:
        rc = subprocess.run([sys.executable, '-B', str(HERE / 'guard.py'),
            '--record', str(HERE / f'{name}_watch.json'), '--seconds', str(seconds),
            str(RUNTIME), '-B', str(HERE / script), *map(str, args)],
            stdout=f, stderr=subprocess.STDOUT).returncode
    r = read(HERE / f'{name}_watch.json')
    assert rc == 0 and r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes'] == 0, (name, r)
    resource_floor()
    print(dict(completed=name, wall_s=r['elapsed_s'], peak_rss=r['max_rss_bytes']), flush=True)
