"""Metadata helpers; no numerical imports or implicit worker launch."""
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
FIT = D / 'current_full_lower_p_fit_v1'
ENDPOINT = D / 'current_p_native_lifecycle_v1'
BANK = D / 'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME = A / 'openblas_runtime_env_v1/bin/python'
PANEL = [40,41,43,44,45,46,47,49,50,51,52,53,54,55]
CONFIRM = [37,39,42,48,59,63,66,67]
PUBLIC = [35,100,150,200,250,300,400,500]
VIRGIN = list(range(3016500,3016506))
INDEPENDENT = PUBLIC + VIRGIN


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def check(item):
    assert pin(item['path']) == {k: item[k] for k in ('path','sha256','bytes')}, item['path']


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def clean_watch(path):
    r = read(path)
    assert r['completed'] and r['exit_code'] == 0 and r['whole_group_retired']
    assert r['max_swap_growth_bytes'] == 0 and not r.get('remaining_rows', [])
    return r


def verify():
    for item in read(HERE / 'sources_sealed.json')['pins']:
        check(item)
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 100_000_000


def admission():
    verify()
    r = read(HERE / 'admission.json')
    assert r['passed'] and r['sole_terminal_fit']
    for item in r['pins']:
        check(item)
    return r


def stage_ready(stage):
    admission()
    if stage != 'control':
        clean_watch(HERE / 'qualify_control_watch.json')
        assert read(HERE / 'control_qualification.json')['passed']
    prior = {'confirmation':'panel', 'independent':'confirmation'}.get(stage)
    if prior:
        clean_watch(HERE / f'score_{prior}_watch.json')
        assert read(HERE / f'{prior}_score.json')['passed']


def guarded(name, seconds, script, *args):
    verify()
    print(dict(starting=name), flush=True)
    with (HERE / f'{name}.log').open('x') as log:
        rc = subprocess.run([sys.executable, '-B', str(FIT / 'guard.py'),
            '--record', str(HERE / f'{name}_watch.json'), '--seconds', str(seconds),
            str(RUNTIME), '-B', str(HERE / script), *map(str,args)],
            stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == 0, (name, rc)
    r = clean_watch(HERE / f'{name}_watch.json')
    verify()
    print(dict(completed=name, wall_s=r['elapsed_s'], rss=r['max_rss_bytes']), flush=True)
