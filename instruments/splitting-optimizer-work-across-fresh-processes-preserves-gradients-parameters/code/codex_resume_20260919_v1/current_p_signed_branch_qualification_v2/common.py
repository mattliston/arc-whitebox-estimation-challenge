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
RELEASE = D / 'current_p_release_v1'
PARENT = RELEASE / 'candidate.py'
SOURCE_PARENT_SHA = '8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
SOURCE = HERE / 'candidate_zero.py'
SHARED = A / 'current_shared384_causal_p_v1/shared_operator.py'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')
NATIVE_RUNTIME = A / 'openblas_runtime_env_v1/bin/python'
BASE_PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
              'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
PKEYS = BASE_PKEYS + ('ungated_w',)
INACTIVE = ('head', 'head_b', 'gain')
INDEPENDENT = [35, 100]
PANEL = []
CONFIRM = [100]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pin(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def check(p):
    assert pin(p['path']) == {k:p[k] for k in ('path', 'sha256', 'bytes')}, p['path']


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify():
    for p in read(HERE / 'sources_sealed.json')['pins']:
        check(p)


def admission():
    verify()
    b = read(HERE / 'build.json')
    assert b['whole_source_inverse_exact'] and b['only_gnn_and_literal_changed']
    assert b['parent'] == pin(PARENT) and b['candidate'] == pin(SOURCE)
    return dict(bank=b['parent'], candidate=b['candidate'])


def stage_ready(stage):
    assert stage in ('independent', 'confirmation')
    assert read(HERE / 'unit_native_result.json')['passed']
    assert read(HERE / 'unit_torch_result.json')['passed']


def clean_watch(p):
    w = read(p)
    assert w['completed'] and w['exit_code'] == 0 and w['whole_group_retired']
    assert w['max_swap_growth_bytes'] == 0 and w['max_rss_bytes'] < 7_500_000_000
    return w


def resource_floor():
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 100_000_000


def guarded(name, seconds, runtime, script, *args):
    resource_floor()
    print(dict(starting=name), flush=True)
    with (HERE / f'{name}.log').open('x') as log:
        rc = subprocess.run([sys.executable, '-B', str(HERE / 'guard.py'),
            '--record', str(HERE / f'{name}_watch.json'), '--seconds', str(seconds),
            str(runtime), '-B', str(HERE / script), *map(str, args)],
            stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == 0, (name, rc)
    w = clean_watch(HERE / f'{name}_watch.json')
    resource_floor()
    print(dict(completed=name, wall_s=w['elapsed_s'], peak_rss_bytes=w['max_rss_bytes']), flush=True)


# The original inert model expects this binding to name its source hash.
PARENT_SHA = read(HERE / 'build.json')['candidate']['sha256'] if (HERE / 'build.json').exists() else None
