"""Release helpers. No stage is admitted until all scientific gates pass."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
ROOT = P2.parents[3]
A = P2 / 'astra_phase2_reexamination_v1'
FIT = D / 'current_full_lower_p_fit_v1'
DOWNSTREAM = D / 'current_p_downstream_retirement_v1'
PIPELINE = D / 'current_p_downstream_retirement_pipeline_v1'
PRIOR = D / 'frozen_covariance_readout_composition_v1'
BANK = D / 'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME = A / 'openblas_runtime_env_v1/bin/python'
COLD = [35, 100, 37, 39, 42]
# These are CLI roster adapters for a byte-exact copy of the qualified
# native harness. They acquire only short-history release references.
# They do not define another independent accuracy study.
INDEPENDENT = [35, 100]
CONFIRM = [100]
PANEL = []


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
    assert pin(item['path']) == {k: item[k] for k in ('path', 'sha256', 'bytes')}, item['path']


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def clean_watch(path):
    r = read(path)
    assert r['completed'] and r['exit_code'] == 0 and r['whole_group_retired']
    assert r['max_swap_growth_bytes'] == 0 and not r.get('remaining_rows', [])
    return r


def resource_floor():
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 50_000_000


def verify():
    for p in read(HERE / 'sources_sealed.json')['pins']:
        check(p)
    resource_floor()


def admission():
    verify()
    r = read(HERE / 'admission.json')
    assert r['passed'] and r['all_fixed_scientific_gates_passed']
    assert r['candidate'] == pin(HERE / 'candidate.py')
    for p in r['pins']:
        check(p)
    return r


def stage_ready(stage):
    assert stage in ('independent', 'confirmation')
    admission()


def guarded(name, seconds, script, *args, native_guard=False):
    return external_guard(name, seconds, [RUNTIME, '-B', HERE / script, *args], native_guard=native_guard)


def external_guard(name, seconds, command, native_guard=False):
    verify()
    guard = FIT / 'guard.py' if native_guard else D / 'run_guarded.py'
    print(dict(starting=name), flush=True)
    with (HERE / f'{name}.log').open('x') as log:
        rc = subprocess.run([sys.executable, '-B', str(guard), '--record', str(HERE / f'{name}_watch.json'),
                             '--seconds', str(seconds), *map(str, command)],
                            stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == 0, (name, rc)
    r = clean_watch(HERE / f'{name}_watch.json')
    resource_floor()
    print(dict(completed=name, wall_s=r['elapsed_s'], rss=r['max_rss_bytes']), flush=True)
    return r
