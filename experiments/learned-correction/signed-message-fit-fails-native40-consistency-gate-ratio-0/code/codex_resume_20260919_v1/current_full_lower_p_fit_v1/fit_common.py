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
ALLOC = D / 'current_full_lower_p_allocator_v1'
OLD = D / 'current_full_lower_p_proposal_v1'
BANK = D / 'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME = Path('/Users/matt/anaconda3/bin/python')
NATIVE_RUNTIME = A / 'openblas_runtime_env_v1/bin/python'
DATA = A / 'joint_prefix_final_data_v1.py'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRAIN_INDEX = A / 'joint_prefix_final_fit_v1/manifest.json'
VALID_INDEX = A / 'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH = P2 / 'public_g22_operator_features_v1/truthpost'


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


def verify(data=False):
    seal = read(HERE / 'sources_sealed.json')
    for p in seal['pins'] + (seal['data_pins'] if data else []):
        check(p)


def resource_floor():
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= 1_000_000_000


def qualified_modules():
    sys.path.insert(0, str(ALLOC))
    import alloc_common
    import allocator
    assert Path(alloc_common.__file__).resolve() == ALLOC / 'alloc_common.py'
    assert Path(allocator.__file__).resolve() == ALLOC / 'allocator.py'
    alloc_common.verify()
    return alloc_common.original_model(), allocator


def done(name):
    r = read(HERE / f'{name}_watch.json')
    assert r['completed'] and r['exit_code'] == 0 and r['whole_group_retired'] and r['max_swap_growth_bytes'] == 0
    return r


def guarded(name, seconds, runtime, script, *args):
    resource_floor()
    print(dict(starting=name), flush=True)
    with (HERE / f'{name}.log').open('x') as f:
        rc = subprocess.run([sys.executable, '-B', str(HERE / 'guard.py'),
            '--record', str(HERE / f'{name}_watch.json'), '--seconds', str(seconds),
            str(runtime), '-B', str(HERE / script), *map(str, args)],
            stdout=f, stderr=subprocess.STDOUT).returncode
    assert rc == 0, (name, rc)
    r = done(name)
    resource_floor()
    print(dict(completed=name, wall_s=r['elapsed_s'], peak_rss=r['max_rss_bytes']), flush=True)
