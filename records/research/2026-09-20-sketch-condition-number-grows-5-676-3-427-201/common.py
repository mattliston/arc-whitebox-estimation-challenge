import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
OLD = D / 'covariance_head_native_response_v1'
PARENT = D / 'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA = 'facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'

def read(p):
    return json.loads(Path(p).read_text())

def write(p, obj):
    with Path(p).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)

def pin(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def verify(p):
    assert pin(p['path']) == p, p['path']

def seal_check():
    for p in read(HERE / 'sources_sealed.json')['pins']:
        verify(p)
    assert pin(PARENT)['sha256'] == PARENT_SHA

def weights_path(rank):
    rows = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE / rows[rank]['path']).parent / 'weights.npy'

def guarded(label, seconds, args):
    print(dict(starting=label), flush=True)
    cmd = [sys.executable, '-B', str(D / 'run_guarded.py'), '--record',
           str(HERE / f'{label}_watch.json'), '--seconds', str(seconds), *map(str, args)]
    with (HERE / f'{label}.log').open('x') as stream:
        process = subprocess.run(cmd, cwd=D, stdout=stream, stderr=subprocess.STDOUT)
    result = read(HERE / f'{label}_watch.json')
    assert process.returncode == 0 and result['completed'] and result['whole_group_retired']
    assert result['max_swap_growth_bytes'] == 0 and result['max_rss_bytes'] < 8_000_000_000
    print(dict(completed=label, elapsed_s=result['elapsed_s'], peak_rss_bytes=result['max_rss_bytes']), flush=True)
    return result
