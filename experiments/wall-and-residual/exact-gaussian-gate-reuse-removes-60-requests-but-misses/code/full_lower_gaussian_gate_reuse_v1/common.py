import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
PARENT = D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA = 'facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
CLI = P2.parents[3]/'.venv/bin/python'
NATIVE = D/'weight5_reuse_v1/native.py'
PROBE = D/'readout_residual_runtime_v1/predict_only_probe_v1'
CONTROLS = D/'full_lower_multilayer_mean_observer_v1'

def read(p):
    return json.loads(Path(p).read_text())

def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def verify(p):
    assert pin(p['path']) == p, p['path']

def seal_check(name='native_sources_sealed.json'):
    for p in read(HERE/name)['pins']:
        verify(p)
    assert sha(PARENT) == PARENT_SHA

def guarded(label, seconds, args):
    print(dict(starting=label), flush=True)
    watch = HERE/f'{label}_watch.json'
    command = [sys.executable, '-B', str(D/'run_guarded.py'), '--record', str(watch),
               '--seconds', str(seconds), *map(str, args)]
    with (HERE/f'{label}.log').open('x') as f:
        result = subprocess.run(command, cwd=D, stdout=f, stderr=subprocess.STDOUT)
    g = read(watch)
    assert result.returncode == 0 and g['completed'] and g['whole_group_retired']
    assert g['max_swap_growth_bytes'] == 0 and g['max_rss_bytes'] < 8_000_000_000
    print(dict(completed=label, elapsed_s=g['elapsed_s'], peak_rss_bytes=g['max_rss_bytes']), flush=True)
    return g
