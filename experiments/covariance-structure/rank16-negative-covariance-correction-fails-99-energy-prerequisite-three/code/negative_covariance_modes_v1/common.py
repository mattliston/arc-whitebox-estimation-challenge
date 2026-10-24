import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2 / 'astra_phase2_reexamination_v1'
RUNTIME = A / 'openblas_runtime_env_v1/bin/python'
PARENT = D / 'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA = 'facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
CAPTURE_RESULT = D / 'full_lower_covariance_innovation_supervision_v1/qualification/result.json'
RANKS = (35, 100)
LAYERS = tuple(range(8, 15))


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def verify(name='sources_sealed.json'):
    x = read(HERE / name)
    for p in x['pins']:
        assert sha(p['path']) == p['sha256'], p['path']
    return x


def captures():
    return read(CAPTURE_RESULT)['records'][:2]


def truth(rank):
    return P2 / f'public_g19_attribution_v1/mps_cov2_rank{rank:04d}.npz'


def guarded(name, seconds, script):
    with (HERE / f'{name}.log').open('x') as f:
        code = subprocess.run([sys.executable, str(D / 'run_guarded.py'), '--record',
            str(HERE / f'{name}_watch.json'), '--seconds', str(seconds),
            str(RUNTIME), '-B', '-W', 'ignore', str(HERE / script)], stdout=f, stderr=subprocess.STDOUT).returncode
    r = read(HERE / f'{name}_watch.json')
    assert code == 0 and r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes'] == 0, (name, r)
    print(json.dumps(dict(stage=name, completed=True, rss=r['max_rss_bytes'], wall_s=r['elapsed_s'])), flush=True)
