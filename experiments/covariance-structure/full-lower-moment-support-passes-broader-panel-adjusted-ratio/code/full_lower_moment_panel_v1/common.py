import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
CANDIDATE = D / 'full_lower_moment_support_v1/candidate.py'
CANDIDATE_SHA = 'bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12'
PARENT_SHA = 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
NETS = (35, 100, 150, 200, 250, 300, 400, 500,
        3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
RUNTIME = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()

def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def verify_pin(item):
    assert sha(item['path']) == item['sha256'], item['path']
    if 'bytes' in item:
        assert Path(item['path']).stat().st_size == item['bytes']

def verify():
    for item in read(HERE / 'sources_sealed.json')['pins']:
        verify_pin(item)
    assert sha(CANDIDATE) == CANDIDATE_SHA

def verify_call(call):
    assert call['includes_initialization']
    assert sha(call['output']) == call['output_sha256']
    assert call['input_flops_separate'] == 0
    assert call['multiplier'] == max(.1, call['flops'] / 2**41)
    assert call['weight5_cache']['completed']
    assert call['weight5_cache']['numeric_cache_retired']

