"""Reuse the original native engine and genuine S34 controls; N2 only, stdlib only."""
from pathlib import Path
import copy
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
T = H / 'hill_climb_s34_ynt_zonal3_v1'

def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def rewrite(value):
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, str):
        return value.replace(T.name, D.name)
    return value

template = T / 'native_screen.json'
assert pin(template)['sha256'] == '442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
base_config = json.loads(template.read_text())
build = json.loads((D / 'build.json').read_text())
assert build['candidate'] == pin(D / 'candidate.py')
assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert build['parent'] == pin(build['parent']['path'])
assert build['source_inverse_exact'] and build['frozen_trunk_bytes_exact']
assert build['optimizer_updates'] == 4
assert build['protocol'] == pin(D / 'protocol.json')
assert build['training_guard'] == pin(D / 'training_guard.json')
primitive = json.loads((D / 'primitive_result.json').read_text())
assert primitive['passed'] and primitive['candidate'] == build['candidate']
assert primitive['build'] == pin(D / 'build.json') and primitive['probe'] == pin(D / 'probe.py')
assert primitive['frozen_trunk_exact'] and primitive['source_inverse_exact']
guard = json.loads((D / 'primitive_guard.json').read_text())
assert guard['completed'] and guard['whole_group_retired'] and guard['exit_code'] == 0
assert guard['error'] is None and guard['remaining_rows'] == [] and guard['max_swap_growth_bytes'] == 0
assert 0 < guard['elapsed_s'] <= 150 and 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] <= 7_500_000_000
assert guard['command'][1:] == ['-B', str(D / 'probe.py'), pin(D / 'build.json')['sha256'], str(D / 'primitive_result.json')]
config = rewrite(copy.deepcopy(base_config))
config['arms']['candidate']['source'] = build['candidate']
assert config['arms']['base'] == base_config['arms']['base']
assert config['arms']['parity_parent'] == base_config['arms']['parity_parent']
assert config['arms']['base']['source'] == build['parent']
config['parity_layers'] = [0]
config['provenance'] = [pin(path) for path in (
    template, D / 'build_native2_config.py', D / 'probe.py', D / 'build.json',
    D / 'protocol.json', D / 'training/completed.json', D / 'training_guard.json',
    D / 'primitive_result.json', D / 'primitive_guard.json',
    H / 'official_bank_S34_v1.json')]
config['interpretation'] = (
    'One fixed S34 native-anchored main/dial fit, four updates on sixteen disjoint TRAIN; '
    'frozen trunk and one endpoint. N2 <= -0.1%, at least one win, individual regression <= 1%; '
    'L0 exact, all layers finite. Fixed-eight gates remain unchanged. '
    'Only N2 is admitted; no boundary, additional-six, broad or serving authority.')
assert config['prospective_scope'] == 'native2_only'
for key in ('2', '8'):
    assert config['stages'][key]['gates'] == base_config['stages'][key]['gates']
with (D / 'native_screen.json').open('x') as stream:
    json.dump(config, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(native_config=pin(D / 'native_screen.json'), scope='native2_only')))
