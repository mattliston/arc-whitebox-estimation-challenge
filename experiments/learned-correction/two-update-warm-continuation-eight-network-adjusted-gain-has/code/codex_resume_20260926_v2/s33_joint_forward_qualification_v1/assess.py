"""Assess one prescribed proposal-response direction; no fit admission."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
protocol_path = HERE / 'protocol.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(protocol_path) == sys.argv[2]
protocol = json.loads(protocol_path.read_text())
assert protocol['root_admitted'] and not protocol['fitting_admitted']
rank = int(sys.argv[1]); assert rank in (35, 100)
for pin in protocol['pins']:
    assert sha(pin['path']) == pin['sha256']
import numpy as np

def rms(v): return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))
def compare(proposal, native):
    a, b = proposal.reshape(-1), native.reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {'cosine': float(np.dot(a, b) / den) if den else 0.,
            'relative_rms_error': rms(a - b) / max(rms(b), 1e-30),
            'proposal_rms': rms(a), 'native_rms': rms(b)}
def aligned(r):
    return r['cosine'] >= .8 and r['relative_rms_error'] <= .5

guards = {}
for kind in ('native', 'torch'):
    p = HERE / (kind + '_rank' + str(rank) + '_guard.json')
    g = json.loads(p.read_text())
    assert g['completed'] and g['exit_code'] == 0 and g['whole_group_retired']
    assert g['max_swap_growth_bytes'] == 0 and g['max_rss_bytes'] < 7500000000
    guards[kind] = {'sha256': sha(p), 'elapsed_s': g['elapsed_s'], 'rss': g['max_rss_bytes']}
n_dir = HERE / ('native_rank' + str(rank)); t_dir = HERE / ('torch_rank' + str(rank))
assert json.loads((n_dir / 'completed.json').read_text())['completed']
assert json.loads((t_dir / 'completion.json').read_text())['completed']
nm = json.loads((n_dir / 'manifest.json').read_text())
tm = json.loads((t_dir / 'manifest.json').read_text())
assert nm['protocol_sha256'] == tm['protocol_sha256'] == sha(protocol_path)
assert nm['direction_metadata'] == tm['direction']
assert nm['weights']['sha256'] == tm['weights_sha256']
with np.load(n_dir / 'predictions.npz', allow_pickle=False) as z:
    native = {key: z[key].copy() for key in z.files}
torch_values, tangent = {}, None
for key in ('zero', 'plus_h', 'minus_h', 'plus_half', 'minus_half'):
    with np.load(t_dir / (key + '.npz'), allow_pickle=False) as z:
        torch_values[key] = z['prediction'].copy()
        if key == 'zero': tangent = z['tangent'].copy()
    assert native[key].shape == torch_values[key].shape == (16, 1024)
    assert np.isfinite(native[key]).all() and np.isfinite(torch_values[key]).all()
assert np.isfinite(tangent).all()
saved = np.load(protocol['saved_native_predictions'][str(rank)]['path'], allow_pickle=False)
zero_exact = native['zero'].tobytes() == saved.tobytes()
nd = [(native['plus_h'] - native['minus_h']) / .02,
      (native['plus_half'] - native['minus_half']) / .01]
td = [(torch_values['plus_h'] - torch_values['minus_h']) / .02,
      (torch_values['plus_half'] - torch_values['minus_half']) / .01]
scopes = {'final': slice(15, 16), 'feedback': slice(1, 15)}
metrics = {}
for name, sl in scopes.items():
    metrics[name] = {'native_torch_h': compare(td[0][sl], nd[0][sl]),
                     'native_torch_half': compare(td[1][sl], nd[1][sl]),
                     'jvp_native_half': compare(tangent[sl], nd[1][sl]),
                     'native_step_stability': compare(nd[0][sl], nd[1][sl]),
                     'torch_step_stability': compare(td[0][sl], td[1][sl])}
response_rms = [rms((native['plus_half'] - native['minus_half'])[l] / 2) for l in range(16)]
signal = response_rms[15] >= 1e-8 and sum(response_rms[l] >= 1e-8 for l in (4, 8, 12)) >= 2
passed = bool(zero_exact and signal and all(aligned(item) for group in metrics.values() for item in group.values()))
record = {'rank': rank, 'passed': passed, 'native_zero_all16_exact': zero_exact,
          'response_signal_passed': signal, 'native_half_step_response_rms_by_layer': response_rms,
          'baseline_difference_rms_by_layer': [rms(torch_values['zero'][l] - saved[l]) for l in range(16)],
          'metrics': metrics, 'guards': guards, 'protocol_sha256': sha(protocol_path),
          'targets_accessed': False, 'fitting_admitted': False, 'full_jacobian_certified': False,
          'meaning': 'A pass supports a separately bounded proposal fit only; no score or native-gradient claim.'}
with (HERE / ('assessment_rank' + str(rank) + '.json')).open('x') as fp:
    json.dump(record, fp, indent=2, allow_nan=False)
print(json.dumps(record, allow_nan=False), flush=True)
