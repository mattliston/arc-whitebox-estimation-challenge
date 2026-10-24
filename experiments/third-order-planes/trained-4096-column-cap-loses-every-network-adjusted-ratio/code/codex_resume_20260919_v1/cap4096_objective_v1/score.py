"""Score one frozen formerly-unscored native candidate; never import its estimator."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import math
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2 / 'astra_phase2_reexamination_v1'
CAP = A / 'current_paircap4096_e2e_v1'
BD = A / 'birth_direction_fit_v1'
LAZY = D / 'lazy_rotation_objective_v1'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TARGETS = {r: P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz' for r in VALID}
TRUTH_INDEX = A / 'joint_prefix_final_judge_cached_v2/receipt.json'
read = lambda p: json.loads(Path(p).read_text())
pins = {}
started = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected=None):
    p = Path(path).resolve()
    assert p not in TARGETS.values(), 'Truth bytes before pre-score seal'
    digest = pins.get(str(p))
    if digest is None:
        digest = sha(p)
        pins[str(p)] = digest
    assert expected is None or digest == expected, str(p)
    return digest


def create(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


for f in [HERE / 'PROTOCOL.md', Path(__file__), CAP / 'common.py', CAP / 'judge_terminal.py',
          CAP / 'TERMINAL_FIDELITY_FAILURE_2026_09_09.md', CAP / 'TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md']:
    pin(f)
pin(TRUTH_INDEX, '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c')
pin(CAP / 'terminal_score_v1/result.json', '7f07b494eed79a512e716cfdd3a4134fb1845ec29bba242899056b843730abce')
old_failure = read(CAP / 'terminal_score_v1/result.json')
assert not old_failure['passed'] and not old_failure['truth_accessed']
pin(CAP / 'training_v1/result.json', '5499433f54ed648c66716cfd9044fa7908c6024f9bb20012f6c091d56f8500d8')
training = read(CAP / 'training_v1/result.json')
assert training['updates'] == 240 and training['validation_ranks'] == list(VALID)
pin(CAP / 'build_terminal_v1/build.json', '027186a362bb487a76f993b7c9b9a9c68c4be1157f7f05cd1aa8e403219ff4a1')
build = read(CAP / 'build_terminal_v1/build.json')
assert build['passed']
terminal_sha = pin(CAP / 'build_terminal_v1/estimator.py', '604d23a2118cfd7781ff3a51b4adc9d9125c586b051a9cc797c23a0573735a6a')
pin(BD / 'qualify_native_v2.py', '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608')
sys.path.insert(0, str(CAP))
judge = module(CAP / 'judge_terminal.py', 'cap4096_prior_judge_utilities')
v2 = module(BD / 'qualify_native_v2.py', 'cap4096_prior_native_verifier')
verify = judge.native_verifier(v2, terminal_sha)
folders = {'old_bank': BD / 'validation_bank_v2', 'null_cap': CAP / 'native_validation_null_v1',
           'terminal_cap': CAP / 'native_validation_terminal_v1'}
acquisitions, raw_bytes = {}, {}
for role, folder in folders.items():
    digest = pin(folder / 'result.json')
    pin(folder / 'completion.json')
    acq, end = read(folder / 'result.json'), read(folder / 'completion.json')
    assert end['passed'] and end['result_sha256'] == digest
    if role == 'terminal_cap':
        assert digest == 'ad5107badc63f81630595db4f5d6058f1ee44cc2c6be44b3af38147308f1c598'
    raw_bytes[role] = verify(acq, folder, VALID, pin)
    acquisitions[role] = acq

import numpy as np


def checked_array(path, expected_bytes=None):
    a = np.load(path, allow_pickle=False)
    assert a.shape == (16,1024) and a.dtype == np.float32 and np.isfinite(a).all() and (a >= 0).all()
    assert expected_bytes is None or hashlib.sha256(a.tobytes()).hexdigest() == expected_bytes
    return a


pred = {role: [np.frombuffer(b, dtype='<f4').reshape(16,1024) for b in values] for role, values in raw_bytes.items()}
shadow = []
for i, rank in enumerate(VALID):
    item = training['validation_outputs'][str(rank)]
    pin(item['path'], item['sha256'])
    shadow.append(checked_array(item['path'], item['raw_sha256']))
    for role in pred:
        a = pred[role][i]
        assert np.isfinite(a).all() and (a >= 0).all()
        assert a[0].tobytes() == pred['old_bank'][i][0].tobytes()

pin(LAZY / 'closeout.json', '73cb703f22b2ac12d78b69ab3821157d8b6cc36195097365ff58e7e01e164313')
lazy_closeout = read(LAZY / 'closeout.json')
pin(LAZY / 'confirmation_score.json', lazy_closeout['artifacts_sha256']['confirmation_score.json'])
lazy_score = read(LAZY / 'confirmation_score.json')
pin(LAZY / 'confirmation_predictions_sealed.json', lazy_closeout['artifacts_sha256']['confirmation_predictions_sealed.json'])
lazy_seal = read(LAZY / 'confirmation_predictions_sealed.json')
lazy_predictions, lazy_flops = {}, {}
for rank in VALID[:8]:
    folder = LAZY / 'confirmation' / f'candidate_{rank}'
    result_path = folder / 'result.json'
    # The native runner owns one fresh process/cold prediction per directory.
    pin(result_path, lazy_closeout['artifacts_sha256'][str(result_path.relative_to(LAZY))])
    result = read(result_path)
    assert result['candidate_sha256'] == lazy_closeout['candidate_sha256']
    assert len(result['records']) == 1 and not result['truth_accessed']
    trial = result['records'][0]
    assert trial['rank'] == rank and trial['index'] == 0 and trial['includes_initialization']
    sealed_record = next(r for r in lazy_seal['records'] if r['rank'] == rank)
    assert trial == sealed_record['arms']['candidate']['call']
    watch_path = Path(sealed_record['arms']['candidate']['watch'])
    pin(watch_path, lazy_closeout['artifacts_sha256'][str(watch_path.relative_to(LAZY))])
    watch = read(watch_path)
    assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
    output = Path(trial['output'])
    pin(output, trial['output_sha256'])
    lazy_predictions[rank] = checked_array(output)
    lazy_flops[rank] = trial['flops']

create('predictions_sealed.json', {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files_sha256': pins, 'candidate_sha256': terminal_sha, 'ranks': list(VALID),
    'reference_means_accessed': False, 'old_fidelity_gate_passed': False,
    'cost_scope': 'Original sequential own bills, initialization once; Lazy comparison uses its actual cold bills.'})
preseal = sha(HERE / 'predictions_sealed.json')
truth_records = read(TRUTH_INDEX)['records']
truth_hashes = {x['rank']: x['target_sha256'] for x in truth_records}
assert set(truth_hashes) == set(VALID)
rows = []
for i, rank in enumerate(VALID):
    assert sha(TARGETS[rank]) == truth_hashes[rank]
    with np.load(TARGETS[rank], allow_pickle=False) as z:
        target = z['mc_post'].astype(np.float64)
    assert target.shape == (16,1024) and np.isfinite(target).all()
    row = {'rank': rank, 'truth_sha256': truth_hashes[rank], 'roles': {}}
    for role, arrays in pred.items():
        raw = float(np.mean((arrays[i][-1].astype(np.float64) - target[-1])**2))
        flops = acquisitions[role]['trials'][i]['flops']
        row['roles'][role] = {'raw': raw, 'flops': flops, 'adjusted': raw * max(.1, flops / 2**41)}
    raw_shadow = float(np.mean((shadow[i][-1].astype(np.float64) - target[-1])**2))
    row['shadow_raw'] = raw_shadow
    row['shadow_to_native_raw'] = raw_shadow / row['roles']['terminal_cap']['raw']
    drift = shadow[i].astype(np.float64) - pred['terminal_cap'][i].astype(np.float64)
    row['native_shadow_maxabs'] = float(np.max(np.abs(drift)))
    row['native_shadow_final_rms'] = float(np.sqrt(np.mean(drift[-1]**2)))
    if rank in lazy_predictions:
        raw = float(np.mean((lazy_predictions[rank][-1].astype(np.float64) - target[-1])**2))
        flops = lazy_flops[rank]
        row['roles']['lazy'] = {'raw': raw, 'flops': flops, 'adjusted': raw * max(.1, flops / 2**41)}
    rows.append(row)


def aggregate(role, selected):
    return {k: math.fsum(r['roles'][role][k] for r in selected) / len(selected) for k in ['raw','flops','adjusted']}


all40 = {role: aggregate(role, rows) for role in pred}
common = rows[:8]
common8 = {role: aggregate(role, common) for role in ['terminal_cap','lazy']}
comparisons = {}
for name, selected, other in [('old_bank40', rows, 'old_bank'), ('null_cap40', rows, 'null_cap'), ('lazy8', common, 'lazy')]:
    own, parent = aggregate('terminal_cap', selected), aggregate(other, selected)
    comparisons[name] = {'adjusted_ratio': own['adjusted']/parent['adjusted'],
        'raw_ratio': own['raw']/parent['raw'],
        'wins': sum(r['roles']['terminal_cap']['adjusted'] < r['roles'][other]['adjusted'] for r in selected),
        'networks': len(selected)}
advance = (comparisons['old_bank40']['adjusted_ratio'] <= .99 and comparisons['old_bank40']['wins'] >= 24
           and comparisons['lazy8']['adjusted_ratio'] <= .99 and comparisons['lazy8']['wins'] >= 6)
assert abs(common8['lazy']['adjusted'] - lazy_score['aggregate']['candidate']['adjusted']) < 1e-23
assert all(sha(p) == h for p, h in pins.items())
assert sha(HERE / 'predictions_sealed.json') == preseal
result = {'completed': True, 'candidate_sha256': terminal_sha, 'old_fidelity_gate_passed': False,
          'rows': rows, 'all40': all40, 'common8': common8, 'comparisons': comparisons,
          'objective_advancement_passed': advance, 'new_network_calls': 0, 'new_training_updates': 0,
          'production_qualified': False, 'truth_accessed': True,
          'cost_scope': 'Cap and old bank: recorded sequence bills with initialization once. Lazy: complete independent-cold bills.',
          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'elapsed_s': time.monotonic() - started, 'pretruth_seal_sha256': preseal,
          'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
create('result.json', result)
print(json.dumps({k: result[k] for k in ['completed','all40','common8','comparisons','objective_advancement_passed','peak_rss_bytes','elapsed_s']}))
