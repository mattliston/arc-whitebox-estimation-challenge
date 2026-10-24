"""Future-only saved native compression TRAIN2/8 assessment; never runs an estimator or reads truth.

Root invocation: NATIVE -B screen.py CONFIG_PATH CONFIG_SHA {2|8} under its guard.
The immutable JSON config is the complete prospective source/input/gate contract.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import os
import sys
import numpy as np

H2 = Path(__file__).resolve().parent.parent
NATIVE = H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
ROSTERS = {2: [35, 100], 6: [36, 45, 52, 58, 64, 70]}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
EVIDENCE = {}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    path = Path(record['path'])
    assert path.is_absolute() and str(path.resolve()) == record['path']
    raw = path.read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], str(path)
    EVIDENCE[str(path)] = record
    return raw


def read(path):
    return json.loads(checked(pin(path)))


def once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def guard(path, limits, command):
    assert 0 < limits['seconds'] <= 600 and 0 < limits['rss_bytes'] <= 7500000000
    value = read(path)
    actual_command = value['command'][1:] if value['command'][:1] == ['--'] else value['command']
    assert actual_command == command
    assert value['completed'] is True and value['exit_code'] == 0 and value['error'] is None
    assert value['whole_group_retired'] is True and value['remaining_rows'] == []
    assert value['rss_trigger_bytes'] == limits['rss_bytes']
    assert 0 < value['max_rss_bytes'] < limits['rss_bytes'] and value['max_swap_growth_bytes'] == 0
    assert 0 < value['elapsed_s'] <= limits['seconds'] and value['samples'] > 0
    return value


def configure(path, digest):
    config_pin = pin(path)
    assert len(digest) == 64 and config_pin['sha256'] == digest
    c = json.loads(checked(config_pin))
    assert c['schema'] == 'native-screen-v1' and c['prospective_authorization'] is True
    assert c['future_only'] is True and c['raw_error_veto'] is True
    assert c['engine'] == pin(__file__); checked(c['engine'])
    assert set(c['arms']) in ({'candidate', 'base'}, {'candidate', 'base', 'parity_parent'})
    assert set(c['stages']) == {'2', '8'}
    assert c['arms']['candidate']['ubc_retired_required'] is True
    boundary = c['boundary_before8']
    checked(boundary['assessor'])
    assert boundary['assessment_command'][2] == boundary['assessor']['path']
    assert boundary['assessment_command'][1] == '-B'
    for record in c['provenance']:
        checked(record)
    for arm in c['arms'].values():
        checked(arm['source'])
        assert type(arm['ubc_retired_required']) is bool
        assert set(arm['blocks']) == {'2', '6'}
        assert arm['setup']['seconds_max'] == 4.1
        assert arm['setup']['counts'] and all(type(v) is int and v > 0 for v in arm['setup']['counts'].values())
        for size in (2, 6):
            block = arm['blocks'][str(size)]
            assert block['ranks'] == ROSTERS[size]
            assert Path(block['tag']).name == block['tag']
            checked(block['driver'])
            assert type(block['no_bytecode']) is bool
    if 'parity_parent' in c['arms']:
        layers = c['parity_layers']
        assert layers == sorted(set(layers)) and layers and all(type(i) is int and 0 <= i < 16 for i in layers)
    else:
        assert c['parity_layers'] == []
    for n, stage in c['stages'].items():
        g = stage['gates']
        assert set(g) == {'adjusted_pct_at_most', 'strict_adjusted', 'wins_min', 'ci_upper_below',
                          'max_per_network_regression_pct', 'additional6_gain_required'}
        assert math.isfinite(g['adjusted_pct_at_most']) and g['adjusted_pct_at_most'] <= 0
        assert type(g['strict_adjusted']) is bool and type(g['additional6_gain_required']) is bool
        assert type(g['wins_min']) is int and 0 <= g['wins_min'] <= int(n)
        assert g['ci_upper_below'] is None or (math.isfinite(g['ci_upper_below']) and g['ci_upper_below'] <= 0)
        assert g['max_per_network_regression_pct'] is None or (math.isfinite(g['max_per_network_regression_pct']) and g['max_per_network_regression_pct'] >= 0)
        assert n == '8' or g['additional6_gain_required'] is False
    return config_pin, c


def panel(arm, size, role):
    block = arm['blocks'][str(size)]; ranks = ROSTERS[size]
    folder = H2/block['tag']
    paths = dict(manifest=folder/'manifest.json', rows=folder/'rows.jsonl', guard=Path(block['guard']))
    records = {key: pin(path) for key, path in paths.items()}
    # Existing controls/parity parents are pinned prospectively; child outputs do not exist yet.
    if role != 'candidate':
        assert records == {k: block['artifacts'][k] for k in ('manifest', 'rows', 'guard')}
    for key in ('manifest', 'rows', 'guard'):
        checked(records[key])
    source = arm['source']; checked(source)
    m = json.loads(checked(records['manifest']))
    assert m['candidate'] == source['path'] and m['sha256'] == source['sha256']
    assert m['ranks'] == ranks and m['runtime'] == str(NATIVE) and m['numpy'] == '2.4.6'
    assert m['thread_env'] == dict.fromkeys(THREADS, '1')
    command = [str(NATIVE)] + (['-B'] if block['no_bytecode'] else [])
    command += [block['driver']['path'], source['path'], ','.join(map(str, ranks)), block['tag']]
    guard(paths['guard'], block['limits'], command)
    setup = m['setup']; assert 0 <= setup['setup_s'] <= arm['setup']['seconds_max']
    for name, count in arm['setup']['counts'].items():
        assert [x[1] for x in setup['pb_log'] if x[0] == name] == [count]
    rows = [json.loads(line) for line in checked(records['rows']).decode().splitlines()]
    assert [row['rank'] for row in rows] == ranks
    for row in rows:
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ('mse', 'adjusted', 'util', 'cpu_s', 'wall_s'))
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert row['util'] == row['flops']/2**41
        assert math.isclose(row['adjusted'], row['mse']*max(.1, row['util']), rel_tol=1e-14, abs_tol=0)
        if arm['ubc_retired_required']:
            ubc = row['ubc']
            assert ubc['retired'] is True and ubc['retained_cache_bytes'] == ubc['live_cache_bytes'] == 0
            assert 0 <= ubc['max_resident_bytes'] <= 512000000
    records['predictions'] = {str(r): pin(folder/f'pred_{r}.npy') for r in ranks}
    if role != 'candidate':
        assert records == block['artifacts']
    arrays = {}
    for rank in ranks:
        data = checked(records['predictions'][str(rank)])
        array = np.load(io.BytesIO(data), allow_pickle=False)
        assert array.shape == (16, 1024) and array.dtype == np.dtype('float64') and np.isfinite(array).all()
        arrays[rank] = array
    return rows, arrays, records


def metrics(base, child):
    base = {r['rank']: r for r in base}; child = {r['rank']: r for r in child}
    ranks = sorted(base); assert ranks == sorted(child)
    keys = ('mse', 'adjusted', 'util', 'cpu_s')
    b = np.array([[base[r][k] for k in keys] for r in ranks])
    c = np.array([[child[r][k] for k in keys] for r in ranks])
    indices = np.random.default_rng(20260926).integers(0, len(ranks), size=(50000, len(ranks)))
    ci = np.quantile(c[indices, :2].mean(1)/b[indices, :2].mean(1)-1, [.025, .975], axis=0)*100
    per = [dict(rank=r, raw_pct=100*(child[r]['mse']/base[r]['mse']-1),
        adjusted_pct=100*(child[r]['adjusted']/base[r]['adjusted']-1),
        util_pct=100*(child[r]['util']/base[r]['util']-1)) for r in ranks]
    return dict(n=len(ranks), ranks=ranks, raw_pct=float(100*(c[:, 0].mean()/b[:, 0].mean()-1)),
        adjusted_pct=float(100*(c[:, 1].mean()/b[:, 1].mean()-1)), raw_ci95_pct=ci[:, 0].tolist(),
        adjusted_ci95_pct=ci[:, 1].tolist(), adjusted_wins=int((c[:, 1] < b[:, 1]).sum()),
        raw_wins=int((c[:, 0] < b[:, 0]).sum()), base_mean_adjusted=float(b[:, 1].mean()),
        candidate_mean_adjusted=float(c[:, 1].mean()), mean_util_pct=float(100*(c[:, 2].mean()/b[:, 2].mean()-1)),
        mean_cpu_pct=float(100*(c[:, 3].mean()/b[:, 3].mean()-1)),
        bill_pct=100*(sum(r['flops'] for r in child.values())/sum(r['flops'] for r in base.values())-1),
        max_per_network_adjusted_regression_pct=max(r['adjusted_pct'] for r in per), per_network=per)


def assess(c, config_pin, n):
    prior_pin = None
    if n == 8:
        prior_pin = pin(c['stages']['2']['result'])
        prior = json.loads(checked(prior_pin))
        assert prior['passed_integrity'] is True and prior['admitted'] is True and prior['n'] == 2
        assert prior['config'] == config_pin and prior['engine'] == c['engine']
        assert prior['candidate'] == c['arms']['candidate']['source']
        command = [str(NATIVE), '-B', c['engine']['path'], config_pin['path'], config_pin['sha256'], '2']
        guard(c['stages']['2']['assessment_guard'], c['stages']['2']['assessment_limits'], command)
        for record in prior['evidence']:
            checked(record)  # Prevent replacement of a previously admitted TRAIN2 input.
        requirement = c['boundary_before8']
        boundary = read(requirement['receipt'])
        assert boundary['passed'] is True and boundary['receipt_checks_passed'] is True
        assert boundary['candidate'] == c['arms']['candidate']['source'] and boundary['native2'] == prior_pin
        assert boundary['assessor'] == requirement['assessor']
        guard(requirement['assessment_guard'], requirement['assessment_limits'], requirement['assessment_command'])
    rows = {role: [] for role in c['arms']}; blocks = []
    for size in ((2,) if n == 2 else (2, 6)):
        outputs = {}; evidence = {}
        for role, arm in c['arms'].items():
            rr, aa, records = panel(arm, size, role)
            rows[role].extend(rr); outputs[role] = aa; evidence[role] = records
        if 'parity_parent' in outputs:
            for rank in ROSTERS[size]:
                for layer in c['parity_layers']:
                    assert outputs['candidate'][rank][layer].tobytes() == outputs['parity_parent'][rank][layer].tobytes(), (rank, layer)
                if c['parity_layers'] == list(range(16)):
                    child_mse = next(r['mse'] for r in rows['candidate'] if r['rank'] == rank)
                    parent_mse = next(r['mse'] for r in rows['parity_parent'] if r['rank'] == rank)
                    assert child_mse == parent_mse
        blocks.append(dict(size=size, arms=evidence))
    result = metrics(rows['base'], rows['candidate']); limits = c['stages'][str(n)]['gates']
    value = result['adjusted_pct']; maximum = limits['adjusted_pct_at_most']
    gates = dict(adjusted_gain=value < maximum if limits['strict_adjusted'] else value <= maximum,
                 wins=result['adjusted_wins'] >= limits['wins_min'])
    gates['raw_budget'] = math.isfinite(result['raw_pct']) and result['raw_pct'] <= 1.0
    if limits['ci_upper_below'] is not None:
        gates['ci_upper_negative'] = result['adjusted_ci95_pct'][1] < limits['ci_upper_below']
    if limits['max_per_network_regression_pct'] is not None:
        gates['max_regression'] = result['max_per_network_adjusted_regression_pct'] <= limits['max_per_network_regression_pct']
    additional = None
    if n == 8:
        additional = metrics([r for r in rows['base'] if r['rank'] in ROSTERS[6]],
                             [r for r in rows['candidate'] if r['rank'] in ROSTERS[6]])
        if limits['additional6_gain_required']:
            gates['additional6_gain'] = additional['adjusted_pct'] < 0
    for record in list(EVIDENCE.values()):
        checked(record)
    return dict(passed_integrity=True, admitted=all(gates.values()), **result, gates=gates,
        candidate=c['arms']['candidate']['source'], comparator=c['arms']['base']['source'],
        config=config_pin, engine=c['engine'], prior2=prior_pin, blocks=blocks, evidence=list(EVIDENCE.values()),
        parity_parent=c['arms'].get('parity_parent', {}).get('source'), parity_layers=c['parity_layers'],
        boundary=pin(c['boundary_before8']['receipt']) if n == 8 else None,
        all16_predictions_finite=True, configured_parent_bytes_exact=bool(c['parity_layers']),
        all16_parent_bytes_exact=c['parity_layers'] == list(range(16)), additional6=additional,
        bootstrap=dict(resamples=50000, seed=20260926), raw_error_veto=True,
        future_only=True, targets_read=False, model_executed=False, release_qualified=False, submitted=False)


def main():
    assert len(sys.argv) == 4 and sys.argv[3] in ('2', '8')
    assert Path(sys.executable).absolute() == NATIVE and np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in THREADS)
    config_pin, c = configure(Path(sys.argv[1]).resolve(), sys.argv[2]); n = int(sys.argv[3])
    target = Path(c['stages'][str(n)]['result']); assert target.is_absolute() and not target.exists()
    try:
        result = assess(c, config_pin, n)
    except Exception as exc:
        once(target, dict(passed_integrity=False, admitted=False, n=n, config=config_pin, engine=c['engine'],
            candidate=c['arms']['candidate']['source'], error=dict(type=type(exc).__name__, message=str(exc)),
            future_only=True, model_executed=False, release_qualified=False, submitted=False))
        raise
    once(target, result)
    print(json.dumps({k: result[k] for k in ('passed_integrity', 'admitted', 'n', 'adjusted_pct', 'adjusted_wins', 'gates')}))


if __name__ == '__main__':
    main()
