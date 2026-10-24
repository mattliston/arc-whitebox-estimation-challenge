"""Replay one unchanged-endpoint confirmation panel; never run an estimator."""
from pathlib import Path
import io
import json
import math
import sys

import numpy as np
import validation_common as common
from validation_common import metrics, once, pin

D = Path(__file__).resolve().parent
H = D.parent
RANKS = [75, 79, 93, 101, 122, 130, 132, 135]
FIRST = [37, 39, 42, 48, 59, 63, 66, 67]
COUNTS = dict(bound=687, lvfuse=521, psdb_fused=9,
              pair_block_indices=1, read7_fused=4, ksplit=24)
G8 = dict(raw_pct_max=0, adjusted_pct_max=0, strict_gain=True,
          wins_min=5, maxreg=1)
G16 = dict(raw_max=-.1, adj_max=-.1, wins_min=10,
           raw_and_adjusted_ci_upper_max=0, maxreg=1)


def assess(config_path, digest):
    evidence = {}

    def checked(record):
        actual = pin(record['path'])
        assert actual == record, record['path']
        if actual['path'] in evidence:
            assert evidence[actual['path']] == actual
        evidence[actual['path']] = actual
        return Path(actual['path']).read_bytes()

    def check_pins(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                checked(value)
            else:
                for child in value.values():
                    check_pins(child)
        elif isinstance(value, list):
            for child in value:
                check_pins(child)

    def saved(record):
        return json.loads(checked(record))

    def read_rows(record):
        return [json.loads(line) for line in checked(record).decode().splitlines()
                if line.strip()]

    def guard_ok(record, command, seconds):
        g = saved(record)
        assert g['command'] == command
        assert g['completed'] is True and g['exit_code'] == 0 and g['error'] is None
        assert g['whole_group_retired'] is True and g['remaining_rows'] == []
        assert 0 < g['elapsed_s'] <= seconds
        assert 0 < g['max_rss_bytes'] < g['rss_trigger_bytes'] == 7500000000
        assert g['max_swap_growth_bytes'] == 0
        return g

    def setup_ok(manifest):
        assert manifest['runtime'] == c['runtime'] and manifest['numpy'] == '2.4.6'
        assert manifest['thread_env'] == {k: '1' for k in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS')}
        setup = manifest['setup']
        assert math.isfinite(setup['setup_s']) and 0 < setup['setup_s'] <= 4.1
        labels = [x[0] for x in setup['pb_log']]
        assert len(labels) == len(set(labels))
        counts = {x[0]: x[1] for x in setup['pb_log']}
        assert all(counts.get(k) == v for k, v in COUNTS.items())
        assert not any('wall guard' in k or 'restored' in k or 'lost' in k for k in labels)
        return setup

    def prediction(record):
        value = np.load(io.BytesIO(checked(record)), allow_pickle=False)
        assert value.shape == (16, 1024) and value.dtype == np.dtype('float64')
        assert np.isfinite(value).all()
        return value

    config_pin = pin(config_path)
    assert config_path == D / 'confirmation_protocol.json' and config_pin['sha256'] == digest
    c = saved(config_pin)
    assert c['schema'] == 'full80-fixed-endpoint-confirmation-v1'
    assert c['assessor'] == pin(__file__) and c['common'] == pin(common.__file__)
    assert c['confirmation8'] == G8 and c['pooled16'] == G16
    assert sys.executable == c['runtime'] and np.__version__ == '2.4.6'
    check_pins(c)
    first = saved(c['first_assessment'])
    first_config = saved(c['first_config'])
    assert first['config'] == c['first_config']
    assert first['passed'] is False and first['admitted'] is False
    assert first['valid8_admitted'] is False and first['failed_stage'] == 'VALID8'
    assert first['candidate'] == first_config['candidate'] == c['candidate']
    assert first['common'] == c['common']
    check_pins(first)
    assert first['assessor']['path'] == str(D / 'assess_validation.py')
    # Reuse the original unchanged replay. This cannot retroactively admit VALID8.
    from assess_validation import assess as assess_first
    _, replayed_first = assess_first(Path(c['first_config']['path']),
                                     c['first_config']['sha256'])
    assert replayed_first == first
    original_protocol = saved(first['protocol'])
    index1 = saved(first['baseline_index'])
    base1 = read_rows(index1['bills'])[:8]
    child1 = read_rows(first['artifacts']['rows.jsonl'])
    assert [r['rank'] for r in base1] == [r['rank'] for r in child1] == FIRST

    data = saved(c['data_index'])
    check_pins(data)
    assert data['schema'] == 'full80-post-valid8-failure-confirmation-data-v1'
    assert data['candidate'] == c['candidate'] and data['selected_endpoint'] == 80
    assert data['source_changes_since_failed_valid8'] is False
    assert data['ranks'] == data['confirmation_ranks'] == RANKS
    assert data['initial_validation_ranks'] == FIRST
    assert data['original_training_ranks'] == original_protocol['training_ranks']
    assert not set(RANKS) & (set(FIRST) | set(data['original_training_ranks']))
    assert set(RANKS) <= set(original_protocol['validation']['fixed40'])
    idx = data['baseline_index']
    assert idx['ranks'] == RANKS and idx['actual_block_order'][:8] == RANKS
    assert idx['source'] == data['baseline'] == first['baseline'] == original_protocol['official']
    assert c['driver'] == idx['source_driver'] and c['guard_source'] == idx['guard_source']
    assert c['runtime'] == idx['runtime']
    bm = saved(idx['manifest'])
    assert bm['candidate'] == idx['source']['path'] and bm['sha256'] == idx['source']['sha256']
    assert bm['ranks'] == idx['actual_block_order']
    setup_ok(bm)
    job = saved(idx['block_job'])
    assert job['passed'] is True and job['guard_exit_code'] == 0
    assert job['candidate'] == idx['source'] and job['guard'] == idx['block_guard']
    assert job['command'] == idx['command'] and job['config'] == idx['broad_config']
    guard_ok(idx['block_guard'], idx['command'], 420)
    full_base2 = read_rows(idx['bills'])
    assert [r['rank'] for r in full_base2] == idx['actual_block_order']
    base2 = full_base2[:8]
    assert len(data['data']) == len(idx['data']) == 8
    assert [r['rank'] for r in data['data']] == [r['rank'] for r in idx['data']] == RANKS
    authority = saved(data['authorities']['original_inputs'])
    for item in data['data']:
        r = str(item['rank'])
        assert item['weights'] == authority['weights'][r]
        assert item['truth'] == authority['targets'][r]

    out = Path(c['output'])
    assert out == H / 'hill_climb_rrpi_full80_v1_confirm8_v1'
    guard_path = Path(c['guard'])
    assert guard_path == D / 'confirm8_guard.json'
    guard_record = pin(guard_path)
    command = [c['runtime'], '-B', c['driver']['path'], c['candidate']['path'],
               ','.join(map(str, RANKS)), out.name]
    guard_ok(guard_record, command, 300)
    artifacts = {k: pin(out / k) for k in ('manifest.json', 'rows.jsonl')}
    manifest = saved(artifacts['manifest.json'])
    assert manifest['candidate'] == c['candidate']['path']
    assert manifest['sha256'] == c['candidate']['sha256'] and manifest['ranks'] == RANKS
    setup = setup_ok(manifest)
    child2 = read_rows(artifacts['rows.jsonl'])
    assert len(child2) == 8 and [r['rank'] for r in child2] == RANKS
    predictions, replay = {}, []
    for item, old, base, row in zip(data['data'], idx['data'], base2, child2):
        rank = item['rank']
        assert old['saved_row'] == base and base['rank'] == row['rank'] == rank
        assert item['truth'] == old['truth'] and item['driver_truth'] == old['driver_truth']
        assert item['truth']['sha256'] == item['driver_truth']['sha256']
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert all(math.isfinite(row[k]) for k in ('mse', 'util', 'adjusted', 'wall_s', 'cpu_s'))
        assert row['mse'] >= 0 and row['wall_s'] > 0 and row['cpu_s'] > 0
        assert row['util'] == row['flops'] / 2**41
        assert row['adjusted'] == row['mse'] * max(.1, row['util'])
        assert row['ubc'].get('retired') is True
        assert row['ubc'].get('retained_cache_bytes') == row['ubc'].get('live_cache_bytes') == 0
        pred_pin = pin(out / f'pred_{rank}.npy')
        predictions[str(rank)] = pred_pin
        y, baseline = prediction(pred_pin), prediction(old['prediction'])
        with np.load(io.BytesIO(checked(item['truth'])), allow_pickle=False) as saved_truth:
            truth = saved_truth['mc_post']
            assert truth.shape == (16, 1024) and np.isfinite(truth).all()
            truth = truth[15].astype(np.float64)
        assert y[0].tobytes() == baseline[0].tobytes()
        assert float(np.mean((y[15] - truth)**2)) == row['mse']
        assert float(np.mean((baseline[15] - truth)**2)) == base['mse']
        replay.append(dict(rank=rank, all16_finite=True, layer0_byte_exact=True,
                           candidate_and_baseline_mse_exact=True))
        del y, baseline, truth
    assert {p.name for p in out.iterdir()} == set(artifacts) | {f'pred_{r}.npy' for r in RANKS}
    m8 = metrics(base2, child2)
    m16 = metrics(base1 + base2, child1 + child2)
    g8 = dict(raw=m8['raw_pct'] < G8['raw_pct_max'],
              adjusted=m8['adjusted_pct'] < G8['adjusted_pct_max'],
              wins=m8['adjusted_wins'] >= G8['wins_min'],
              max_regression=m8['max_per_network_adjusted_regression_pct'] <= G8['maxreg'])
    g16 = dict(raw=m16['raw_pct'] <= G16['raw_max'],
               adjusted=m16['adjusted_pct'] <= G16['adj_max'],
               wins=m16['adjusted_wins'] >= G16['wins_min'],
               raw_ci=m16['raw_ci95_pct'][1] < G16['raw_and_adjusted_ci_upper_max'],
               adjusted_ci=m16['adjusted_ci95_pct'][1] < G16['raw_and_adjusted_ci_upper_max'],
               max_regression=m16['max_per_network_adjusted_regression_pct'] <= G16['maxreg'])
    passed = all(g8.values()) and all(g16.values())
    checked(c['first_assessment'])  # Original failure receipt must still be unchanged.
    return dict(schema='full80-fixed-endpoint-confirmation-assessment-v1',
        passed=passed, retained_evidence_passed=passed, admitted=False,
        original_valid8_admitted=False, original_valid8_preserved=True,
        config=config_pin, candidate=c['candidate'], baseline=idx['source'],
        first_assessment=c['first_assessment'], data_index=c['data_index'],
        assessor=pin(__file__), common=c['common'], driver=c['driver'],
        guard=guard_record, artifacts=artifacts, predictions=predictions,
        setup=setup, replay=replay, all16_finite=True, layer0_byte_exact=True,
        receipt_checks_passed=True, resource_gates_passed=True,
        confirmation8=m8, confirmation8_gates=g8, pooled16=m16, pooled16_gates=g16,
        input_evidence=list(evidence.values()), adaptive_confirmation=True,
        historically_reused_validation=True, nominal_ci_descriptive_not_fixed_design=True,
        own_boundary_checked=False, retained_component=False, current_winner_changed=False,
        bank_qualified=False, serving_qualified=False, broad_admitted=False,
        submitted=False, model_executed=False, targets_read=True)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    result = assess(Path(sys.argv[1]).resolve(), sys.argv[2])
    output = D / 'confirmation_assessment.json'
    once(output, result)
    print(json.dumps(dict(passed=result['passed'], retained_evidence_passed=result['retained_evidence_passed'],
                         output=pin(output))))
