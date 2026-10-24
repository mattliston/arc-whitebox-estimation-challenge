"""Read-only replay of one adaptive, unchanged-endpoint follow-up; no model execution."""
from pathlib import Path
import io
import json
import math
import sys
import numpy as np
import validation_common as common
import assess_validation as original
from validation_common import checked, metrics, once, pin

D = Path(__file__).resolve().parent
RANKS = [37, 39, 42, 48, 59, 63, 66, 67]
G6 = dict(raw_pct_max=0.0, adjusted_pct_max=0.0, strict_gain=True,
          wins_min=4, max_regression_pct=2.0)
G8 = dict(raw_pct_max=-0.1, adjusted_pct_max=-0.1, wins_min=6,
          max_regression_pct=2.0, raw_ci_upper_max=0.0, adjusted_ci_upper_max=0.0)


def assess(path, digest):
    assert path.parent == D and pin(path)['sha256'] == digest
    p = json.loads(path.read_text())
    assert p['schema'] == 'detached-star8-adaptive-followup-v1' and p['root_admitted'] is True
    for record in p['pins']:
        checked(record)
    for key in ('original_config', 'original_assessment', 'original_guard', 'candidate',
                'common', 'driver', 'guard_source'):
        checked(p[key])
    assert Path(p['assessor_path']).resolve() == Path(__file__).resolve()
    assert p['common'] == pin(common.__file__)
    assert p['assessor_source_seal_required_before_assessment'] is True
    seal_pin = pin(D / 'followup_assessor_seal.json')
    seal = json.loads(checked(seal_pin))
    assert seal['passed'] is True and seal['protocol'] == pin(path) and seal['assessor'] == pin(__file__)
    checked(seal['review'])
    assert p['ranks'] == RANKS and p['known_ranks'] == RANKS[:2] and p['new_ranks'] == RANKS[2:]
    assert p['gates_new6'] == G6 and p['gates_pooled8'] == G8
    assert all(p[k] is True for k in ('source_unchanged', 'no_parameter_checkpoint_roster_or_damping_search',
                                    'no_bank_from_this_study', 'bank_requires_original_full40_and_resources'))
    assert p['guard_limits'] == dict(seconds=300, rss_bytes=7500000000, added_swap_bytes=0)
    assert sys.executable == p['runtime'] and np.__version__ == '2.4.6'
    # Recheck every original source/control/authority and replay the failed initial screen.
    c, old = original.assess(Path(p['original_config']['path']), p['original_config']['sha256'])
    assert old == json.loads(checked(p['original_assessment']))
    assert old['admitted'] is False and old['passed'] is False and old['stage'] == 'VALID2'
    assert old['failed_stage'] == 'VALID2' and old['gates2']['max_regression'] is False
    assert all(v for k, v in old['gates2'].items() if k != 'max_regression')
    assert old['native_guard'] == p['original_guard']
    assert p['candidate'] == c['candidate'] and p['common'] == c['common']
    assert p['runtime'] == c['runtime'] and p['guard_source'] == c['guard_source']
    assert p['setup_counts'] == c['setup_counts'] == original.SETUP_COUNTS
    c2, index, base = common.configuration(Path(p['original_config']['path']), p['original_config']['sha256'])
    assert c2 == c
    assert p['driver'] == pin(D.parent / 'hill_climb_e1_ynt_psdb_d5_v1' / 'native_probe.py')
    out = Path(p['output'])
    assert p['tag'] == 'hill_climb_rrpi_detached_star_v1_followup8_v1'
    assert out == D.parent / p['tag']
    assert Path(p['native_guard']) == D / 'followup8_guard.json'
    guard_pin = pin(p['native_guard'])
    guard = json.loads(checked(guard_pin))
    expected = [p['runtime'], '-B', p['driver']['path'], p['candidate']['path'],
                ','.join(map(str, RANKS)), p['tag']]
    assert guard['command'] == expected
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert 0 < guard['elapsed_s'] <= 300 and 0 < guard['max_rss_bytes'] < 7500000000
    assert guard['rss_trigger_bytes'] == 7500000000 and guard['max_swap_growth_bytes'] == 0
    artifacts = {name: pin(out / name) for name in ('manifest.json', 'rows.jsonl')}
    manifest = json.loads(checked(artifacts['manifest.json']))
    assert manifest['candidate'] == p['candidate']['path'] and manifest['sha256'] == p['candidate']['sha256']
    assert manifest['ranks'] == RANKS and manifest['runtime'] == p['runtime'] and manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == {k: '1' for k in
        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    setup = manifest['setup']
    assert math.isfinite(setup['setup_s']) and 0 < setup['setup_s'] <= 4.1
    labels = [v[0] for v in setup['pb_log']]
    assert len(labels) == len(set(labels))
    counts = {v[0]: v[1] for v in setup['pb_log']}
    assert all(counts.get(k) == v for k, v in p['setup_counts'].items())
    assert not any('wall guard' in k or 'restored' in k or 'lost' in k for k in labels)
    rows = [json.loads(s) for s in checked(artifacts['rows.jsonl']).decode().splitlines() if s.strip()]
    assert [r['rank'] for r in rows] == RANKS
    oldrows = [json.loads(s) for s in checked(old['artifacts']['rows.jsonl']).decode().splitlines() if s.strip()]
    assert len(oldrows) == 2
    predictions, replay = {}, []
    for i, row in enumerate(rows):
        rank = row['rank']
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert all(math.isfinite(row[k]) for k in ('mse', 'util', 'adjusted', 'wall_s', 'cpu_s'))
        assert row['mse'] >= 0 and row['wall_s'] > 0 and row['cpu_s'] > 0
        assert row['util'] == row['flops'] / 2**41 and row['adjusted'] == row['mse'] * max(.1, row['util'])
        assert row['ubc'].get('retired') is True
        assert row['ubc'].get('retained_cache_bytes') == row['ubc'].get('live_cache_bytes') == 0
        assert isinstance(row['ynt'], dict)
        predpin = pin(out / f'pred_{rank}.npy')
        predictions[str(rank)] = predpin
        pred = original.load_prediction(predpin)
        parent = original.load_prediction(index['data'][i]['prediction'])
        with np.load(io.BytesIO(checked(c['data'][i]['truth'])), allow_pickle=False) as target:
            truth = target['mc_post']
            assert truth.shape == (16, 1024) and np.isfinite(truth).all()
            truth = truth[15].astype(np.float64)
        assert pred[0].tobytes() == parent[0].tobytes()
        assert float(np.mean((pred[15] - truth)**2)) == row['mse']
        assert float(np.mean((parent[15] - truth)**2)) == base[i]['mse']
        if i < 2:
            previous = old['predictions'][str(rank)]
            assert predpin['sha256'] == previous['sha256'] and predpin['bytes'] == previous['bytes']
            assert checked(predpin) == checked(previous)
            assert all(row[k] == oldrows[i][k] for k in ('rank', 'mse', 'flops', 'util', 'adjusted'))
        replay.append(dict(rank=rank, all16_finite=True, layer0_byte_exact=True,
                           candidate_mse_replayed=True, baseline_mse_replayed=True,
                           known_prefix_prediction_and_bill_exact=True if i < 2 else None))
        del pred, parent, truth
    assert {f.name for f in out.iterdir()} == set(artifacts) | {f'pred_{r}.npy' for r in RANKS}
    m6, m8 = metrics(base[2:], rows[2:]), metrics(base, rows)
    g6 = dict(raw=m6['raw_pct'] < G6['raw_pct_max'], adjusted=m6['adjusted_pct'] < G6['adjusted_pct_max'],
              wins=m6['adjusted_wins'] >= G6['wins_min'],
              max_regression=m6['max_per_network_adjusted_regression_pct'] <= G6['max_regression_pct'])
    g8 = dict(raw=m8['raw_pct'] <= G8['raw_pct_max'], adjusted=m8['adjusted_pct'] <= G8['adjusted_pct_max'],
              wins=m8['adjusted_wins'] >= G8['wins_min'],
              max_regression=m8['max_per_network_adjusted_regression_pct'] <= G8['max_regression_pct'],
              raw_ci=m8['raw_ci95_pct'][1] < G8['raw_ci_upper_max'],
              adjusted_ci=m8['adjusted_ci95_pct'][1] < G8['adjusted_ci_upper_max'])
    passed = all(g6.values()) and all(g8.values())
    return dict(schema='detached-star8-adaptive-followup-assessment-v1', passed=passed,
        retained_evidence_passed=passed, receipt_checks_passed=True, protocol=pin(path), assessor=pin(__file__),
        original_config=p['original_config'], original_assessment=p['original_assessment'],
        original_guard=p['original_guard'], original_prefix=old['artifacts']['prefix2.json'],
        original_admitted=False, original_failure_preserved=True, candidate=p['candidate'], baseline=c['baseline'],
        common=p['common'], driver=p['driver'], guard_source=p['guard_source'], native_guard=guard_pin,
        assessor_source_seal=seal_pin,
        baseline_index=c['baseline_index'], input_pins=p['pins'], artifacts=artifacts, predictions=predictions,
        setup=setup, replay=replay, known2_prediction_bytes_mse_bill_exact=True,
        metrics_new6=m6, gates_new6=g6, metrics_pooled8=m8, gates_pooled8=g8,
        all16_finite=True, layer0_byte_exact=True, historical_validation_reused_not_pristine=True,
        adaptive_followup=True, bootstrap_intervals_descriptive_not_fixed_design=True,
        interpretation=p['interpretation'], targets_read=True, model_executed=False,
        broad_admitted=False, boundary_qualified=False, serving_qualified=False, bank_qualified=False,
        release_qualified=False, submitted=False)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    report = assess(Path(sys.argv[1]).resolve(), sys.argv[2])
    output = D / 'followup_assessment.json'
    once(output, report)
    print(json.dumps(dict(passed=report['passed'], output=pin(output))))
