"""Replay the fixed full80 VALID2/8 execution; never execute the estimator."""
from pathlib import Path
import io
import json
import math
import sys

import numpy as np
import validation_common as common
from validation_common import checked, configuration, gates, metrics, once, pin

D = Path(__file__).resolve().parent
SETUP_COUNTS = dict(bound=687, lvfuse=521, psdb_fused=9,
                    pair_block_indices=1, read7_fused=4, ksplit=24)


def load_prediction(record):
    value = np.load(io.BytesIO(checked(record)), allow_pickle=False)
    assert value.shape == (16, 1024) and value.dtype == np.dtype('float64')
    assert np.isfinite(value).all()
    return value


def assess(config_path, digest):
    c, index, base = configuration(config_path, digest)
    assert config_path.parent == D
    assert c['assessor'] == pin(__file__)
    assert c['common'] == pin(common.__file__)
    checked(c['driver'])
    checked(c['validation_plan'])
    checked(c['guard_source'])
    checked(c['research_parent'])
    assert sys.executable == c['runtime'] and np.__version__ == '2.4.6'
    protocol = json.loads(checked(c['protocol']))
    assert c['baseline'] == protocol['official']
    assert c['research_parent'] == protocol['baseline']
    assert c['guard_source'] == protocol['guard']
    assert c['setup_counts'] == SETUP_COUNTS
    assert c['gates2']['all16_finite'] is True and c['gates2']['layer0_exact'] is True
    assert len(c['data']) == 8 and [v['rank'] for v in c['data']] == c['ranks']
    authority = json.loads(checked(protocol['validation']['roster_source']))
    assert c['ranks'] == authority['valid'][:8]
    for item in c['data']:
        rank = str(item['rank'])
        assert item['weights'] == authority['weights'][rank]
        assert item['truth'] == authority['targets'][rank]
    complete = json.loads(checked(c['training_complete']))
    assert complete['completed'] and complete['updates'] == complete['selected_endpoint'] == 80
    exported = json.loads(checked(c['export_manifest']))
    assert exported['passed'] and exported['candidate'] == c['candidate']
    assert exported['baseline'] == c['research_parent']

    out = Path(c['output'])
    guard_path = Path(c['native_guard'])
    assert out.parent == D and out.is_dir()
    assert guard_path == D / 'native_guard.json'
    assert Path(c['assessment_output']) == D / 'native_assessment.json'
    guard_pin = pin(guard_path)
    guard = json.loads(checked(guard_pin))
    expected_command = [c['runtime'], '-B', c['driver']['path'], str(config_path), digest]
    assert guard['command'] == expected_command
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert guard['rss_trigger_bytes'] == c['guard_limits']['rss_bytes']
    assert 0 < guard['max_rss_bytes'] < c['guard_limits']['rss_bytes']
    assert 0 < guard['elapsed_s'] <= c['guard_limits']['seconds']
    assert guard['max_swap_growth_bytes'] == c['guard_limits']['added_swap_bytes'] == 0

    artifacts = {k: pin(out / k) for k in
                 ('manifest.json', 'rows.jsonl', 'prefix2.json', 'completed.json')}
    manifest = json.loads(checked(artifacts['manifest.json']))
    assert manifest['candidate'] == c['candidate'] and manifest['config'] == pin(config_path)
    assert manifest['driver'] == c['driver'] and manifest['requested_ranks'] == c['ranks']
    assert manifest['runtime'] == c['runtime'] and manifest['numpy'] == '2.4.6'
    assert manifest['single_estimator'] is True and manifest['cache_order_matches_baseline'] is True
    assert manifest['thread_env'] == {k: '1' for k in
             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    setup = manifest['setup']
    assert math.isfinite(setup['setup_s']) and 0 < setup['setup_s'] <= 4.1
    labels = [v[0] for v in setup['pb_log']]
    assert len(labels) == len(set(labels))
    counts = {v[0]: v[1] for v in setup['pb_log']}
    assert all(counts.get(k) == v for k, v in SETUP_COUNTS.items())
    assert not any('wall guard' in k or 'restored' in k or 'lost' in k for k in labels)
    rows = [json.loads(s) for s in checked(artifacts['rows.jsonl']).decode().splitlines() if s.strip()]
    assert len(rows) in (2, 8)
    assert [r['rank'] for r in rows] == c['ranks'][:len(rows)]
    prediction_pins = {}
    replay = []
    for i, row in enumerate(rows):
        rank = row['rank']
        assert set(row) == {'rank', 'mse', 'util', 'adjusted', 'flops', 'wall_s', 'cpu_s', 'ubc', 'ynt'}
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert all(math.isfinite(row[k]) for k in ('mse', 'util', 'adjusted', 'wall_s', 'cpu_s'))
        assert row['mse'] >= 0 and row['wall_s'] > 0 and row['cpu_s'] > 0
        assert row['util'] == row['flops'] / 2**41
        assert row['adjusted'] == row['mse'] * max(.1, row['util'])
        assert row['ubc'].get('retired') is True
        assert row['ubc'].get('retained_cache_bytes') == row['ubc'].get('live_cache_bytes') == 0
        assert isinstance(row['ynt'], dict)
        pred_pin = pin(out / f'pred_{rank}.npy')
        prediction_pins[str(rank)] = pred_pin
        pred = load_prediction(pred_pin)
        parent = load_prediction(index['data'][i]['prediction'])
        with np.load(io.BytesIO(checked(c['data'][i]['truth'])), allow_pickle=False) as target:
            truth = target['mc_post']
            assert truth.shape == (16, 1024) and np.isfinite(truth).all()
            truth = truth[15].astype(np.float64)
        assert parent[0].tobytes() == pred[0].tobytes()
        assert float(np.mean((parent[15] - truth)**2)) == base[i]['mse']
        assert float(np.mean((pred[15] - truth)**2)) == row['mse']
        assert base[i]['util'] == base[i]['flops'] / 2**41
        assert base[i]['adjusted'] == base[i]['mse'] * max(.1, base[i]['util'])
        replay.append(dict(rank=rank, all16_finite=True, layer0_byte_exact=True,
                           candidate_mse_replayed=True, baseline_mse_replayed=True))
        del pred, parent, truth
    expected_files = set(artifacts) | {f'pred_{r["rank"]}.npy' for r in rows}
    assert {p.name for p in out.iterdir()} == expected_files

    m2 = metrics(base[:2], rows[:2])
    g2 = gates(m2, c['gates2'])
    prefix = dict(**m2, gates=g2, accuracy_gates_passed=all(g2.values()),
                  resource_qualification_pending=True, admitted=False)
    assert json.loads(checked(artifacts['prefix2.json'])) == prefix
    assert (len(rows) == 8) == all(g2.values()), 'one-process prefix stop/admission mismatch'
    completion = dict(completed=True, n=len(rows), ranks=[r['rank'] for r in rows],
                      config=pin(config_path), candidate=c['candidate'], early_stopped=len(rows) == 2,
                      resource_qualification_pending=True, admitted=False)
    assert json.loads(checked(artifacts['completed.json'])) == completion
    m6 = metrics(base[2:], rows[2:]) if len(rows) == 8 else None
    m8 = metrics(base, rows) if len(rows) == 8 else None
    g8 = gates(m8, c['gates8'], m6) if m8 is not None else None
    admitted = all(g2.values()) and g8 is not None and all(g8.values())
    report = dict(schema='full80-native-validation-assessment-v1', passed=admitted,
        admitted=admitted, valid8_admitted=admitted, failed_stage=None if admitted else
        ('VALID2' if not all(g2.values()) else 'VALID8'), receipt_checks_passed=True,
        resource_gates_passed=True, stage='VALID8' if len(rows) == 8 else 'VALID2',
        config=pin(config_path), assessor=pin(__file__), common=c['common'], driver=c['driver'],
        candidate=c['candidate'], baseline=c['baseline'], research_parent=c['research_parent'],
        protocol=c['protocol'], validation_plan=c['validation_plan'], training_complete=c['training_complete'],
        export_manifest=c['export_manifest'], baseline_index=c['baseline_index'], native_guard=guard_pin,
        artifacts=artifacts, predictions=prediction_pins, setup=setup, replay=replay,
        metrics2=m2, gates2=g2, metrics_additional6=m6, metrics8=m8, gates8=g8,
        all16_finite=True, layer0_byte_exact=True, single_process_prefix_verified=True,
        historical_validation_reused_not_pristine=True, targets_read=True, model_executed=False,
        broad_admitted=False, boundary_qualified=False, serving_qualified=False,
        bank_qualified=False, release_qualified=False, submitted=False)
    return c, report


if __name__ == '__main__':
    assert len(sys.argv) == 3
    config_path = Path(sys.argv[1]).resolve()
    c, result = assess(config_path, sys.argv[2])
    once(c['assessment_output'], result)
    print(json.dumps(dict(passed=result['passed'], admitted=result['admitted'],
                          failed_stage=result['failed_stage'], output=pin(c['assessment_output']))))
