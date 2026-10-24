"""Offline exact-component TRAIN8 accounting. Root executes after native6 retires."""
from pathlib import Path
import hashlib
import io
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
H2, P2 = HERE.parent, HERE.parent.parent
NATIVE = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
OLD = H2 / 'hill_climb_exact_bundle_v1'
PROTOCOL_SHA = '086b976446f9f18ff9c5946c35af12c829e9ee84f76584d98195b2d00d9a05d6'
EVAL_SHA = 'ba52c5a930ce46f9e8db1a84f0f88e7d1cb535cb2ea4b460a85eddff1b17bea1'
PROBE_SHA = '8acc3e338ca5f991f778edaf0376bc8265ff69b330137c5bba4d079766ca8af5'
BLOCKS = [[35, 100], [36, 45, 52, 58, 64, 70]]


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    raw = Path(record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256'] and len(raw) == record['bytes']
    return raw


def read(path):
    record = pin(path)
    return json.loads(checked(record)), record


def panel(index, candidate, protocol):
    source = protocol['source' if candidate else 'parent']
    tag = protocol['candidate_tags' if candidate else 'base_tags'][index]
    size = (2, 6)[index]
    evaluator = H2 / 'native_eval.py'
    guard_path = H2 / 'hill_climb_star_projection32_v1' / f'train{size}_guard.json'
    if candidate:
        guard_path = OLD / 'native2_guard.json' if index == 0 else HERE / 'native6_guard.json'
        if index == 0:
            evaluator = H2 / 'hill_climb_uvp_base_bcache_probe_v1/native_probe.py'
    evaluator_pin = pin(evaluator)
    assert evaluator_pin['sha256'] == (PROBE_SHA if candidate and index == 0 else EVAL_SHA)
    guard, guard_pin = read(guard_path)
    assert guard['command'] == [str(NATIVE), '-B', str(evaluator), source['path'],
                               ','.join(map(str, BLOCKS[index])), tag]
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and guard['remaining_rows'] == []
    assert guard['max_swap_growth_bytes'] == 0
    assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] <= protocol['rss_bytes']
    assert 0 < guard['elapsed_s'] <= (150 if index == 0 else protocol['guard_seconds'])
    manifest, manifest_pin = read(H2 / tag / 'manifest.json')
    assert manifest['candidate'] == source['path'] and manifest['sha256'] == source['sha256']
    assert manifest['ranks'] == BLOCKS[index] and manifest['runtime'] == str(NATIVE)
    assert manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == {k: '1' for k in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    rows_pin = pin(H2 / tag / 'rows.jsonl')
    rows = [json.loads(line) for line in checked(rows_pin).decode().splitlines()]
    assert [row['rank'] for row in rows] == BLOCKS[index]
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ('mse', 'adjusted', 'util', 'cpu_s', 'wall_s'))
        assert row['mse'] > 0 and type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert row['util'] == row['flops']/2**41
        assert math.isclose(row['adjusted'], row['mse']*max(.1, row['util']), rel_tol=1e-14, abs_tol=0)
    return dict(tag=tag, source=source, evaluator=evaluator_pin, guard=guard_pin,
                manifest=manifest_pin, rows=rows_pin), rows, guard


def pooled(base, candidate):
    bill0, bill1 = sum(r['flops'] for r in base), sum(r['flops'] for r in candidate)
    raw0, raw1 = math.fsum(r['mse'] for r in base), math.fsum(r['mse'] for r in candidate)
    adj0, adj1 = math.fsum(r['adjusted'] for r in base), math.fsum(r['adjusted'] for r in candidate)
    return dict(n=len(base), base_total_flops=bill0, candidate_total_flops=bill1,
                saved_flops=bill0-bill1, bill_pct=100*(bill1/bill0-1),
                raw_pct=100*(raw1/raw0-1), adjusted_pct=100*(adj1/adj0-1),
                base_mean_adjusted=adj0/len(base), candidate_mean_adjusted=adj1/len(base),
                each_bill_no_regression=all(c['flops'] <= b['flops'] for b, c in zip(base, candidate)),
                pooled_bill_saving_positive=bill1 < bill0)


def main():
    output = HERE / 'assessment.json'
    assert not output.exists()
    protocol, protocol_pin = read(HERE / 'protocol.json')
    assert protocol_pin['sha256'] == PROTOCOL_SHA
    assert [protocol['train2'], protocol['train6']] == BLOCKS
    assert protocol['base_tags'] == ['hill_climb_star_projection32_train2_v1', 'hill_climb_star_projection32_train6_v1']
    assert protocol['candidate_tags'] == ['hill_climb_exact_bundle_native2_v1', 'hill_climb_exact_bundle_native6_v1']
    assert protocol['source']['sha256'] == '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
    assert protocol['parent']['sha256'] == 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
    for record in [protocol['source'], protocol['parent'], *protocol['evidence']]:
        checked(record)
    retained, retained_pin = read(OLD / 'retained_bundle.json')
    assert retained['passed'] and retained['retained']
    assert retained['candidate'] == protocol['source'] and retained['parent'] == protocol['parent']
    panels, rows, resources = [[], []], [[], []], []
    for index in range(2):
        for side in range(2):
            item, block_rows, guard = panel(index, bool(side), protocol)
            panels[side].append(item)
            rows[side].extend(block_rows)
            resources.append(guard)
    # All four complete source/guard/roster checks precede hashes and array decoding.
    for side in panels:
        for index, item in enumerate(side):
            item['predictions'] = [pin(H2 / item['tag'] / f'pred_{rank}.npy') for rank in BLOCKS[index]]
    equality = []
    for index in range(2):
        for rank, bp, cp in zip(BLOCKS[index], panels[0][index]['predictions'], panels[1][index]['predictions']):
            b = np.load(io.BytesIO(checked(bp)), allow_pickle=False)
            c = np.load(io.BytesIO(checked(cp)), allow_pickle=False)
            assert b.shape == c.shape == (16, 1024) and b.dtype == c.dtype == np.dtype('float64')
            assert np.isfinite(b).all() and np.isfinite(c).all()
            equality.append(dict(rank=rank, all16_byte_exact=b.tobytes() == c.tobytes(), all16_finite=True))
    per_network = []
    for b, c, eq in zip(rows[0], rows[1], equality):
        assert b['rank'] == c['rank'] == eq['rank']
        per_network.append(dict(**eq, raw_mse_equal=b['mse'] == c['mse'],
            base_mse=b['mse'], candidate_mse=c['mse'], base_flops=b['flops'], candidate_flops=c['flops'],
            saved_flops=b['flops']-c['flops'], bill_pct=100*(c['flops']/b['flops']-1),
            adjusted_pct=100*(c['adjusted']/b['adjusted']-1), bill_no_regression=c['flops'] <= b['flops']))
    metrics = pooled(rows[0], rows[1])
    decisions = dict(all16_byte_exact=all(x['all16_byte_exact'] for x in equality),
        same_raw_mse=all(x['raw_mse_equal'] for x in per_network),
        each_bill_no_regression=metrics['each_bill_no_regression'],
        pooled_bill_saving_positive=metrics['pooled_bill_saving_positive'])
    assert set(decisions) == set(protocol['admission']) and all(protocol['admission'].values())
    passed = all(decisions.values())
    result = dict(assessment_completed=True, passed=passed, retained_exact_component=passed,
        protocol=protocol_pin, assessor=pin(__file__), candidate=protocol['source'], parent=protocol['parent'],
        inherited_component=retained_pin, inherited_evidence=protocol['evidence'], panels=panels,
        ranks=BLOCKS[0]+BLOCKS[1], process_blocks=BLOCKS, metrics=metrics,
        native2=pooled(rows[0][:2], rows[1][:2]), additional6=pooled(rows[0][2:], rows[1][2:]),
        decisions=decisions, failures=[k for k, v in decisions.items() if not v], per_network=per_network,
        max_group_rss_bytes=max(g['max_rss_bytes'] for g in resources),
        all_guards_retired_zero_swap=True, predictions_compared=8, prediction_values_compared=8*16*1024,
        bootstrap_performed=False, target_arrays_decoded=False, source_or_coefficients_changed=False,
        worker20_qualified=False, broad_validation=False, release_qualified=False, submitted=False,
        scope='Native TRAIN8 exact-output component cost accounting only; no automatic release or aggregate forecast.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=passed, retained_exact_component=passed, metrics=metrics, decisions=decisions)))


if __name__ == '__main__':
    main()
