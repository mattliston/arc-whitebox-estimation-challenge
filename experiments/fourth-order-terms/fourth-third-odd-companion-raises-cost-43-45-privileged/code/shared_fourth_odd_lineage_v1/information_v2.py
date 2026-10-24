"""Fixed own-cold scalar-oracle screen, with capture sealed before targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from oracle import clipped_scalar_oracle

HERE = Path(__file__).resolve().parent
D = HERE.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    sources = read(HERE/'sources_sealed_v2.json')
    for name, value in sources['artifacts_sha256'].items():
        assert sha(HERE/name) == value, name
    assert read(HERE/'unit_v2/result.json')['passed']
    for name in ('unit_v2_watch.json', 'qualification_v2_watch.json', 'cold100_v2_watch.json'):
        w = read(HERE/name)
        assert w['completed'] and w['whole_group_retired']
        assert w['max_rss_bytes'] < 8_000_000_000 and w['max_swap_growth_bytes'] == 0
    qualification = read(HERE/'qualification_v2/completed.json')
    cold = read(HERE/'cold100_v2/completed.json')
    assert qualification['repeat_byte_exact']
    assert qualification['parent_all16_and_flops_exact'] and cold['parent_all16_and_flops_exact']
    assert [r['rank'] for r in qualification['records']] == [35, 100, 35]
    assert [r['rank'] for r in cold['records']] == [100]
    for rec in qualification['records']+cold['records']:
        assert sha(rec['capture']) == rec['capture_sha256']
        assert rec['bindings_restored'] and rec['numeric_cache_retired']
        assert rec['max_rss_bytes'] < 8_000_000_000
    with np.load(qualification['records'][1]['capture']) as warm, np.load(cold['records'][0]['capture']) as other:
        assert warm.files == other.files
        for name in warm.files:
            assert warm[name].dtype == other[name].dtype and warm[name].tobytes() == other[name].tobytes()
    pairs = []
    pins = {str(HERE/'sources_sealed_v2.json'):sha(HERE/'sources_sealed_v2.json')}
    for current, parent_dir in ((qualification['records'][0], 'dev'), (cold['records'][0], 'cold100')):
        parent_path = D/'lazy_rotation_v1'/parent_dir/'result.json'
        parent = read(parent_path)['records'][0]
        assert current['rank'] == parent['rank'] and current['includes_initialization'] and parent['includes_initialization']
        assert current['weights_sha256'] == parent['weights_sha256']
        assert current['parent_flops'] == parent['flops']
        assert current['total_flops'] == current['parent_flops']+current['feature_flops']
        assert sha(parent['output']) == parent['output_sha256']
        with np.load(current['capture']) as capture:
            assert capture['prediction'].tobytes() == np.load(parent['output']).tobytes()
            arrays = {k:capture[k].copy() for k in capture.files}
        pairs.append((current, parent, arrays))
        pins.update({current['capture']:sha(current['capture']), str(parent_path):sha(parent_path),
                     parent['output']:sha(parent['output'])})
    for name in ('qualification_v2/completed.json', 'cold100_v2/completed.json', 'unit_v2/result.json',
                 'unit_v2_watch.json', 'qualification_v2_watch.json', 'cold100_v2_watch.json'):
        pins[str(HERE/name)] = sha(HERE/name)
    with (HERE/'predictions_v2_sealed_before_targets.json').open('x') as stream:
        json.dump(dict(pins=pins, all16_parent_bytes_exact=True, repeated_features_exact=True,
                       warm_cold100_all_arrays_exact=True, truth_accessed=False,
                       complete_cold_bills=[dict(rank=c['rank'], total=c['total_flops'],
                                                feature=c['feature_flops'], parent=p['flops'])
                                            for c, p, unused in pairs]), stream, indent=2)
    rows = []
    output_arrays = {}
    for current, parent, arrays in pairs:
        rank = current['rank']
        path = D.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        expected = {35:'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',
                    100:'637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
        assert sha(path) == expected[rank]
        truth = np.load(path)['mc_post'][-1].astype(np.float64)
        p = arrays['prediction'][-1].astype(np.float64)
        oracle, ideal, served = clipped_scalar_oracle(p, arrays['feature'], truth)
        parent_raw = float(np.mean((p-truth)**2))
        parent_adjusted = parent_raw*max(.1, parent['flops']/2**41)
        billed = current['total_flops']+1_000_000
        multiplier = max(.1, billed/2**41)
        oracle.update(ideal_adjusted=oracle['ideal_raw']*multiplier,
                      served_adjusted=oracle['served_raw']*multiplier,
                      rounding_allowance_lower_adjusted=oracle['rounding_allowance_lower_raw']*multiplier)
        ratio = oracle['served_adjusted']/parent_adjusted
        lower_ratio = oracle['rounding_allowance_lower_adjusted']/parent_adjusted
        passed = ratio <= .95
        numerical_check_same_conclusion = (lower_ratio <= .95) == passed
        assert numerical_check_same_conclusion, (rank, ratio, lower_ratio)
        rows.append(dict(rank=rank, truth_sha256=sha(path), parent_raw=parent_raw,
                         parent_flops=parent['flops'], parent_adjusted=parent_adjusted,
                         capture_flops=current['total_flops'], feature_flops=current['feature_flops'],
                         readout_reserve_flops=1_000_000, complete_flops=billed, multiplier=multiplier,
                         cost_ratio=billed/parent['flops'], oracle=oracle,
                         served_adjusted_ratio=ratio, rounding_allowance_lower_ratio=lower_ratio,
                         gate_passed=passed, numerical_check_same_conclusion=numerical_check_same_conclusion))
        output_arrays[f'ideal_{rank}'] = ideal
        output_arrays[f'served_{rank}'] = served
    target = HERE/'oracle_predictions_v2.npz'
    assert not target.exists()
    np.savez(target, **output_arrays)
    result = dict(rows=rows, information_gate_passed=all(r['gate_passed'] for r in rows),
                  target_conditioned_oracle=True, deployable_fit=False, fresh_generalization_evidence=False,
                  all16_parent_bytes_and_base_bills_exact=True, warm_cold100_all_arrays_exact=True,
                  predictions_sha256=sha(target), source_sha256=sha(__file__),
                  oracle_source_sha256=sha(HERE/'oracle.py'), no_interval_certificate_claimed=True)
    with (HERE/'information_result_v2.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
