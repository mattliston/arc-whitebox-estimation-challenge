"""Verify completed acquisition artifacts before any target decoding."""
from common import *


def verify_run(name, stage, role, ranks):
    admitted = admission()
    meta = read(HERE/'inputs.json')
    watch_path = HERE/f'{name}_watch.json'
    clean_watch(watch_path)
    path = HERE/name/'result.json'
    result = read(path)
    source = admitted['bank'] if role == 'parent' else admitted['candidate']
    assert result['completed'] and result['source'] == source
    assert result['stage'] == stage and result['role'] == role
    assert result['all_16_domains_and_45_selectors_qualified'] and result['passive_observers_restored']
    assert result['owned_fnp_input_and_closed_context_lifecycle']
    assert not result['truth_accessed'] and result['passive_extra_fnp_ops'] == 0
    rows = result['records']
    assert [r['rank'] for r in rows] == ranks and result['full_estimator_calls'] == len(ranks)
    pins = [pin(path),pin(watch_path),source]
    for index,r in enumerate(rows):
        assert r['index'] == index and r['includes_initialization'] == (index == 0)
        assert r['weights'] == meta['weights'][str(r['rank'])]
        assert r['numeric_cache_retired'] and r['passive_extra_fnp_ops'] == 0 and not r['truth_accessed']
        assert set(r['domain']) == {str(l) for l in range(16)}
        assert set(r['selectors']) == {f'{l}:{s}' for l in range(15) for s in range(3)}
        assert all(d['finite'] and d['minimum'] >= 1e-10 for d in r['domain'].values())
        for item in r['selectors'].values():
            assert item['floor'] == .01 and 0 <= item['count'] <= 1024 and item['count'] % 32 == 0
        for key in ['prediction','bill','weights','input_bill','input_reset','lifecycle']:
            check(r[key]); pins.append(r[key])
        pins.append(pin(HERE/name/f'call_{index:02d}.json'))
        bill = read(r['bill']['path'])
        assert bill['flops'] == r['flops'] and bill['multiplier'] == r['multiplier'] == max(.1,r['flops']/2**41)
        assert 0 < r['flops'] <= 2**41 and bill['wall_s'] == r['wall_s'] < 120
        assert bill['includes_initialization'] == r['includes_initialization'] and bill['complete_counted_predict']
        assert not bill['initialization_subtracted'] and bill['extra_participant_array_operations'] == 0
        assert bill['input_flops_separate'] == 16*1024*1024
        ingress = read(r['input_bill']['path'])
        assert ingress['flops'] == 16*1024*1024 and not ingress['charged_to_participant']
        assert ingress['exact_bytes'] and ingress['owned_fnp_float32'] and ingress['independent_buffers'] and ingress['context_closed']
        assert ingress['weight'] == r['weights'] and ingress['wall_s'] < 120
        for reset_key,bill_key in [('input_reset','input_bill'),('lifecycle','bill')]:
            reset = read(r[reset_key]['path'])
            assert reset['complete'] and reset['context_released'] and reset['closed_context_bill_unchanged_by_reset']
            assert reset['after_context_exit'] and not reset['candidate_state_reset']
            assert reset['saved_bill_sha256'] == r[bill_key]['sha256'] and reset['saved_bill'] == r[bill_key]['path']
            assert reset['rank'] == r['rank'] and reset['index'] == index
    if len(ranks) > 1:
        assert result['repeat_output_domain_selector_bytes_exact'] and ranks[-1] == ranks[0]
        assert rows[-1]['prediction']['sha256'] == rows[0]['prediction']['sha256']
        assert rows[-1]['domain'] == rows[0]['domain'] and rows[-1]['selectors'] == rows[0]['selectors']
    return rows, pins


def verify_retired_run(name,stage,role,ranks):
    rows,pins = verify_run(name,stage,role,ranks)
    return rows,pins+qualify_retirements(name,rows)
