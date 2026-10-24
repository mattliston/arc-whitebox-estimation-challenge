from endpoint_common import *
endpoint_verify()
inputs, build = read(HERE / 'inputs.json'), read(HERE / 'build.json')
pins = [pin(HERE / n) for n in ('sources_sealed.json', 'endpoint_sources_sealed.json', 'inputs.json',
    'build.json', 'native_null_qualification.json', 'qualify_null_watch.json', 'fit_complete.json')]
roles = {}
for role, source in (('parent', build['parent']), ('trained', build['candidate'])):
    name = f'native_{role}'
    done(name)
    receipt = HERE / name / 'result.json'
    result = read(receipt)
    assert result['completed'] and result['source'] == source and result['role'] == role
    assert result['all_16_domains_and_45_selectors_qualified'] and result['repeat_output_domain_selector_bytes_exact']
    assert result['passive_observers_restored'] and not result['truth_accessed'] and result['passive_extra_fnp_ops'] == 0
    assert [r['rank'] for r in result['records']] == inputs['valid'] + inputs['valid'][:1]
    assert len(result['records']) == result['full_estimator_calls'] == 41
    for i, r in enumerate(result['records']):
        assert r['index'] == i and r['includes_initialization'] == (i == 0)
        assert r['weights'] == inputs['weights'][str(r['rank'])]
        assert r['numeric_cache_retired'] and r['input_flops_separate'] == r['passive_extra_fnp_ops'] == 0
        bill = read(r['bill']['path'])
        assert bill['flops'] == r['flops'] and bill['multiplier'] == r['multiplier']
        assert bill['includes_initialization'] == r['includes_initialization'] and bill['complete_counted_predict']
        assert not bill['initialization_subtracted'] and bill['wall_s'] < 120 and 0 < r['flops'] <= 2**41
        assert set(r['domain']) == {str(l) for l in range(16)}
        assert set(r['selectors']) == {f'{l}:{s}' for l in range(15) for s in range(3)}
        assert all(d['finite'] and d['minimum'] >= 1e-10 for d in r['domain'].values())
        for k in ('prediction', 'bill', 'weights'):
            check(r[k])
            pins.append(r[k])
    assert result['records'][0]['prediction']['sha256'] == result['records'][-1]['prediction']['sha256']
    assert result['records'][0]['domain'] == result['records'][-1]['domain']
    assert result['records'][0]['selectors'] == result['records'][-1]['selectors']
    pins += [source, pin(receipt), pin(HERE / f'{name}_watch.json')]
    roles[role] = dict(result=pin(receipt), source=source, watch=pin(HERE / f'{name}_watch.json'))
write(HERE / 'native40_predictions_sealed.json', dict(completed=True, before_validation_targets=True,
    scored_ranks=inputs['valid'], roles=roles, pins=list({p['path']: p for p in pins}.values()),
    own_complete_bills=True, repeats_excluded=True, comparison_gate=dict(adjusted_ratio_strictly_below=1, wins_at_least=24)))
print(dict(sealed=True, parent_calls=41, trained_calls=41, validation_targets_decoded=False), flush=True)
