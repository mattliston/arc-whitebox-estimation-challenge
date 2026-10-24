from common import *
from records import verify_run
admission()
inputs = read(HERE / 'inputs.json')
warm, pins = verify_run('independent_trained', 'independent', 'trained', [35,100,35])
cold, other_pins = verify_run('confirmation_100', 'confirmation', 'trained', [100])
pins += other_pins
comparisons = []
for actual, expected in zip(warm + cold, inputs['warm_reference'] + [inputs['cold100_reference']]):
    assert actual['rank'] == expected['rank'] and actual['index'] == expected['index']
    assert actual['includes_initialization'] == expected['includes_initialization']
    assert actual['weights'] == expected['weights']
    assert actual['prediction']['sha256'] == expected['prediction']['sha256']
    assert actual['domain'] == expected['domain'] and actual['selectors'] == expected['selectors']
    assert 0 < actual['flops'] - expected['flops'] < .002 * expected['flops']
    comparisons.append(dict(rank=actual['rank'], index=actual['index'],
        prediction=actual['prediction'], all16_bytes_exact=True, all_domains_and_selectors_exact=True,
        candidate_flops=actual['flops'], parent_flops=expected['flops'],
        added_flops=actual['flops']-expected['flops'], cost_ratio=actual['flops']/expected['flops']))
assert warm[1]['prediction']['sha256'] == cold[0]['prediction']['sha256']
write(HERE / 'native_qualification.json', dict(passed=True, comparisons=comparisons,
    full_native_calls=4, own_bills_measured=True, cross_role_cost_equality_assumed=False,
    warm_cold100_output_exact=True, targets_decoded=False, candidate=pin(SOURCE), parent=pin(PARENT),
    sources=pin(HERE / 'sources_sealed.json'), native_artifacts=pins))
print(dict(passed=True, comparisons=comparisons), flush=True)
