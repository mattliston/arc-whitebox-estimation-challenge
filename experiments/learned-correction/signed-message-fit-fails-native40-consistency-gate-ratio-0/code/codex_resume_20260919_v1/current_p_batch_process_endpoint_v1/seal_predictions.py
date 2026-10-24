from common import *
from records import verify_run

build = admission()
done('qualify_null')
null = read(HERE/'native_null_qualification.json')
assert null['passed'] and null['all_retirements_qualified']
inputs = read(HERE/'inputs.json')
pins = [pin(HERE/n) for n in ['sources_sealed.json','inputs.json','build.json','native_null_qualification.json']]
pins += null['pins'] + [pin(F/'fit_complete.json'),pin(F/'controller_result.json')]
roles = {}
for role in ['parent','trained']:
    name = f'native_{role}'
    records,more = verify_run(name,'validation',role,inputs['valid']+inputs['valid'][:1])
    assert len(records)==41
    pins += more + qualify_retirements(name,records)
    roles[role] = dict(result=pin(HERE/name/'result.json'),
        source=build['parent' if role=='parent' else 'candidate'],watch=pin(HERE/f'{name}_watch.json'))
write(HERE/'native40_predictions_sealed.json',dict(completed=True,before_validation_targets=True,
    scored_ranks=inputs['valid'],roles=roles,pins=list({p['path']:p for p in pins}.values()),
    own_complete_bills=True,repeats_excluded=True,all_retirements_qualified=True,
    comparison_gate=dict(adjusted_ratio_strictly_below=1,wins_at_least=24)))
print(dict(sealed=True,parent_calls=41,trained_calls=41,validation_targets_decoded=False),flush=True)
