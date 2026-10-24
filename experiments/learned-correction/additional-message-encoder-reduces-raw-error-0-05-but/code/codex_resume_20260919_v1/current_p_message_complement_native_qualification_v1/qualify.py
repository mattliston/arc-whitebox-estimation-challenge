from common import *
from records import verify_records
import numpy as np

b=admission();meta=read(HERE/'inputs.json');pins=[]
ref=meta['parent_reference'];coldref=meta['cold_parent_reference']
parent,more=verify_records(Path(ref['result']['path']).parent,ref['source'],'trained','independent',[35,100,35],ref['watch']['path']);pins+=more
parent_cold,more=verify_records(Path(coldref['result']['path']).parent,coldref['source'],'trained','confirmation',[100],coldref['watch']['path']);pins+=more
roles={}
for role in ('disabled','zero','sentinel'):
    rows,more=verify_records(HERE/f'native_{role}',b[role],role,'control',[35,100,35],HERE/f'native_{role}_watch.json')
    pins+=more;roles[role]=rows
cold,more=verify_records(HERE/'native_zero_cold100',b['zero'],'zero','control',[100],HERE/'native_zero_cold100_watch.json');pins+=more
for reference in (ref,coldref):
    assert reference['source']==pin(PARENT)
extra=[]
for role,rows in roles.items():
    for index,(baseline,row) in enumerate(zip(parent,rows)):
        assert row['gnn_layers']==list(range(16)) and row['recurrent_state_ownership_verified']
        assert [r['layer'] for r in row['gnn_states']]==list(range(16))
        if role in ('disabled','zero'):
            assert row['prediction']['sha256']==baseline['prediction']['sha256']
            assert row['domain']==baseline['domain'] and row['selectors']==baseline['selectors']
        if role=='disabled':
            assert row['flops']==baseline['flops'] and row['multiplier']==baseline['multiplier']
        if role=='zero':
            assert row['gnn_states']==roles['disabled'][index]['gnn_states']
            delta=row['flops']-baseline['flops'];assert 0<delta<=12_000_000_000
            extra.append(dict(rank=row['rank'],cold=row['includes_initialization'],extra_complete_flops=delta,
                relative_complete_bill_overhead=delta/baseline['flops']))
        if role=='sentinel':
            x=np.load(row['prediction']['path'],allow_pickle=False);y=np.load(baseline['prediction']['path'],allow_pickle=False)
            assert x[0].tobytes()==y[0].tobytes() and x[1:].tobytes()!=y[1:].tobytes()
            assert row['gnn_states'][0]==roles['disabled'][index]['gnn_states'][0]
    assert rows[0]['gnn_states']==rows[-1]['gnn_states']
assert cold[0]['prediction']['sha256']==roles['zero'][1]['prediction']['sha256']
assert cold[0]['domain']==roles['zero'][1]['domain'] and cold[0]['selectors']==roles['zero'][1]['selectors']
assert cold[0]['gnn_states']==roles['zero'][1]['gnn_states']
assert cold[0]['gnn_layers']==list(range(16)) and cold[0]['recurrent_state_ownership_verified']
delta=cold[0]['flops']-parent_cold[0]['flops'];assert 0<delta<=12_000_000_000
extra.append(dict(rank=100,cold=True,extra_complete_flops=delta,
    relative_complete_bill_overhead=delta/parent_cold[0]['flops']))
for name in ['native_disabled','native_zero','native_zero_cold100','native_sentinel']:done(name)
unit=read(SMALL/'unit_result.json');assert unit['passed']
write(HERE/'qualification.json',dict(passed=True,full_native_calls=10,disabled_outputs_and_complete_bills_exact=True,
    zero_all16_outputs_domains_selectors_states_exact=True,zero_warm_cold100_exact=True,
    sentinel_layer0_exact_and_later_outputs_changed=True,all_same_history_repeats_exact=True,
    direct_recurrent_state_ownership_verified=True,own_complete_zero_extra_flops=extra,
    all_full_calls_18_object_retirement_qualified=True,targets_decoded=False,fitted=False,
    native_derivative_certified=False,training_translation_qualified=False,
    unit_result=pin(SMALL/'unit_result.json'),pins=list({p['path']:p for p in pins}.values())))
print(dict(passed=True,full_native_calls=10,zero_extra_flops=extra),flush=True)
