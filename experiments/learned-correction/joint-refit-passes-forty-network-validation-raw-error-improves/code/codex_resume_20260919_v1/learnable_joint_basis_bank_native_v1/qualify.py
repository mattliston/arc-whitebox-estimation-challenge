from common import *
from records import verify_records
import numpy as np
b=admission();meta=read(HERE/'inputs.json');pins=[]
ref=meta['parent_reference'];coldref=meta['cold_parent_reference']
parent,more=verify_records(Path(ref['result']['path']).parent,ref['source'],'trained','independent',[35,100,35],ref['watch']['path']);pins+=more
parent_cold,more=verify_records(Path(coldref['result']['path']).parent,coldref['source'],'trained','confirmation',[100],coldref['watch']['path']);pins+=more
fixed={}
for name,ref in meta['fixed_basis_references'].items():
    fixed[name],more=verify_records(Path(ref['result']['path']).parent,ref['source'],ref['role'],'control',ref['ranks'],ref['watch']['path']);pins+=more
roles={}
for role in ('disabled','zero','identity_sentinel','sentinel'):
    rows,more=verify_records(HERE/f'native_{role}',b[role],role,'control',[35,100,35],HERE/f'native_{role}_watch.json')
    roles[role]=rows;pins+=more
cold,more=verify_records(HERE/'native_zero_cold100',b['zero'],'zero','control',[100],HERE/'native_zero_cold100_watch.json');pins+=more
for role in ('basis_zero','physical_only','head_only'):
    rows,more=verify_records(HERE/f'native_{role}_cold35',b[role],role,'control',[35],HERE/f'native_{role}_cold35_watch.json')
    roles[role]=rows;pins+=more
extra=[];basis_extra=[]
def exact(a,b):
    assert a['prediction']['sha256']==b['prediction']['sha256']
    assert a['domain']==b['domain'] and a['selectors']==b['selectors']
def basis_cost(row,base):
    delta=row['flops']-base['flops'];assert 0<delta<=100_000_000
    basis_extra.append(dict(rank=row['rank'],cold=row['includes_initialization'],extra_complete_flops=delta))
def total_cost(row,base):
    delta=row['flops']-base['flops'];assert 0<delta<=40_000_000_000
    extra.append(dict(rank=row['rank'],cold=row['includes_initialization'],extra_complete_flops=delta,relative_complete_bill_overhead=delta/base['flops']))
for role,rows in roles.items():
    for index,row in enumerate(rows):
        baseline=parent[index]
        assert row['gnn_layers']==list(range(16)) and row['recurrent_state_ownership_verified'] and row['source_causal_ownership_verified']
        assert [x['layer'] for x in row['source_diagnostics']]==(list(range(8,15)) if role not in ('disabled','head_only') else [])
        if role in ('disabled','zero','basis_zero'):exact(row,baseline)
        if role=='disabled':assert row['flops']==baseline['flops'] and row['multiplier']==baseline['multiplier']
        if role in ('zero','basis_zero'):
            assert row['gnn_states']==roles['disabled'][index]['gnn_states']
            total_cost(row,baseline);basis_cost(row,fixed['zero'][index])
            for s in row['source_diagnostics']:
                assert s['source_relative_variance_max']==s['source_normalized_third_absmax']==s['source_normalized_fourth_absmax']==0
        if role=='identity_sentinel':
            old=fixed['sentinel'][index];exact(row,old)
            assert row['gnn_states']==old['gnn_states'] and row['source_diagnostics']==old['source_diagnostics']
            basis_cost(row,old)
        if role in ('sentinel','identity_sentinel','physical_only','head_only'):
            x=np.load(row['prediction']['path'],allow_pickle=False);y=np.load(baseline['prediction']['path'],allow_pickle=False)
            assert x[:8].tobytes()==y[:8].tobytes() and row['gnn_states'][:8]==roles['disabled'][index]['gnn_states'][:8]
            if role in ('sentinel','identity_sentinel','physical_only'):assert x[8:15].tobytes()!=y[8:15].tobytes()
            if role=='head_only':assert x[:15].tobytes()==y[:15].tobytes() and x[15].tobytes()!=y[15].tobytes()
    if len(rows)>1:
        assert rows[0]['gnn_states']==rows[-1]['gnn_states'] and rows[0]['source_diagnostics']==rows[-1]['source_diagnostics']
physical=np.load(roles['physical_only'][0]['prediction']['path'],allow_pickle=False)
sentinel=np.load(roles['sentinel'][0]['prediction']['path'],allow_pickle=False)
identity=np.load(roles['identity_sentinel'][0]['prediction']['path'],allow_pickle=False)
assert physical[:15].tobytes()==sentinel[:15].tobytes() and physical[15].tobytes()!=sentinel[15].tobytes()
assert identity[8:].tobytes()!=sentinel[8:].tobytes()
exact(cold[0],roles['zero'][1]);exact(cold[0],parent_cold[0])
assert cold[0]['gnn_states']==roles['zero'][1]['gnn_states'] and cold[0]['source_diagnostics']==roles['zero'][1]['source_diagnostics']
total_cost(cold[0],parent_cold[0]);basis_cost(cold[0],fixed['zero_cold100'][0])
write(HERE/'qualification.json',dict(passed=True,full_native_calls=16,disabled_outputs_and_complete_bills_exact=True,
    zero_all16_outputs_domains_selectors_states_exact=True,zero_warm_cold100_exact=True,
    identity_basis_old_sentinel_all16_bytes_exact=True,basis_zero_bank_exact=True,changed_basis_has_observed_effect=True,
    sentinel_first8_layers_and_states_exact=True,physical_and_head_controls_have_separate_effects=True,
    all_same_history_repeats_exact=True,direct_recurrent_and_source_ownership_verified=True,
    own_complete_zero_extra_flops=extra,own_complete_basis_extra_flops=basis_extra,
    all_full_calls_18_object_retirement_qualified=True,source_scale_diagnostics_recorded=True,
    targets_decoded=False,fitted=False,native_derivative_certified=False,training_translation_qualified=False,
    pins=list({p['path']:p for p in pins}.values())))
print(dict(passed=True,full_native_calls=16,zero_extra_flops=extra,basis_extra_flops=basis_extra),flush=True)
