"""Freeze one exact static DAG schedule; no runtime optimization or model fit."""
import ast, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=HERE.parent.parent
K=RUNS/'fable_oct19_kc_v1';D=RUNS/'astra_oct20_bill_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
price=json.loads((HERE/'PRICE_METADATA_v2.json').read_text())
assert price['projected_saved_calls_beyond_compact']==270
def immutable(x):return tuple(map(immutable,x)) if isinstance(x,list) else x
schedules={(r['record'],r['kind']):immutable(r['groups']) for r in price['plans']}
compact=D/'tail_smdirect_compact_ap_complete_v1.py'
assert sha(compact)=='7de468b98606dbd3aef619ebfad35559b54775b893e86b06bc6194d0582ee05a'
text=compact.read_text().replace('o20_direct_compact_ap_v1','o20_direct_smdag_v1')
text+='\n# Static exact DAG schedule: every node keeps its original operator and ordered operands.\n'
text+='_SMDAG_FIXED_GROUPS = '+repr(schedules)+'\n'
text+='''
assert not getattr(_SMX_ENV, '_sm40_group_cache', {})
assert not getattr(_ASTRA_SM40_ENV, '_sm40_group_cache', {})
_SMDAG_ORIGINAL_PLANS = {}
for (_smdag_name, _smdag_kind), _smdag_groups in _SMDAG_FIXED_GROUPS.items():
    _smdag_record = _SMX_RECORDS[_smdag_name]
    _smdag_old = _smdag_record['prepared'][_smdag_kind]
    _SMDAG_ORIGINAL_PLANS[_smdag_name, _smdag_kind] = _smdag_old
    _smdag_old_nodes = {r: (tuple(a), tag[1:]) for tag, items in _smdag_old[1] for r, a in items}
    _smdag_new_nodes = {r: (tuple(a), tag[1:]) for tag, items in _smdag_groups for r, a in items}
    assert _smdag_old_nodes == _smdag_new_nodes
    assert sum(len(items) for tag, items in _smdag_old[1]) == sum(len(items) for tag, items in _smdag_groups)
    _smdag_have = set(_smdag_old[0])
    for _smdag_tag, _smdag_items in _smdag_groups:
        assert all(a in _smdag_have for r, args in _smdag_items for a in args)
        _smdag_have.update(r for r, args in _smdag_items)
    assert set(_smdag_old[2]) <= _smdag_have
    # The existing record's prepared DICTIONARY is shared with every existing
    # kernel.circuits. Replacing its L/R tuple is visible to both group-view
    # allocation and evaluation. Inputs, output positions and use counts stay.
    _smdag_record['prepared'][_smdag_kind] = (_smdag_old[0], _smdag_groups, _smdag_old[2], _smdag_old[3])
_O20_SMDIRECT_PLANS = {
    id(record['prepared'][kind]): record['prepared'][kind]
    for record in _SMX_RECORDS.values() for kind in ('L', 'R')
}
'''
tail=HERE/'tail_smdag_schedule_v1.py'
ast.parse(text)
with tail.open('x') as f:f.write(text)
parents={
 'science':(K/'candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py','434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'),
 'release':(K/'candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py','660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200')}
nogov=K/'agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py'
outputs={}
for role,(parent,pin) in parents.items():
 assert sha(parent)==pin
 target=HERE/f'candidate_smdag_{role}_v1.py'
 candidate=parent.read_text()+'\n'+text+'\n'+nogov.read_text()
 assert ast.unparse(ast.parse(candidate).body[-1])=='_FAST_RESCUE_FRAC = {}'
 with target.open('x') as f:f.write(candidate)
 outputs[role]=dict(path=str(target),sha256=sha(target),parent_path=str(parent),parent_sha256=pin)
r=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),outputs=outputs,
       tail=dict(path=str(tail),sha256=sha(tail)),metadata_price=dict(path=str(HERE/'PRICE_METADATA_v2.json'),sha256=sha(HERE/'PRICE_METADATA_v2.json')),
       no_runtime_search=True,no_model_fit=True,no_scalar_arithmetic_change=True,no_output_permutation=True,
       no_fg_pair_change=True,no_new_numeric_owner=True,projected_bill_delta=0,
       limitation='Frozen source only; official current graph parity, tariff, cache, import, shared-alias proof and own native qualification pending.')
with (HERE/'BUILD_v1.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
