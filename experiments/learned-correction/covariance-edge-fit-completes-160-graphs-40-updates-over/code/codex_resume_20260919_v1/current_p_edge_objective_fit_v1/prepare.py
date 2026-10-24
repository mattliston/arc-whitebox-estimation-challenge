import ast
from fit_common import *

assert not numerical_workers()
qresult=read(Q/'result.json');assert qresult['completed'] and qresult['passed'] and qresult['error'] is None
for key in ('unit','full'):check(qresult[key])
full=read(Q/'full_result.json');assert full['passed'] and full['real_adam_boundary_exact'] and full['next_synthetic_update_exact']
assert not full['targets_decoded'] and not full['production_payload_exported']
qw=[]
for name in ('unit','full'):
    w=read(Q/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    qw.append(w)
for p in read(Q/'sources_sealed.json')['pins']:check(p)
for p in full['records']:check(p)
io=(Q/'state_io.py').read_text();assert io.count('from common import *')==1
newio=io.replace('from common import *','from fit_common import *')
assert newio.replace('from fit_common import *','from common import *')==io
with (HERE/'state_io.py').open('x') as f:f.write(newio)
with (HERE/'guard.py').open('xb') as f:f.write((Q/'guard.py').read_bytes())
old_inputs=D/'current_full_lower_p_fit_v1/inputs.json'
with (HERE/'inputs.json').open('xb') as f:f.write(old_inputs.read_bytes())
check(read(Q/'initial_rng.json')['payload'])
with (HERE/'initial_rng.npz').open('xb') as f:f.write((Q/'initial_rng.npz').read_bytes())
write(HERE/'initial_rng.json',dict(payload=pin(HERE/'initial_rng.npz'),origin=pin(Q/'initial_rng.npz'),
    seed=9409213,used_for_initialization_only=True,qualified_synthetic_parameter_state_not_imported=True))
inputs=read(HERE/'inputs.json')
assert len(inputs['train'])==160 and len(inputs['valid'])==40 and not set(inputs['train'])&set(inputs['valid'])
data=list(inputs['weights'].values())+list(inputs['targets'].values())
for p in data:check(p)
write(HERE/'admission.json',dict(passed=True,qualification=pin(Q/'result.json'),qualification_full=pin(Q/'full_result.json'),
    qualification_source_seal=pin(Q/'sources_sealed.json'),qualification_guards=[pin(Q/f'{n}_watch.json') for n in ('unit','full')],
    qualification_peak_rss_bytes=max(w['max_rss_bytes'] for w in qw),zero_added_swap=True,all_workers_retired=True,
    exact_qualified_state_io_body=True,state_io_parent=pin(Q/'state_io.py'),state_io=pin(HERE/'state_io.py'),
    original_inputs=pin(old_inputs),input_copy_exact=True,targets_decoded=False,
    initialized_from_official_parent_only=True,synthetic_qualification_edge_checkpoint_not_used=True,
    frozen_schedule=dict(seed=9409213,graphs=160,updates=40,batch=4,lr_start=.001,lr_end=.0001)))
paths=[p for p in HERE.iterdir() if p.is_file()]
paths += [Path(p['path']) for p in read(Q/'sources_sealed.json')['pins']]
paths += [Q/n for n in ('result.json','unit_result.json','full_result.json','unit_watch.json','full_watch.json','initial_rng.npz')]
paths += [Path(p['path']) for p in full['records']]
for p in paths:
    if p.suffix=='.py':ast.parse(p.read_text())
pins=[pin(p) for p in sorted(set(paths))]
write(HERE/'sources_sealed.json',dict(pins=pins,data_pins=data,protocol=pin(HERE/'PROTOCOL.md'),targets_decoded=False))
print(dict(prepared=True,pins=len(pins),data_pins=len(data),qualification_peak_rss_bytes=max(w['max_rss_bytes'] for w in qw)),flush=True)
