"""Inactive fixed .5 parameter midpoint. Root must supply an explicit selection receipt."""
from pathlib import Path
import ast, base64, copy, hashlib, json, math, sys, zlib
HERE=Path(__file__).resolve().parent

def pin(path):
 p=Path(path).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def checked(record):
 assert pin(record['path'])==record,record['path']
 return json.loads(Path(record['path']).read_text())
assert len(sys.argv)==4
plan_pin=pin(HERE/'export_plan.json');assert plan_pin['sha256']==sys.argv[1]
plan=checked(plan_pin)
selection_pin=pin(Path(sys.argv[2]));assert selection_pin['sha256']==sys.argv[3]
selection=checked(selection_pin)
assert selection['root_selected'] is True and selection['export_plan']==plan_pin
assert selection['coefficient']==plan['coefficient']==.5
assert selection['exporter']==plan['exporter']==pin(__file__)
assert plan['schema']=='fixed-detached-star8-16-parameter-midpoint-v1'
assert plan['native_parameters']==pin(HERE/'native_parameters.py')
assert not (HERE/'candidate.py').exists() and not (HERE/'export_manifest.json').exists()
endpoints=[]
for expected,record in zip((8,16),plan['exports']):
 ex=checked(record)
 assert ex['passed'] and ex['optimizer_updates']==ex['selected_endpoint']==expected
 assert ex['baseline']==plan['baseline'] and ex['source_inverse_exact'] and ex['unrelated_statements_ast_exact']
 assert ex['only_five_learned_and_six_dial_assignments'] and ex['frozen_parameter_bytes_exact']
 for field in ('candidate','endpoint','dials','training_completion'):
  assert pin(ex[field]['path'])==ex[field]
 assert ex['training_guards_checked']==expected//4
 for r in ex['training_guards']+ex.get('prior_training_guards',[]):
  g=checked(r)
  assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows']
  assert 0<g['elapsed_s']<=600 and 0<g['max_rss_bytes']<g['rss_trigger_bytes']==7500000000 and g['max_swap_growth_bytes']==0
 endpoints.append(ex)
for r in plan['failed_assessments']:
 failure=checked(r);assert failure['passed'] is False and not failure.get('admitted',False)
assert pin(plan['literal_export_template']['path'])==plan['literal_export_template']
sys.path.insert(0,str(HERE))
from native_parameters import SOURCE,SOURCE_SHA,FROZEN_P,LITERALS,load_s33_arrays,_source_values,_nest
assert pin(SOURCE)==plan['baseline'] and SOURCE_SHA==plan['baseline']['sha256']
assert FROZEN_P==('centers','scales','rms','gain','first')
# No numerical imports or endpoint reads occur before the explicit selection and evidence checks.
import numpy as np
base,base_dials=load_s33_arrays(np)
loaded=[]
for ex in endpoints:
 with np.load(ex['endpoint']['path'],allow_pickle=False) as saved:
  loaded.append({k:saved[k].copy() for k in saved.files})
 assert set(loaded[-1])==set(base)
 for k,v in loaded[-1].items():
  assert v.dtype==np.float32 and v.shape==base[k].shape and np.isfinite(v).all(),k
frozen={'p.'+k for k in FROZEN_P};trainable=set(base)-frozen
assert len(base)==28 and len(trainable)==23
arrays={}
for k in base:
 a,b=loaded[0][k],loaded[1][k]
 if k in frozen:
  assert a.tobytes()==b.tobytes()==base[k].tobytes(),k
  arrays[k]=base[k].copy()
 else:
  arrays[k]=((a.astype(np.float64)+b.astype(np.float64))*.5).astype(np.float32)
 assert np.isfinite(arrays[k]).all()
dial_pair=[checked(ex['dials']) for ex in endpoints]
assert all(set(x)==set(base_dials) for x in dial_pair)
frozen_dials={'X3','c1'};assert len(set(base_dials)-frozen_dials)==10
dials={}
for k in base_dials:
 a,b=dial_pair[0][k],dial_pair[1][k]
 assert math.isfinite(a) and math.isfinite(b)
 if k in frozen_dials:
  assert a==b==base_dials[k];dials[k]=base_dials[k]
 else:dials[k]=.5*(a+b)
 assert math.isfinite(dials[k])
out=HERE/'candidate.py'
literal, nodes = _source_values()
packed = {}
for key in literal['_GNN_LITERAL']:
    value = np.ascontiguousarray(arrays['p.' + key], dtype=np.float32)
    packed[key] = (tuple(value.shape), base64.b64encode(zlib.compress(value.tobytes())).decode())
replacements = dict(_GNN_LITERAL=packed,
    _BDNH_A_LITERAL=_nest(literal['_BDNH_A_LITERAL'], arrays['a'].tolist()),
    _BD_COEFFICIENT_LITERAL=_nest(literal['_BD_COEFFICIENT_LITERAL'], arrays['c'].tolist()),
    _EDGE_LITERAL={key: _nest(value, arrays['e.' + key].tolist()) for key, value in literal['_EDGE_LITERAL'].items()},
    _MESSAGE_EDGE_LITERAL={key: _nest(value, arrays['e.context_' + key].tolist()) for key, value in literal['_MESSAGE_EDGE_LITERAL'].items()},
    _BD_J1_PHYS=dials['X1'], _S19_BD_ADD=(dials['X0'], 0., dials['X2'], dials['X3']),
    _S19_BSCALE=dials['bscale'], _S19_LAM=(dials['s_carry'], dials['s_new']),
    _S19_K4BAND=(dials['k4a'], dials['k4b'], dials['k4c']))
source = SOURCE.read_text()
old_tree = ast.parse(source)
edits = {}
for name, value in replacements.items():
    node = nodes[name][-1]
    if name in LITERALS:
        assert len(nodes[name]) == 1
    edits[node.lineno] = (node.end_lineno, f'{name} = {value!r}')
table = copy.deepcopy(nodes['_K31R_TABLE'][-1])
assert isinstance(table.value, ast.DictComp)
original_generators = ast.dump(table.value.generators[0])
original_key = ast.dump(table.value.key)
table.value.value = ast.Tuple(elts=[ast.Constant(dials['c0']), ast.Constant(dials['c1'])], ctx=ast.Load())
edits[table.lineno] = (table.end_lineno, ast.unparse(table))
lines = source.splitlines(keepends=True)
for start, (end, text) in sorted(edits.items(), reverse=True):
    lines[start - 1:end] = [text + '\n']
candidate = ''.join(lines)
new_tree = ast.parse(candidate)
assert len(new_tree.body) == len(old_tree.body)
allowed = set(replacements) | {'_K31R_TABLE'}
assert len(allowed) == 11 and set(LITERALS).issubset(allowed)
changed = []
for old, new in zip(old_tree.body, new_tree.body):
    if ast.dump(old) != ast.dump(new):
        assert isinstance(old, ast.Assign) and isinstance(old.targets[0], ast.Name)
        name = old.targets[0].id
        assert name in allowed and old.lineno in edits and new.targets[0].id == name
        changed.append(name)
new_nodes = {}
for node in new_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        new_nodes.setdefault(node.targets[0].id, []).append(node)
for key, value in replacements.items():
    assert ast.literal_eval(new_nodes[key][-1].value) == value
new_table = new_nodes['_K31R_TABLE'][-1].value
assert ast.dump(new_table.key) == original_key and ast.dump(new_table.generators[0]) == original_generators
assert ast.literal_eval(new_table.value) == (dials['c0'], dials['c1'])
# Undo the 11 replacements and require exact original source bytes.
old_lines = source.splitlines(keepends=True)
undo = candidate.splitlines(keepends=True)
for new, old in sorted(((new_nodes[name][-1], nodes[name][-1]) for name in allowed),
                       key=lambda pair: -pair[0].lineno):
    undo[new.lineno - 1:new.end_lineno] = old_lines[old.lineno - 1:old.end_lineno]
assert ''.join(undo).encode() == SOURCE.read_bytes()
compile(candidate, str(out), 'exec')
with out.open('x') as stream:
    stream.write(candidate)
record=dict(schema='detached-star8-16-midpoint-export-v1',passed=True,candidate=pin(out),baseline=pin(SOURCE),exporter=pin(__file__),export_plan=plan_pin,selection=selection_pin,source_exports=plan['exports'],source_endpoints=[x['endpoint'] for x in endpoints],source_dials=[x['dials'] for x in endpoints],failed_assessments_preserved=plan['failed_assessments'],coefficient=.5,averaged_arrays=23,averaged_dials=10,total_arrays=28,frozen_parameter_bytes_exact=True,frozen_dials=['X3','c1'],changed_statements=changed,only_five_learned_and_six_dial_assignments=True,unrelated_statements_ast_exact=True,k31_table_keys_unchanged=True,source_inverse_exact=True,weight_mean_is_not_output_mean=True,adaptive_selection=True,new_training_executed=False,native_executed=False,retained=False,bank_qualified=False,submitted=False)
with (HERE/'export_manifest.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(candidate=record['candidate'],export_manifest=pin(HERE/'export_manifest.json'))))
