"""Build one prospective retained-READ7 extension; source/JSON/stdlib only."""
from pathlib import Path
import ast
import copy
import hashlib
import json
from types import SimpleNamespace

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_s34_read7_v1'
def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
def once(p, x):
    with p.open('x') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')
parent = pin(P/'candidate.py')
assert parent['sha256'] == '9ea3a975561d6fae2ae8806b9812d81fe7cadc23845f86cce14b80696d0bf7d6'
retained = json.loads((P/'retained_component.json').read_text())
price_path = H/'hill_climb_s34_read7_extended_v1/kpad_scalar_price.json'
price = json.loads(price_path.read_text())
assert pin(price_path)['sha256'] == 'cd0e5dff457c08eea2914f7a3337dc4aa051fc0cfd3a26ffcd62ba888379400e'
assert price['selected_extra_calls'] == 4 and price['forecast_gain_pct'] >= .15
assert pin(P/'retained_component.json')['sha256'] == 'ad0e0e3e710de25bef877d5eb1efdc1da838086b436373bf78e3deeebf279843'
assert retained['candidate'] == parent and retained['native8_runtime_recheck']['passed']
fragment = (D/'fragment.py').read_text(); source = (P/'candidate.py').read_text() + '\n' + fragment
ast.parse(source)
tree = ast.parse(fragment)
allowed = {'_r7x_bill7', '_r7x_plan', '_r7x_acc', '_r7x_product', '_mm', '_pb_bind'}
assert {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} == allowed
assert 'if (m, k, n) not in _R7_SHAPES:' in fragment
assert 'b7 = _r7x_bill7(M, K, N) + _KS_PAD * pad + (m * n if M != m or N != n else 0)' in fragment
# Capacity and prebind checks are scalar dimensions only, no array execution.
A = ((640,1792),(640,1920),(768,1664),(768,1792),(768,1920))
B = ((1664,896),(1792,896),(1920,896))
C = ((640,896),(768,896))
assert len(A)+len(B)+2*len(C) == 12
assert all(7**7*(a//128)*(b//128) <= 7**7*128 for a,b in A+B)
assert all(2*7**7*(a//128)*(b//128) <= 7**7*128 for a,b in C)
def scalar_defs(path, names):
    parsed = ast.parse(path.read_text())
    found = {n.name:n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names}
    assert set(found) == set(names)
    return ast.Module(body=list(found.values()), type_ignores=[])
ns = dict(_IPT_REGION={6:(90354432,120472576,120472576)}, _SCRATCH=(SimpleNamespace(shape=(391602176,)),),
    _KS_C5=18.36, _KS_C6=30.60, _KS_PAD=1.35, _KS_CAP=1<<20, _KS_PRICE=750000., _KS_REQ5=72,
    _KS_PLANS={}, _R7X_PLANS={})
exec(compile(scalar_defs(P/'candidate.py', ('_ipt_fits','_ks_plan')), 'pinned_scalar_parent', 'exec'), ns)
exec(compile(scalar_defs(D/'fragment.py', ('_r7x_bill7','_r7x_plan')), 'scalar_candidate_plan', 'exec'), ns)
fixed = {(1024,2048,1024),(1024,4096,1024),(992,6144,1024)}
expected = {(x['rank'],x['layer']):x for x in price['selected_extras']}
observed = {}
for row in json.loads((H/'young_read_shapes_train2_v1/shape_census.json').read_text())['records']:
    m,k,n = row['response_rows'],row['actual_young_columns'],row['current_live']
    plan = None if (m,k,n) in fixed else ns['_r7x_plan'](m,k,n)
    if plan is None: continue
    M,K,N,sizes = plan
    assert sum(sizes)==K and K==(k+127)//128*128 and all(q%128==0 and 0<q<=2048 for q in sizes)
    assert all((M,q) in A and (q,N) in B for q in sizes) and (M,N) in C
    assert all(7**7*(M//128)*(q//128)<=7**7*128 and 7**7*(q//128)*(N//128)<=7**7*128 for q in sizes)
    assert 2*7**7*(M//128)*(N//128)<=7**7*128 and 3*7**7*128<=391602176
    key=(row['rank'],row['layer']); assert key in expected
    assert [M,K,N]==expected[key]['padded'] and list(sizes)==expected[key]['chunks7']
    observed[key]=dict(shape=[m,k,n],padded=[M,K,N],chunks=list(sizes),capacity_passed=True)
assert set(observed)==set(expected) and len(observed)==4
once(D/'scalar_shape_checks.json', dict(fragment=pin(D/'fragment.py'), parent=parent, pricing=pin(price_path),
    selected=[dict(rank=k[0],layer=k[1],**v) for k,v in observed.items()], scalar_only=True,arrays_imported=False,model_executed=False))
with (D/'candidate.py').open('x') as f: f.write(source)
once(D/'protocol.json', dict(parent=parent, retained_parent=pin(P/'retained_component.json'), fragment=pin(D/'fragment.py'),
    scope='native2_only', comparator='retained READ7 fixed three routes', parity_layers=[0],
    eligibility='Existing _ks_plan not None; min(M,N)>=640; 64 padding equals 128 padding; k>=4096 divisible64; K rounded up128; modeled D7 <=0.98 times selected KS6 model.',
    fixed_three_routes_unchanged=True, chunks='Balanced multiples128, each<=2048, sumK; zero-pad both whole operands whenever innerK grows.',
    private_region_floats=[7**7*128]*3, arena_floats=391602176, extra_prebound_views=12,
    prebind=dict(A=A,B=B,C_and_acc=C), forecast_only=dict(additional_bill_gain_pct=price['forecast_gain_pct'],additional_native_seconds=price['full_chunk_seconds_forecast'],selected_extra_calls=price['selected_extra_calls'],selected_extra_classes=price['selected_extra_classes'],scalar_receipt=pin(price_path),shape_checks=pin(D/'scalar_shape_checks.json')),
    no_new_shape_census=True, no_threshold_sweep=True, original_s34_comparison_required_later=True,no_further_variant_if_failed=True,
    broad_qualified=False, serving_qualified=False, bank_min_gain_pct=1.0))
once(D/'build.json', dict(candidate=pin(D/'candidate.py'), parent=parent, fragment=pin(D/'fragment.py'),
    builder=pin(__file__), protocol=pin(D/'protocol.json'), exact_parent_prefix=True,
    exact_source_inverse=True, original_three_routes_unchanged=True, scalar_capacity_checks=True))
cfg = json.loads((P/'native_screen_continuation_v3.json').read_text())
original = copy.deepcopy(cfg)
retained_arm = copy.deepcopy(cfg['arms']['candidate'])
receipt = json.loads((P/'native8_assessment.json').read_text())
assert receipt['admitted'] and receipt['passed_integrity'] and receipt['candidate'] == parent
for block in receipt['blocks']:
    retained_arm['blocks'][str(block['size'])]['artifacts'] = block['arms']['candidate']
cfg['arms']['base'] = retained_arm
cfg['arms']['parity_parent'] = copy.deepcopy(retained_arm)
child = cfg['arms']['candidate']
child['source'] = pin(D/'candidate.py')
child['setup']['counts']['bound'] = 698
child['setup']['counts']['read7_extended_fused'] = 12
for size in ('2','6'):
    child['blocks'][size].pop('artifacts', None)
    child['blocks'][size]['tag'] = D.name+'_train'+size+'_v1'
    child['blocks'][size]['guard'] = str(D/('native'+size+'_guard.json'))
for stage in ('2','8'):
    cfg['stages'][stage]['result'] = str(D/('native'+stage+'_assessment.json'))
    cfg['stages'][stage]['assessment_guard'] = str(D/('native'+stage+'_assess_guard.json'))
cfg['parity_layers'] = [0]
req = cfg['boundary_before8']
req['receipt'] = str(D/'boundary_not_admitted.json')
req['assessment_guard'] = str(D/'boundary_guard_not_admitted.json')
req['assessment_command'][3:] = [str(D/'boundary_config_not_admitted.json'), '0'*64]
cfg.pop('continuation', None)
cfg['provenance'] = [pin(p) for p in (Path(__file__), D/'fragment.py', D/'protocol.json', D/'build.json',
    P/'candidate.py', P/'native_screen_continuation_v3.json', P/'native8_assessment.json', P/'native8_assess_guard.json',
    P/'retained_component.json', H/'depth7_inner_reads_v1/fragment.py', price_path, D/'scalar_shape_checks.json')]
cfg['interpretation'] = 'One cost-rule extension compared with retained fixed-three READ7; original score/resource gates and layer0 parity. TRAIN2 only, no inherited boundary pass. Original S34 reporting required before broader progression.'
cfg['prospective_scope'] = 'native2_only'
assert all(cfg['stages'][s][k] == original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
assert (D/'candidate.py').read_bytes()[:-len( ('\n'+fragment).encode())] == (P/'candidate.py').read_bytes()
once(D/'native_screen.json', cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'), config=pin(D/'native_screen.json'), build=pin(D/'build.json'), protocol=pin(D/'protocol.json'))))
