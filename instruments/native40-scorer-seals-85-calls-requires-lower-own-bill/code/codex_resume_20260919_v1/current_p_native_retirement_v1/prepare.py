"""Only after allocator qualification: authenticate44 reused calls and seal41 new ones."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
OLD = D/'current_p_native_lifecycle_v1'
QUAL = D/'current_p_native_allocator_qualification_v1'
FIT = D/'current_full_lower_p_fit_v1'


def read(path):
    return json.loads(path.read_text())


def pin(path):
    path = path.resolve();b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def check(p):
    assert pin(Path(p['path'])) == {k:p[k] for k in ['path','bytes','sha256']},p['path']


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


q = read(QUAL/'controller_result.json')
assert q['completed'] and q['passed'] and q['completed_calls'] == 8
check(q['qualification'])
q = read(QUAL/'qualification.json')
assert q['completed'] and q['passed'] and q['integrity_passed'] and not q['target_arrays_opened']
assert q['exact_four_call_outputs_bills_domains_selectors'] and q['historical_prefix_exact_calls'] == 3
assert q['minimum_reduction_bytes'] == 128*1024**2
assert q['baseline_maximum_retired_rss_bytes']-q['relief_maximum_retired_rss_bytes'] >= q['minimum_reduction_bytes']
assert q['maximum_single_relief_rss_drop_bytes'] >= q['minimum_reduction_bytes']
for name in ['native_baseline','qualify_baseline','native_relief','qualify_comparison']:
    w = read(QUAL/f'{name}_watch.json')
    assert w['completed'] and w['exit_code'] == 0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes'] == 0
assert not (OLD/'native40_score.json').exists() and not (FIT/'native40_score.json').exists()
assert not read(OLD/'controller_result.json')['completed']
for p in read(QUAL/'sources_sealed.json')['pins']+read(QUAL/'endpoint_sources_sealed.json')['pins']:
    check(p)
copies = []
names = ['fit_common.py','endpoint_common.py','guard.py','lifecycle.py','inputs.json','build.json','build_watch.json',
    'fit_complete.json','fit_watch.json','seal_predictions.py','score_native40.py','score_with_lifecycle.py',
    'native_null_qualification.json','qualify_null_watch.json','native_null_watch.json','qualify_null_lifecycle_watch.json',
    'null_lifecycle_qualified.json','native_parent_watch.json','qualify_parent_lifecycle_watch.json','parent_lifecycle_qualified.json',
    'native_null/result.json','native_parent/result.json']
for name in names:
    dest = HERE/name;dest.parent.mkdir(exist_ok=True)
    with dest.open('xb') as f:f.write((OLD/name).read_bytes())
    copies.append(dict(original=pin(OLD/name),copy=pin(dest),byte_exact=True,reused_reference=True))
with (HERE/'retirement.py').open('xb') as f:f.write((QUAL/'retirement.py').read_bytes())
copies.append(dict(original=pin(QUAL/'retirement.py'),copy=pin(HERE/'retirement.py'),byte_exact=True))
for role,count in [('null',3),('parent',41)]:
    result = read(HERE/f'native_{role}/result.json')
    assert result['completed'] and result['full_estimator_calls'] == count and not result['truth_accessed']
    qrole = read(HERE/f'{role}_lifecycle_qualified.json')
    assert qrole['passed'] and qrole['calls'] == count and not qrole['truth_accessed']
    for p in qrole['lifecycle_pins']+qrole['original_comparison_pins']+[qrole['records'],qrole['watch']]:check(p)
source = (QUAL/'native.py').read_text()
before_admission = '''ranks, source = [37, 39, 42, 37], build['candidate']
if args.role == 'relief':
    done('qualify_baseline')
    assert read(HERE/'baseline_qualified.json')['passed']
'''
after_admission = '''ranks, source = inputs['valid'] + inputs['valid'][:1], build['candidate']
done('qualify_null')
done('qualify_null_lifecycle')
done('qualify_parent_lifecycle')
assert read(HERE/'native_null_qualification.json')['passed']
assert read(HERE/'null_lifecycle_qualified.json')['passed']
assert read(HERE/'parent_lifecycle_qualified.json')['passed']
qualification = read(D/'current_p_native_allocator_qualification_v1/controller_result.json')
assert qualification['completed'] and qualification['passed']
'''
changes = [("choices=('baseline', 'relief')","choices=('trained',)"),
    (before_admission,after_admission),
    ("released = allocator.relieve() if args.role == 'relief' else None",'released = allocator.relieve()'),
    ("allocator_called=args.role == 'relief'",'allocator_called=True')]
modified = source
for before,after in changes:
    assert modified.count(before) == 1
    modified = modified.replace(before,after)
recovered = modified
for before,after in reversed(changes):
    assert recovered.count(after) == 1
    recovered = recovered.replace(after,before)
assert recovered == source
def paid(s):
    return next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.With) and
        any(isinstance(i.optional_vars,ast.Name) and i.optional_vars.id == 'ctx' for i in n.items))
assert ast.dump(paid(modified),include_attributes=False) == ast.dump(paid((OLD/'native.py').read_text()),include_attributes=False)
with (HERE/'native.py').open('x') as f:f.write(modified)
seal = (OLD/'seal_with_lifecycle.py').read_text()
before = "    for p in q['lifecycle_pins']+q['original_comparison_pins']+[q['records'],q['watch']]:"
after = "    if role == 'trained':\n        assert q['all_retirements_qualified'] and q['historical_prefix_exact_calls'] == 30\n        check(q['allocator_qualification']);pins.append(q['allocator_qualification'])\n    for p in q['lifecycle_pins']+q['original_comparison_pins']+q.get('retirement_pins',[])+[q['records'],q['watch']]:"
assert seal.count(before) == 1
new_seal = seal.replace(before,after)
assert new_seal.replace(after,before) == seal
with (HERE/'seal_with_lifecycle.py').open('x') as f:f.write(new_seal)
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert not {'numpy','torch','flopscope'} & set(sys.modules)
write(HERE/'source_copies.json',dict(copies=copies,original=pin(QUAL/'native.py'),successor=pin(HERE/'native.py'),
    native_changes=[dict(before=a,after=b) for a,b in changes],inverse_source_byte_exact=True,paid_body_AST_exact=True,
    input_preparation_unchanged=True,seal_original=pin(OLD/'seal_with_lifecycle.py'),seal_inverse_exact=True,
    seal_change=dict(before=before,after=after),candidate=read(HERE/'build.json')['candidate'],
    original_scoring_and_prediction_sealer_byte_exact=True,reused_qualified_calls=44,planned_new_calls=41,
    estimator_calls_during_preparation=0,target_arrays_opened=False,fit_and_build_reused_without_execution=True))
external = [QUAL/'sources_sealed.json',QUAL/'endpoint_sources_sealed.json',QUAL/'qualification.json',
    QUAL/'controller_result.json',QUAL/'native.py',QUAL/'retirement.py',OLD/'controller_result.json',
    OLD/'native_trained_watch.json',OLD/'native.py',OLD/'seal_with_lifecycle.py',FIT/'candidate.py']
external += [QUAL/f'{name}_watch.json' for name in ['native_baseline','qualify_baseline','native_relief','qualify_comparison']]
external += [OLD/f'native_trained/call_{i:02d}.json' for i in range(30)]
external += [Path(p['path']) for i in range(30) for p in [read(OLD/f'native_trained/call_{i:02d}.json')['prediction'],read(OLD/f'native_trained/call_{i:02d}.json')['bill']]]
pins = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]+[pin(p) for p in external]
write(HERE/'sources_sealed.json',dict(pins=pins,data_pins=[],metadata_only=True,planned_new_calls=41,reused_qualified_calls=44,
    no_candidate_change=True,no_target_access=True,allocator_qualification_passed=True))
write(HERE/'endpoint_sources_sealed.json',dict(pins=[pin(HERE/'sources_sealed.json')]+pins,
    paid_body_AST_exact=True,input_preparation_unchanged=True,original_score_and_gates_unchanged=True))
print(dict(prepared=True,source_seal=pin(HERE/'sources_sealed.json'),paid_body_AST_exact=True,
    new_calls=41,reused_qualified_calls=44),flush=True)
