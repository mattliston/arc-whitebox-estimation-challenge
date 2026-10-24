"""Bind the unchanged native scientific test to the new qualified batch-process fit."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
D=HERE.parent
OLD=D/'current_p_signed_branch_endpoint_v1'
F=D/'current_p_batch_process_fit_v1'
ORIGINAL_FIT=D/'current_p_signed_branch_fit_v1'
QUAL=D/'current_p_batch_process_qualification_v1'
Q=D/'current_p_signed_branch_qualification_v2'
PARENT=D/'current_p_release_v1/candidate.py'
ZERO=Q/'candidate_zero.py'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def write_text(p,value):
    with Path(p).open('x') as stream:stream.write(value)

assert not (HERE/'sources_sealed.json').exists() and not (HERE/'run_claim.json').exists()
assert shutil.disk_usage(HERE).free>=470_000_000_000
assert pin(OLD/'sources_sealed.json')['sha256']=='308e1c1f94a78b66890985a92180b093eab4298279d9428a365f07244c3b5cce'
assert pin(F/'sources_sealed.json')['sha256']=='7cd87bc172bd6ea45a78a447430852381edcf8d6973a61eca656bf89a60d9efe'
for entry in read(OLD/'sources_sealed.json')['pins']+read(F/'sources_sealed.json')['pins']:check(entry)
assert read(QUAL/'result.json')['passed'] and read(F/'admission.json')['passed']
assert read(ORIGINAL_FIT/'resource_failure.json')['closed']
assert not (OLD/'run_claim.json').exists() and not (OLD/'build.json').exists()
assert pin(PARENT)['sha256']=='8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
assert pin(ZERO)['sha256']=='4baff6bf94fa11edbad1117995c5ec18f1e6c44aea629ad071dcad71aa13e469'
assert read(D/'official_best_331760.json')['completed']==100

old_common=(OLD/'common.py').read_text();common=old_common
needle="F = D/'current_p_signed_branch_fit_v1'"
assert common.count(needle)==1
common=common.replace(needle,"F = D/'current_p_batch_process_fit_v1'",1)
start=common.index('def fit_success():');stop=common.index('\n\ndef clean_watch(',start)
old_gate=common[start:stop]
new_gate='''def fit_resources():
    summary = read(F/'fit_resource_summary.json')
    assert summary['completed'] and summary['passed']
    assert summary['kind']=='aggregate_of_40_separate_batch_guards_and_finalizer'
    assert summary['guard_count']==41 and summary['every_group_retired'] and not summary['remaining_rows']
    expected = [F/f'batch_{i:02d}_watch.json' for i in range(40)]+[F/'finalize_watch.json']
    assert summary['guards']==[pin(p) for p in expected]
    watches = [clean_watch(p) for p in expected]
    for i,w in enumerate(watches):
        command = [str(RUNTIME),'-B',str(F/('worker.py' if i<40 else 'finalize.py'))]
        if i<40:
            command += [str(i)]
            if i:command += ['--resume',str(F/f'batch_{i-1:02d}'/f'checkpoint_{i:02d}.json')]
        assert w['command']==command and w['error'] is None
    assert summary['max_rss_bytes']==max(w['max_rss_bytes'] for w in watches)<7_500_000_000
    assert summary['max_swap_growth_bytes']==max(w['max_swap_growth_bytes'] for w in watches)==0
    assert summary['max_retired_graph_rss_bytes']<2_000_000_000 and summary['elapsed_s']<14400
    return summary


def fit_success():
    verify()
    for entry in read(F/'sources_sealed.json')['pins']:check(entry)
    result = read(F/'controller_result.json')
    assert result['completed'] and result['passed'] and result['error'] is None
    assert result['sources']==pin(F/'sources_sealed.json')
    fit = read(F/'fit_result.json')
    assert fit['completed'] and fit['passed'] and fit['training_graphs']==160 and fit['updates']==40 and fit['epoch']==1
    assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
    assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
    assert 'ungated_w' in fit['changed_trainable_keys']
    assert fit['old_completed_graphs_exact']==134 and fit['old_completed_updates_exact']==33
    assert fit['qualified_gradient_graphs_exact']==8 and fit['qualified_state_boundaries_exact']==2
    assert fit['all_checkpoint_arrays_reverified'] and fit['all_worker_groups_retired']
    resources = fit_resources()
    assert fit['max_retired_graph_rss_bytes']==resources['max_retired_graph_rss_bytes']
    fit_scripts = {(F/name).resolve() for name in ['run_fit.py','guard.py','worker.py','finalize.py']}
    for process in processes():
        paths = [Path(x) for x in process['command'].split() if x.endswith(('/run_fit.py','/guard.py','/worker.py','/finalize.py'))]
        assert not any((p if p.is_absolute() else ROOT/p).resolve() in fit_scripts for p in paths),process
    assert fit['source_seal']==pin(F/'sources_sealed.json')
    for key in ['terminal_payload','training_order','source_seal']:check(fit[key])
    complete = read(F/'fit_complete.json')
    assert complete['completed'] and complete['result']==pin(F/'fit_result.json')
    assert complete['resource_summary']==pin(F/'fit_resource_summary.json')
    assert complete['sources']==pin(F/'sources_sealed.json') and result['fit_complete']==pin(F/'fit_complete.json')
    return fit
'''
common=common[:start]+new_gate+common[stop:]
old_done="    return clean_watch((F if name=='fit' else HERE)/f'{name}_watch.json')"
new_done="    return fit_resources() if name=='fit' else clean_watch(HERE/f'{name}_watch.json')"
assert common.count(old_done)==1
common=common.replace(old_done,new_done,1)
inverse=common.replace(new_done,old_done,1).replace(new_gate,old_gate,1).replace(
    "F = D/'current_p_batch_process_fit_v1'",needle,1)
assert inverse==old_common
write_text(HERE/'common.py',common)

names=['endpoint_common.py','native.py','records.py','retirement.py','guard.py',
    'qualify_null.py','seal_predictions.py','score_native40.py','run.py']
copies=[]
for name in names:
    with (HERE/name).open('xb') as stream:stream.write((OLD/name).read_bytes())
    assert (HERE/name).read_bytes()==(OLD/name).read_bytes()
    copies.append(dict(original=pin(OLD/name),copy=pin(HERE/name),byte_exact=True))
assert (HERE/'guard.py').read_bytes()==(F/'guard.py').read_bytes()
assert (HERE/'score_native40.py').read_bytes()==(D/'current_full_lower_p_fit_v1/score_native40.py').read_bytes()
build=(OLD/'build.py').read_text()
old_build=build
before="done('fit')";after='fit_success()'
assert build.count(before)==1;build=build.replace(before,after,1)
old_field="fit_guard=pin(F / 'fit_watch.json')"
new_field="fit_resources=pin(F / 'fit_resource_summary.json')"
assert build.count(old_field)==1;build=build.replace(old_field,new_field,1)
assert build.replace(new_field,old_field,1).replace(after,before,1)==old_build
write_text(HERE/'build.py',build)

protocol=(OLD/'PROTOCOL.md').read_text()
start=protocol.index('This stage is prepared');stop=protocol.index('The comparison parent',start)
protocol=protocol[:start]+'''This version binds the unchanged scientific test to the separately sealed
current_p_batch_process_fit_v1. It performs no numerical work until all160
graphs,40 updates,40 separate batch guards and the guarded final verifier
have passed, with zero added swap and complete worker/controller retirement.
Require exact reproduction of the old134-graph/33-update completed prefix
and the qualified eight-gradient/two-state-boundary prefix. Verify the
explicit41-guard resource summary; do not substitute a single guard receipt.
Only the sole terminal payload is eligible. No automatic bridge is installed.
The old failed fit and its endpoint/downstream chain stay closed.

'''+protocol[stop:]
write_text(HERE/'PROTOCOL.md',protocol)
inputs=read(OLD/'inputs.json')
assert inputs['fit_inputs']==pin(ORIGINAL_FIT/'inputs.json')
assert read(F/'training_order.json')==read(ORIGINAL_FIT/'training_order.json')
inputs.update(fit_execution_source=pin(F/'sources_sealed.json'),fit_training_order=pin(F/'training_order.json'),
    original_training_manifest_reused_exactly=True)
write(HERE/'inputs.json',inputs)
def paid_scopes(text):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text))
        if isinstance(n,ast.With) and any(isinstance(w.context_expr,ast.Call)
        and isinstance(w.context_expr.func,ast.Attribute) and w.context_expr.func.attr=='BudgetContext' for w in n.items)]
native_original=D/'current_p_release_v1/native.py'
assert len(paid_scopes(native_original.read_text()))==2
assert paid_scopes((HERE/'native.py').read_text())==paid_scopes(native_original.read_text())
write(HERE/'source_copies.json',dict(copies=copies,old_endpoint_sources=pin(OLD/'sources_sealed.json'),
    native_original=pin(native_original),native_adapted=pin(HERE/'native.py'),
    input_and_predict_BudgetContext_AST_exact=True,native_code_byte_exact_to_prior_sealed_endpoint=True,
    scoring_code_byte_exact=True,retirement_helper_byte_exact=True,
    common_inverse_source_exact=True,common_changes='fit path and explicit completed batch-process resource admission only',
    build_inverse_source_exact=True,builder_changes='stronger full-fit admission and resource-summary metadata only',
    fit_sources=pin(F/'sources_sealed.json'),fit_admission=pin(F/'admission.json'),
    reused_VALID40=True,adjusted_ratio_strictly_below=1,wins_at_least=24,
    numerical_calls=0,target_arrays_decoded=False,no_automatic_launch=True))
local=sorted(p for p in HERE.iterdir() if p.is_file())
for path in local:
    if path.suffix=='.py':ast.parse(path.read_text(),str(path))
external=[Path(p['path']) for p in read(OLD/'sources_sealed.json')['pins']]
external += [OLD/'sources_sealed.json']+[OLD/name for name in ['common.py','build.py','PROTOCOL.md','inputs.json','source_copies.json']+names]
external += [F/name for name in ['sources_sealed.json','PROTOCOL.md','admission.json','run_claim.json','training_order.json',
    'worker.py','state_io.py','iso_common.py','run_fit.py','finalize.py','guard.py']]
external += [QUAL/'result.json',QUAL/'qualification.json',ORIGINAL_FIT/'resource_failure.json']
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(set(local+external))],
    prepared_at_utc=datetime.now(timezone.utc).isoformat(),prepared_before_fit_result=not (F/'fit_result.json').exists(),
    numerical_calls=0,target_arrays_decoded=False,expected_data_pins=read(OLD/'sources_sealed.json')['expected_data_pins'],
    data_identities_reused_from_authenticated_fit_manifest=True,actual_data_reverified_before_use=True,
    no_automatic_downstream_or_upload=True,no_bridge_installed=True))
assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
print(dict(prepared=True,sources=pin(HERE/'sources_sealed.json'),source_pins=len(read(HERE/'sources_sealed.json')['pins']),
    planned_native_calls=86,scientific_code_byte_exact=True,no_native_started=True),flush=True)
