"""Metadata helpers; no numerical imports or implicit worker launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
ROOT = D.parents[4]
A = P2/'astra_phase2_reexamination_v1'
FIT = D/'current_p_edge_objective_fit_v1'
TRAIN_MANIFEST = FIT/'inputs.json'
ENDPOINT = D/'current_p_edge_objective_native_v1'
NATIVE_BRIDGE = D/'current_p_edge_objective_native_bridge_v1'
REFERENCE = D/'current_p_release_v1'
PREVIOUS = D/'current_p_downstream_retirement_v1'
BANK = REFERENCE/'candidate.py'
RUNTIME = A/'openblas_runtime_env_v1/bin/python'
PANEL = [40,41,43,44,45,46,47,49,50,51,52,53,54,55]
CONFIRM = [37,39,42,48,59,63,66,67]
PUBLIC = [35,100,150,200,250,300,400,500]
VIRGIN = list(range(3016500,3016506))
INDEPENDENT = PUBLIC+VIRGIN


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20),b''):
            h.update(b)
    return dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size)


def check(item):
    assert pin(item['path']) == {k:item[k] for k in ('path','sha256','bytes')},item['path']


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def clean_watch(path):
    r = read(path)
    assert r['completed'] and r['exit_code']==0 and r['whole_group_retired']
    assert r['max_swap_growth_bytes']==0 and not r.get('remaining_rows',[])
    assert r['max_rss_bytes']<7_500_000_000
    return r


def verify():
    for item in read(HERE/'sources_sealed.json')['pins']:
        check(item)
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000


def admission():
    verify()
    r = read(HERE/'admission.json')
    assert r['passed'] and r['sole_terminal_fit']
    assert r['bank']==pin(BANK)
    for item in r['pins']:
        check(item)
    return r


def stage_ready(stage):
    assert stage in ('control','panel','confirmation','independent')
    admission()
    if stage!='control':
        clean_watch(HERE/'qualify_control_watch.json')
        q = read(HERE/'control_qualification.json')
        assert q['passed'] and q['all_retirements_qualified']
    prior = {'confirmation':'panel','independent':'confirmation'}.get(stage)
    if prior:
        clean_watch(HERE/f'score_{prior}_watch.json')
        assert read(HERE/f'{prior}_score.json')['passed']


def qualify_retirements(name,records):
    pins = []
    for i,r in enumerate(records):
        path = HERE/name/f'call_{i:02d}_retirement.json'
        x = read(path)
        assert x['complete'] and x['index']==i and x['rank']==r['rank']
        assert x['all_tracked_call_objects_released'] and x['tracked_objects']==18 and not x['live_objects']
        assert x['after_closed_saved_bill'] and x['saved_bill_unchanged'] and x['allocator_called']
        assert not x['allocator_statistics_called'] and not x['candidate_state_reset']
        check(x['saved_bill'])
        assert x['saved_bill']==r['bill']
        pins.append(pin(path))
    return pins


def guarded(name,seconds,script,*args):
    verify()
    assert not endpoint_metadata().numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc = subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),
            '--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),
            str(RUNTIME),'-B',str(HERE/script),*map(str,args)],
            stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    r = clean_watch(HERE/f'{name}_watch.json')
    assert not endpoint_metadata().numerical_workers()
    verify()
    print(dict(completed=name,wall_s=r['elapsed_s'],rss=r['max_rss_bytes']),flush=True)


def endpoint_metadata():
    spec=importlib.util.spec_from_file_location('edge_endpoint_metadata',ENDPOINT/'common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prior_workers_retired():
    ec=endpoint_metadata();assert not ec.numerical_workers()
    known=[(14323,'run_fit.py')]
    for folder,name in [(NATIVE_BRIDGE,'bridge.py'),(ENDPOINT,'run.py')]:
        claim=read(folder/'run_claim.json');known.append((claim['pid'],name))
    for line in ec.process_rows():
        fields=line.strip().split(None,3)
        if len(fields)!=4:continue
        command=fields[3];pid=int(fields[0])
        assert not any(folder.name+'/' in command for folder in (FIT,ENDPOINT,NATIVE_BRIDGE)),line
        assert not any(pid==owner and command.endswith(name) for owner,name in known),line


def admit():
    verify();prior_workers_retired();ec=endpoint_metadata()
    fit=ec.fit_success();build=ec.admission()
    controller=read(ENDPOINT/'controller_result.json')
    assert controller['completed'] and controller['scientific_passed'] and controller['error'] is None
    assert controller['new_native_calls']==87 and controller['no_external_actions'] and controller['no_automatic_downstream']
    bridge=read(NATIVE_BRIDGE/'result.json')
    assert bridge['completed'] and bridge['launched'] and bridge['error'] is None
    assert bridge['native_completed'] and bridge['scientific_passed'] and bridge['native_exit_code']==0
    assert bridge['native_result']==pin(ENDPOINT/'controller_result.json')
    assert read(NATIVE_BRIDGE/'launch.json')['pid']==read(ENDPOINT/'run_claim.json')['pid']
    valid=read(TRAIN_MANIFEST)['valid']
    names=['build','native_null','native_null_cold100','qualify_null','native_trained_control','qualify_controls']
    names += [f'{role}_{rank}' for rank in valid for role in ('parent','trained')]
    names += ['seal_predictions','score_native40']
    assert len(names)==88
    for name in names:ec.done(name)
    null=read(ENDPOINT/'null_qualification.json');control=read(ENDPOINT/'control_qualification.json')
    assert null['passed'] and null['all_retirements_qualified'] and null['own_complete_parent_bills_exact']
    assert null['warm_cold100_output_exact'] and control['passed'] and control['all_retirements_qualified']
    assert control['trained_repeat_bytes_exact'] and control['first8_parent_layers_exact']
    score=read(ENDPOINT/'native40_score.json');sealed=read(ENDPOINT/'native40_predictions_sealed.json')
    assert score['completed'] and score['integrity_passed'] and score['passed'] and score['native_score_measured']
    assert score['adjusted_ratio']<1 and score['adjusted_wins']>=24 and score['first8_layers_byte_exact']
    assert score['matched_cold_initialization'] and score['all_predictions_and_bills_verified_before_validation_targets']
    assert score['candidate']==build['candidate'] and score['prediction_seal']==pin(ENDPOINT/'native40_predictions_sealed.json')
    assert controller['score']==pin(ENDPOINT/'native40_score.json')
    assert sealed['completed'] and sealed['before_validation_targets'] and sealed['own_complete_bills']
    assert sealed['matched_cold_initialization'] and sealed['control_calls_excluded'] and sealed['all_retirements_qualified']
    assert sealed['scored_ranks']==valid and len(sealed['rows'])==40
    assert build['parent']==pin(BANK) and build['terminal_payload']==fit['terminal_payload']
    assert build['whole_source_inverses_exact'] and build['only_five_edge_values_changed']
    assert build['fixed_literals_and_functions_unchanged'] and build['fp32_payload_bytes_exact']
    for artifact in (null,control,sealed):
        for p in artifact['pins']:check(p)
    baseline=read(HERE/'baseline_reuse.json')
    assert baseline['passed'] and baseline['cold_calls']==22 and baseline['source_bytes_equal_official_331760']
    assert baseline['own_complete_bills'] and baseline['cold_initialization'] and baseline['closed_contexts_qualified']
    assert baseline['all_cold_worker_groups_retired'] and not baseline['historical_per_object_release_audited']
    copies=read(HERE/'source_copies.json');assert copies['input_and_predict_BudgetContext_AST_exact'] and copies['score_code_byte_exact']
    pins=[pin(FIT/n) for n in ('fit_result.json','controller_result.json','sources_sealed.json')]
    pins += [pin(FIT/f'{name}_watch.json') for name in [f'batch_{i:02d}' for i in range(40)]+['verify_fit']]
    pins += [pin(ENDPOINT/n) for n in ('controller_result.json','sources_sealed.json','build.json','null_qualification.json','control_qualification.json','native40_predictions_sealed.json','native40_score.json','preparation.json')]
    pins += [pin(ENDPOINT/f'{name}_watch.json') for name in names]
    pins += [pin(NATIVE_BRIDGE/n) for n in ('sources_sealed.json','run_claim.json','launch.json','result.json')]
    pins += [fit['terminal_payload'],build['parent'],build['candidate'],build['null_candidate'],pin(HERE/'baseline_reuse.json')]
    write(HERE/'admission.json',dict(passed=True,sole_terminal_fit=True,bank=build['parent'],candidate=build['candidate'],
        pins=list({p['path']:p for p in pins}.values()),native40_artifacts_verified=True,
        previous_fit_native_and_bridge_workers_retired=True,all87_endpoint_calls_complete=True,
        all88_endpoint_guards_pinned=True,all41_fit_guards_pinned=True,
        scientific_rosters_and_gates_unchanged=True,new_calls_18_object_retirement_required=True,
        historical_cold_baselines_closed_context_and_process_retirement_qualified=True))
