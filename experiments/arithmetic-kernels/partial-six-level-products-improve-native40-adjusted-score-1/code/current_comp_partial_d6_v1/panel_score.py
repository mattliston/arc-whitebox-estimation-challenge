"""Original independent14 NC score, current COMP comparator, no work on import."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
_HELPER_SHA = '77d238cbaa3f046ecadcaa7520d5b7edf4075843b3bc329696ab4457dd90571c'
if hashlib.sha256((HERE/'panel_common_v2.py').read_bytes()).hexdigest() != _HELPER_SHA:
    raise RuntimeError('Frozen independent14 acquisition helper changed')
import panel_common_v2 as c

SCORE_SCHEMA = c.PREFIX+'score_v1'
SCORE_PROTOCOL_SHA = '29f059c4993a5e40bf0c775a894a496a443159c8148cb75ea50eb77118b4f83b'
SCORE_OWNERS = dict(scorer='panel_score.py',score_protocol='SCORE_PANEL_PROTOCOL.md',
    score_watcher='panel_watch_score.py',score_watch_spec='SCORE_PANEL_WATCH_SPEC.md')
ACQUISITION_SOURCES = {
    'panel_common_v2.py':_HELPER_SHA,
    'panel_admit_v2.py':'9f93131823df7c7be4ccd2ebd6433f8c100f380c4e788d5c7d3792f585eebdc7',
    'panel_native_v2.py':'1f3388271c3bfcb58dbc66b4b8a1c07091f3ce5b0e4b7f781d7de1cca94ce3f3',
    'PANEL_PROTOCOL_V2.md':'cc4bd6dfb06d4903ad2c54fd1a222132b5b004eac95ded498ac2765e7df69233',
    'panel_hash_v2.py':'db708ef69a539f918c1ad2b0ad68ffb3a9d293a05a2bb2ddae92624e1e12b653',
    # Exact owner of the unchanged historical COMP two-seal admission function.
    'panel_admit.py':'ac38f0c27d61289c53b1278309ef99b5e9d126fb7c03f06188354fdad015b83f',
}
COMP_SCORER = c.COMP/'panel_score.py'
COMP_SCORER_SHA = 'b902977680b39413f57389e19cddd02ab140fe66ff2f727f5650fefb8a7ea7f5'
AUTHORITY_KEYS = (*c.OWNER_FILES,'score40','score40_watch')
RESULT_KEYS = ('context','admission_watch','acquisition','acquisition_watch')
METADATA_SHA = c.sha
SCORE_METADATA_HASH_EDITS = [
    ("    require(all(sha(path) == digest for path,digest in pins.items()),'Every output/source/bill seal before numeric imports')\n",
     "    require(all(METADATA_SHA(path) == digest for path,digest in pins.items()),'Every output/source/bill seal before numeric imports')\n"),
    ("    require(all(sha(path) == digest for path,digest in pins.items()),'End source/output/bill seals')\n",
     "    require(all(METADATA_SHA(path) == digest for path,digest in pins.items()),'End source/output/bill seals')\n"),
]


def score_arguments(role):
    parser = argparse.ArgumentParser(description='Fixed saved-output CD independent14 '+role)
    parser.add_argument('--run',action='store_true')
    for key in (*AUTHORITY_KEYS,*SCORE_OWNERS,*RESULT_KEYS):
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    if role == 'watcher':
        parser.add_argument('--stage',choices=('score',))
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return None
    c.require(role in ('scorer','watcher'),'No acquisition or fitting command')
    for key in (*AUTHORITY_KEYS,*SCORE_OWNERS,'acquisition','acquisition_watch'):
        c.require(re.fullmatch('[0-9a-f]{64}',getattr(args,key+'_sha256') or ''),'Explicit reviewed '+key)
    for key in ('context','admission_watch'):
        value = getattr(args,key+'_sha256')
        c.require((role == 'watcher' and value is None) or re.fullmatch('[0-9a-f]{64}',value or ''),
            'Exact retired context/handoff for scorer; supervisor may derive them from native watch')
    c.require(args.protocol_sha256 == c.PROTOCOL_SHA and args.score_protocol_sha256 == SCORE_PROTOCOL_SHA
        and args.score40_sha256 == c.SCORE_SHA and args.score40_watch_sha256 == c.SCORE_WATCH_SHA,
        'Exactly the completed40 selection and two separate prospective protocols')
    if role == 'watcher':
        c.require(args.stage == 'score' and args.prior_verdict_reviewed and args.numerical_slot_clear,
            'Explicit separately reviewed score launch and exclusive CPU slot')
    return args


def score_owners(args,tick=None):
    values = c.owners(args,tick)
    for name,digest in ACQUISITION_SOURCES.items():
        c.require(values.get(str(HERE/name)) == digest,'Unchanged reviewed acquisition source '+name)
    values.update({str(HERE/name):getattr(args,key+'_sha256') for key,name in SCORE_OWNERS.items()})
    values[str(COMP_SCORER)] = COMP_SCORER_SHA
    c.bind(values,tick)
    return values


def score_command(args,role='scorer',context_sha=None,admission_watch_sha=None):
    c.require(role == 'scorer','Only one saved-output scoring child')
    c.require(context_sha is None or context_sha == args.context_sha256,'Exact inherited context argument')
    c.require(admission_watch_sha is None or admission_watch_sha == args.admission_watch_sha256,
        'Exact inherited retired-admission argument')
    command = [str(c.OFFPY),'-B','-u',str(HERE/'panel_score.py'),'--run']
    for key in (*AUTHORITY_KEYS,*SCORE_OWNERS,*RESULT_KEYS):
        value = getattr(args,key+'_sha256')
        c.require(re.fullmatch('[0-9a-f]{64}',value or ''),'Complete child source/result command')
        command += ['--'+key.replace('_','-')+'-sha256',value]
    return command


def score_helper():
    """New score CLI only; native admission closures still own their native maps."""
    return SimpleNamespace(**dict(vars(c),arguments=score_arguments,owners=score_owners,
        command=score_command,SCORE_SCHEMA=SCORE_SCHEMA,
        COMP_CANDIDATE=c.pc.CANDIDATE,COMP_SHA=c.pc.CANDIDATE_SHA))


def verify_acquisition(args,context):
    pins,guard = context['pins'],context['score_guard']
    watch = c.verify_watch(args,context,pins,guard)
    record,end,final = c.seal3(c.OUTPUT,args.acquisition_sha256,pins,guard)
    outputs = c.verify_native(record,end,args.acquisition_sha256,args,context,pins)
    c.require(watch['child_finalization_sha256'] == c.sha(c.OUTPUT/'finalization.json',guard)
        and all(record.get(key) is True for key in ('all90_artifacts_qualified','all16_domains_qualified',
            'all47_selectors_qualified','additional_bindings_paid_once','same_existing_arena_qualified',
            'dispatch_tariffs_qualified','initialization_ownership_passed','COMP_parameter_initialization_owned',
            'native40_expansion_prerequisite_passed','full_admission_in_retired_process'))
        and record['full_model_calls'] == record['retired_call_count'] == len(outputs) == 15,
        'Complete current15 acquisition, actual third seal, own D6 bills and full domains')
    c.bind(pins,guard)
    context['qualified_acquisition'] = record
    context['acquisition_finalization_sha256'] = watch['child_finalization_sha256']
    return record,outputs


def control_admission(guard,pins,context):
    """Same complete target-free COMP two-seal comparator admission as native."""
    path = HERE/'panel_admit.py'
    c.require(c.sha(path,guard) == ACQUISITION_SOURCES[path.name],'Frozen control admission source')
    original = c.function_source(path,'qualified_comp')
    source = c.splice(original,[
        ('def qualified_comp(guard,inputs):','def qualified_control(guard,inputs):'),
        ('    pc.verify_native(native,end,c.CONTROL_SHA,oa,context,inputs)\n',
         '    control_outputs = pc.verify_native(native,end,c.CONTROL_SHA,oa,context,inputs)\n'),
        ('    return fixtures,rows,refs\n',
         '    return native,score,control_outputs,fixtures,rows\n')])
    namespace = dict(c=c)
    exec(compile(source,str(path)+':same-COMP-control-and-return-bytes','exec'),namespace)
    control,score,outputs,fixtures,rows = namespace['qualified_control'](guard,pins)
    c.require(fixtures == context['fixtures'] and rows == context['rows'] and len(outputs) == 15,
        'Current COMP uses exactly the admitted14 physical inputs and own repeat')
    context['control_admission_proof'] = dict(owner=str(path),owner_sha256=ACQUISITION_SOURCES[path.name],
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),full_source_inverse_verified=True,
        only_returned_already_verified_output_bytes=True,old_COMP_two_seal_ownership_preserved=True)
    return control,score,outputs


def score_function(h):
    c.require(c.sha(COMP_SCORER) == COMP_SCORER_SHA,'Frozen independent14 scorer arithmetic owner')
    original = c.function_source(COMP_SCORER,'score_function')
    source = c.splice(original,[
        ('def score_function(h):','def inherited_score_function(h):'),
        ('    labels = original.replace("\'current_d50\'","\'P1\'")\n'
         '    c.require(labels.replace("\'P1\'","\'current_d50\'") == original, \'Comparator-label inverse\')\n',
         '    labels = original.replace("\'current_d50\'","\'COMP\'")\n'
         '    c.require(labels.replace("\'COMP\'","\'current_d50\'") == original, \'Comparator-label inverse\')\n'),
        ('comparator label P1 and archived terminal score-role key',
         'comparator label COMP and archived terminal score-role key'),
        ('    source = c.splice(labels,edits)\n','    source = c.splice(labels,edits+SCORE_METADATA_HASH_EDITS)\n')])
    namespace = dict(globals(),c=score_helper())
    exec(compile(source,str(COMP_SCORER)+':same-score-factory-COMP-label','exec'),namespace)
    function,proof = namespace['inherited_score_function'](h)
    proof.update(COMP_score_factory_inverse_verified=True,COMP_score_owner_sha256=COMP_SCORER_SHA,
        original_factory_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_factory_sha256=hashlib.sha256(source.encode()).hexdigest(),
        numeric_operations_and_aggregate_order_unchanged=True,
        only_two_target_free_pin_hash_loops_memoized=True,truth_hashes_never_memoized=True,
        unconditional_final_union_still_required=True)
    return function,proof


def finish_score(result,context):
    control = context['control_score']
    c.require(result['per_role']['COMP'] == control['per_role']['terminal']
        and result['summaries']['COMP'] == control['summaries']['terminal']
        and len(result['per_role']['COMP']) == len(result['per_role']['terminal']) == 14,
        'Exact all14 COMP row dictionaries and all3 group dictionaries')
    acquisition = context['qualified_acquisition']
    c.require(result['all_predictions_verified_before_truth'] and result['baseline_recomputed_exact']
        and result['retain_local_component'] == (result['group_strict_gain_gate']['public8']
            and result['group_strict_gain_gate']['virgin6']),'Only the prospective BOTH-group NC-adjusted gate')
    pareto = {}
    for group in ('public8','virgin6','combined14'):
        parent,new = result['summaries']['COMP'][group],result['summaries']['terminal'][group]
        pareto[group] = dict(observed_raw_and_complete_F_nonincreasing=
            new['mean_raw_observed'] <= parent['mean_raw_observed'] and new['mean_flops'] <= parent['mean_flops'],
            NC_raw_and_complete_F_nonincreasing=
            new['mean_raw_noise_corrected'] <= parent['mean_raw_noise_corrected'] and new['mean_flops'] <= parent['mean_flops'])
    result.update(exact_COMP14_rows_and_three_summaries_replay=True,all15_new_outputs_verified=True,
        all90_artifacts_qualified=True,all15_domains_and_selectors_reverified=True,
        new_three_seal_acquisition_verified=True,old_COMP_two_seal_ownership_preserved=True,
        acquisition_finalization_sha256=context['acquisition_finalization_sha256'],
        native40_expansion_prerequisite_passed=True,additional_bindings_paid_once=True,
        same_existing_arena_qualified=True,dispatch_tariffs_qualified=True,
        initialization_ownership_passed=True,COMP_parameter_initialization_owned=True,
        own_complete_bills=True,full_model_calls=0,retired_call_count=0,
        estimator_executed=False,MPS_executed=False,no_fit=True,no_refit=True,
        no_tuning_or_new_scientific_choice=True,scientific_advancement=result['retain_local_component'],
        native_calls_verified=acquisition['full_model_calls'],native_artifacts_verified=len(acquisition['artifacts']),
        control_admission_proof=context['control_admission_proof'],pareto_diagnostics=pareto,
        public8_is_independent_model_holdout=False,original_virgin6_seeds_unchanged=True)
    guard = context['score_guard']
    result.update(c.final_bind((result['source_bindings'],result.get('input_bindings',{}),result['artifacts']),guard))
    # Original target-file hashes remain fresh and separate from target-free memoization.
    context['h'].check_bindings(result['truth_bindings'])
    guard()


def derived_main():
    c.require(c.sha(COMP_SCORER) == COMP_SCORER_SHA,'Exact qualified separate score main')
    original = c.function_source(COMP_SCORER,'main')
    before_control = """        c.verify_watch(args,context,pins,guard)
        control,_ = c.sealed(c.CONTROL,c.CONTROL_SHA,pins,c.CONTROL_END_SHA,tick=guard)
        control_score,_ = c.sealed(c.CONTROL_SCORE,c.CONTROL_SCORE_SHA,pins,tick=guard)
        h,p,v = c.panel_modules()
        control_outputs = c.five_artifact_verifier(context,c.P1_CANDIDATE,c.P1_SHA,1383913786)(
            control,c.CONTROL,c.RANKS,c.pin_to(pins,guard))
"""
    after_control = """        # Native watch is reverified once in verify_acquisition before any numeric import.
        control,control_score,control_outputs = control_admission(guard,pins,context)
        h,p,v = c.panel_modules()
"""
    source = c.splice(original,[
        ('def main():','def score_main():'),
        ("        pins = {**context['source_bindings'],**context['input_bindings']}\n",
         "        pins = {**context['source_bindings'],**context['input_bindings'],**c.owners(args,guard)}\n"
         "        context['score_guard'] = guard\n"),
        (before_control,after_control),
        ('tl=SimpleNamespace(BANK=c.P1_CANDIDATE,BANK_SHA=c.P1_SHA)',
         'tl=SimpleNamespace(BANK=c.COMP_CANDIDATE,BANK_SHA=c.COMP_SHA)'),
        ("len(result['per_role']['P1']) == 14", "len(result['per_role']['COMP']) == 14"),
        ("result['summaries']['P1'] == control_score['summaries']['terminal'], 'Exact all14 P1 rows/all3 summaries')",
         "result['summaries']['COMP'] == control_score['summaries']['terminal'], 'Exact all14 COMP rows/all3 summaries')\n"
         "        finish_score(result,context)"),
        ("        result.update(proof=proof,prerequisites=context['prerequisites'])\n",
         "        result.update(proof=proof,prerequisites=context['prerequisites'],score_main_source_inverse=SCORE_MAIN_PROOF)\n"),
        ('    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)\n',
         '    c.set_hash_guard(guard)\n'
         '    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)\n'),
        ('    finally:\n        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)\n',
         '    finally:\n        c.set_hash_guard(None)\n'
         '        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)\n')])
    return source,dict(owner=str(COMP_SCORER),owner_sha256=COMP_SCORER_SHA,
        original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_main_sha256=hashlib.sha256(source.encode()).hexdigest(),full_source_inverse_verified=True,
        changed_metadata_admission_and_current_comparator_only=True,resource_bounds_unchanged=True,
        redundant_early_native_watch_check_moved_to_pretruth_acquisition_check=True,
        score_owner_pins_not_added_to_native_context=True)


def main():
    began = time.monotonic()
    if score_arguments('scorer') is None:
        return 0
    source,proof = derived_main()
    namespace = dict(globals(),c=score_helper(),SCORE_MAIN_PROOF=proof)
    exec(compile(source,str(COMP_SCORER)+':same-independent14-main','exec'),namespace)
    code = namespace['score_main']()
    record,end = c.read(c.SCORE_OUT/'result.json'),c.read(c.SCORE_OUT/'completion.json')
    if record['passed']: c.require_fresh_union(record)
    result_sha,completion_sha = c.sha(c.SCORE_OUT/'result.json'),c.sha(c.SCORE_OUT/'completion.json')
    elapsed = time.monotonic()-began
    passed = bool(code == 0 and record['passed'] and record['completed'] and record['integrity_passed']
        and end['passed'] and end['completed'] and end['result_sha256'] == result_sha
        and elapsed < 120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000)
    c.write(c.SCORE_OUT/'finalization.json',dict(passed=passed,completed=passed,result_sha256=result_sha,
        completion_sha256=completion_sha,elapsed_s=elapsed))
    return 0 if passed and time.monotonic()-began < 120 else 1


if __name__ == '__main__':
    raise SystemExit(main())
