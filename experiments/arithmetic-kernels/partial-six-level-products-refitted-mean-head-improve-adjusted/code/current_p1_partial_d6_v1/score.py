"""Fixed saved-output D6/P1 native40 score; inert without explicit --run.

No estimator/builder execution, fitting or partial score. The exact score
arithmetic used by COMP score_v3 is reused after new complete acquisition,
retired-admission and independent-monitor checks. PASS is integrity only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
COMP = NEW/'current_p1_node_composition_v1'
RC = NEW/'current_d50_response10_candidate_v1'
OUT = HERE/'score_v1'
PROTOCOL_SHA = 'b4fc1edd8cabec7b59d11ede87de42ce3f3c142aa0b142582bcccac8bdcdaff9'
NATIVE_SHA = 'fe5a9d8639b2c2cbd5d1d7886d6c82e953f13e20820161aafb885d5133592e1f'
WATCH_SHA = 'a4def42901c47f85c6d5d8a9c388b0488a877e565ccf85a3e46fb26f8fbbf167'
BUILDER_SHA = '850de8855cfd50a0ff98e5bce072d401aafd0e3d87282c79b17256dfa1397f08'
BUILD_SHA = '43773c4204d869732b40ca345115e272782fe1d17b64da343bcaf98137b0fab4'
CANDIDATE_SHA = 'eaeaac6ebe75aa272b712b4dfed85f220e876a4236227e77e47933e0d363f1d0'
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
P1_NATIVE_SHA = '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3'
P1_SCORE_SHA = '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5'
ARITHMETIC_SHA = 'e1d4874068b6638c95b4b66063d0abfe2e5327a07cb0ebfef1c442105e037dfe'
HELPER_SHA = 'e4a9ef56dd5309e33cde2d67ad983e8655918af12909a8fdec58db0292c27eaf'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
BUDGET, WALL_LIMIT, RSS_LIMIT = 2**41, 120., 1_000_000_000


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed helper before import: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    adapter = load(RC/'score_v2.py','partial_d6_saved_fp32_adapter',HELPER_SHA)
    source = adapter.derived_source()
    namespace = dict(__file__=str(RC/'score_v2.py'),__name__='partial_d6_saved_output_helpers')
    exec(compile(source,str(__file__)+':saved_fp32_helpers','exec'),namespace)
    return namespace


def arithmetic():
    """The truth loop, exact P1 replay, mean/cost/win gates remain verbatim."""
    original = (COMP/'score.py').read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == ARITHMETIC_SHA,
            'Original arithmetic source preserved by completed COMP score_v3')
    anchor = '    records = []\n'
    require(original.count(anchor) == 1, 'Unique fixed score tail')
    start = original.index(anchor)
    stop = original.index("\n\nif __name__=='__main__':",start)
    tail = original[start:stop]
    header = ('def score_arrays(result,out,pins,h,guard,np,parents,arrays,baseline,pscore,acquisitions):\n'
              "    require,sha,read,write,bind = [h[k] for k in ('require','sha','read','write','bind')]\n")
    source = header+tail
    require(original[:start]+source[len(header):]+original[stop:] == original,
            'Whole original source restored after extracting unchanged arithmetic')
    namespace = dict(P2=P2,VALID=VALID,BUDGET=BUDGET,math=math)
    exec(compile(source,str(__file__)+':unchanged_score_arithmetic','exec'),namespace)
    proof = dict(original_scorer_sha256=ARITHMETIC_SHA,
        arithmetic_tail_sha256=hashlib.sha256(tail.encode()).hexdigest(),
        source_inverse_verified=True,raw_cost_aggregation_and_gates_unchanged=True,
        result_role_names={'P1':'retained P1','composition':'fixed partial-D6 P1'},
        completed_source_owner=str(COMP/'score_v3.py'))
    return namespace['score_arrays'],proof


def source_pins(args, e):
    owners = e.owners(args)
    owners.update({str(Path(__file__).resolve()):args.source_sha256,
        str(COMP/'score.py'):ARITHMETIC_SHA,
        str(COMP/'score_v2.py'):'caf8ba0dca7a1b03a8db47c6a8d4ddde98b3e1705604ac08934b63fff1105229',
        str(COMP/'score_v3.py'):'566f7c255ea7da36aa43a67acab1e3f98cdb5feaad9d46beac6cf14fa0fdcd58',
        str(RC/'score_v2.py'):HELPER_SHA,
        str(RC/'score.py'):'3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',
        str(RC/'SCORE_DTYPE_V2.md'):'9995cd8ae375ae0decef7615e1b08678941510f00f1bd3bcbb7436a649bf30ac'})
    # e.owners explicitly includes TL/watch_fit.py, original TL/native/common,
    # BD/native_v2, RC/native, census lifecycle, monitor/mock and classifier.
    require(owners[str(e.HELPER)] == e.HELPER_SHA, 'Transitive TL resource helper is an explicit source owner')
    e.bind(owners)
    return owners


def check_watch(args, e, watch, native, context, h, guard):
    read = h['read']
    flags = ('passed','completed','hashes_unchanged','sources_unchanged',
        'whole_owned_tree_retired','child_execution_completed','admission_process_reaped_before_native',
        'observational_groups_retired','additional_bindings_paid_once','same_existing_arena_qualified',
        'dispatch_tariffs_qualified','own_complete_bills','prior_verdict_reviewed','numerical_slot_clear',
        'progress_requires_closed_lifecycle')
    require(watch['schema'] == e.WATCH_SCHEMA and watch['stage'] == 'native' and watch['role'] == 'terminal'
            and all(watch.get(k) is True for k in flags) and watch['truth_accessed'] is False
            and watch['truth_arrays_opened'] is False and watch['numerical_imports'] is False
            and watch['candidate_sha256'] == CANDIDATE_SHA and watch['build_sha256'] == BUILD_SHA
            and watch['child_result_sha256'] == args.native_sha256
            and watch['child_completion_sha256'] == sha(e.OUTPUT/'completion.json')
            and watch['prerequisites'] == context['prerequisites']
            and watch['completed_bill_count'] == len(watch['progress']) == 41,
            'Complete independently retired target-free native40/repeat acquisition')
    require(watch['wall_limit_s'] == 1900 and watch['admission_limit_s'] == 60
            and watch['no_completed_bill_limit_s'] == 120 and watch['combined_rss_limit_bytes'] == 8_000_000_000
            and watch['swap_growth_limit_bytes'] == 0 and watch['sample_interval_limit_s'] == 1
            and 0 < watch['elapsed_s'] < 1900 and 0 < watch['maximum_sampled_rss_bytes'] < 8_000_000_000
            and watch['maximum_swap_growth_bytes'] == 0
            and watch['monitor_mock_result_sha256'] == e.MONITOR_TEST_RESULT_SHA,
            'Original whole-stage/resource gates and independently certified transport')
    wrapper = load(HERE/'watch_native.py','partial_d6_reviewed_watch_source',WATCH_SHA)
    _source,proof = wrapper.derived_source()
    require(watch['watcher_source_inverse'] == proof and proof['resource_limits_unchanged']
            and proof['next_completed_sample_absolute_deadline'], 'Exact executed monitor source inverse')
    phases = watch['phase_records']
    require([x['phase'] for x in phases] == ['admission','native']
            and len({x['pid'] for x in phases}) == 2
            and all(type(x['pid']) is int and x['pid'] > 1 and x['process_group'] == x['pid']
                    and x['exit_code'] == 0 and x['whole_owned_tree_retired'] for x in phases)
            and 0 < phases[0]['elapsed_s'] < 60 and 0 < phases[1]['elapsed_s'] < 1900,
            'Full source admission reaped before distinct complete native process')
    intervals = watch['completed_sample_intervals_s']
    require(intervals and all(math.isfinite(x) and 0 < x <= 1 for x in intervals)
            and max(intervals) == watch['maximum_sample_interval_s'], 'Every authoritative completed sample within1s')
    path = e.WATCH_OUT/'resources.jsonl'
    require(watch['artifacts'][str(path)] == sha(path) and not path.is_symlink(), 'Immutable actual resource log')
    counts = {'admission':0,'native':0}; samples = []
    with path.open() as stream:
        for line in stream:
            require(len(line) <= 2*1024**2, 'Bounded resource record')
            item = json.loads(line); guard()
            require(item['phase'] in counts and item['one_shared_process_table']
                    and item['observational_commands_reaped'] and item['foreign_numerical_workers'] == []
                    and item['cached_observation'] is False and item['observation_retry_count'] == 0
                    and 0 < item['combined_rss_bytes'] < 8_000_000_000 and item['swap_growth_bytes'] <= 0
                    and 0 <= item['no_completed_bill_s'] < 120 and 0 <= item['elapsed_s'] < 1900,
                    'Every captured shared-table resource/inventory/progress gate')
            commands = item['transport']
            require([x['argv'] for x in commands] == [
                ['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,command='],['/usr/sbin/sysctl','vm.swapusage']]
                and len({x['pid'] for x in commands}) == 2
                and all(type(x['pid']) is int and x['pid'] > 1 and 0 <= x['finished_since_start_s'] < 1
                        for x in commands), 'Only the two completed owned observational commands')
            if item['live_child_interval_gate']:
                counts[item['phase']] += 1
            samples.append((item['combined_rss_bytes'],item['swap_growth_bytes']))
    final = watch['final_resource_observation']
    require(final['live_child_interval_gate'] is False and not final['child_pids']
            and final['foreign_numerical_workers'] == [] and final['observational_commands_reaped']
            and 0 < final['completed_interval_s'] <= 1,
            'Fresh completed final observation after all experiment/observational groups retired')
    samples.append((final['combined_rss_bytes'],final['swap_growth_bytes']))
    require(all(counts.values()) and sum(counts.values()) == len(intervals)
            and watch['samples'] == len(samples)
            and max(x[0] for x in samples) == watch['maximum_sampled_rss_bytes']
            and max(0,max(x[1] for x in samples)) == watch['maximum_swap_growth_bytes'],
            'Resource-log counts/maxima agree with authoritative completed intervals')
    prior = 0.
    for call,(item,trial) in enumerate(zip(watch['progress'],native['trials'])):
        bp,lp = e.OUTPUT/f'call_{call:02d}_bill.json',e.OUTPUT/f'call_{call:02d}_lifecycle.json'
        elapsed = item['observed_elapsed_s']
        require(item['call'] == call and item['rank'] == trial['rank']
                and item['path'] == str(bp) and item['lifecycle_path'] == str(lp)
                and item['sha256'] == sha(bp) and item['lifecycle_sha256'] == sha(lp)
                and read(bp) == trial and 0 < elapsed-prior < 120,
                'Each next exact own bill plus released lifecycle met the unchanged deadline')
        prior = elapsed
    require(watch['first_progress_elapsed_s'] == watch['progress'][0]['observed_elapsed_s']
            and watch['last_progress_elapsed_s'] == prior and watch['elapsed_s']-prior < 120
            and watch['initialization_flops'] == native['initialization_flops']
            and watch['primary_dispatch_count'] == native['primary_dispatch_count'],
            'Admission/sealing deadlines and measured startup/dispatch summary remain bound')


def admit(args, e, result, pins, h, guard):
    sealed,read,bind = (h[k] for k in ('sealed','read','bind'))
    native = sealed(e.OUTPUT,args.native_sha256,pins)
    watch = sealed(e.WATCH_OUT,args.watch_sha256,pins)
    args.context_sha256 = watch['admission_context_sha256']
    args.admission_watch_sha256 = watch['admission_watch_sha256']
    context = e.read_context(args,args.context_sha256,guard)
    retired = e.retired_admission(args,context)
    for group in ('source_bindings','input_bindings'):
        bind(context[group],pins)
    result['source_bindings'].update(context['source_bindings'])
    bind(e.transport_admission(guard),pins)
    for path,digest in e.owners(args,guard).items():
        require(native['source_bindings'].get(path) == watch['source_bindings'].get(path) == digest,
                'Exact reviewed source owners in both actual native and monitor receipts')
    require(watch['admission_result_sha256'] == sha(e.ADMISSION/'result.json')
            and watch['admission_completion_sha256'] == sha(e.ADMISSION/'completion.json')
            and retired['admission_pid'] == watch['phase_records'][0]['pid'], 'Same immutable retired producer')
    admitted = read(e.ADMISSION/'result.json')
    require(0 < admitted['elapsed_s'] < 60 and 0 < admitted['peak_rss_bytes'] < 2_000_000_000,
            'Full lexical admission met its own60s/2GB limit')
    e.verify_acquisition(native,read(e.OUTPUT/'completion.json'),args.native_sha256,args,context,guard)
    check_watch(args,e,watch,native,context,h,guard)
    baseline = sealed(P1/'native2_terminal_v1',P1_NATIVE_SHA,pins)
    pscore = sealed(P1/'score2_v1',P1_SCORE_SHA,pins)
    require(baseline['candidate_sha256'] == P1_SHA and baseline['truth_accessed'] is False
            and [t['rank'] for t in baseline['trials']] == list(VALID+(37,))
            and len(baseline['input_preparation']) == 41 and baseline['all16_domains_qualified']
            and baseline['all47_selectors_qualified'] and baseline['own_complete_bills'],
            'Exact retained P1 native40 cold/warm comparator')
    require(pscore['candidate_sha256'] == P1_SHA and pscore['integrity_passed']
            and pscore['advances_to_independent14'] and len(pscore['records']) == 40
            and [x['rank'] for x in pscore['records']] == list(VALID), 'Original fixed1e9 P1 score/target frame')
    lifecycle = e.load(e.CENSUS,'partial_d6_score_lifecycle',e.CENSUS_SHA)
    timings = e.base().legacy().TIMINGS
    for call,(trial,prepared) in enumerate(zip(baseline['trials'],baseline['input_preparation'])):
        rank = (VALID+(37,))[call]
        bp,lp = P1/'native2_terminal_v1'/f'call_{call:02d}_input_bill.json',P1/'native2_terminal_v1'/f'call_{call:02d}_input_reset.json'
        require(trial['call'] == prepared['call'] == call and trial['rank'] == prepared['rank'] == rank
                and prepared['flops'] == 16*1024*1024 and prepared['layers'] == 16
                and not prepared['charged_to_participant'] and all(prepared[k] for k in
                    ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed'))
                and prepared['weight_file_sha256'] == context['reference_by_rank'][rank]['weight_file_sha256']
                and all(math.isfinite(prepared[k]) and prepared[k] >= 0 for k in timings)
                and prepared['wall_s'] < 120 and prepared['context_wall_s'] < 120
                and 0 < prepared['peak_rss_bytes'] < 8_000_000_000 and read(bp) == prepared
                and baseline['artifacts'][str(bp)] == sha(bp) and baseline['artifacts'][str(lp)] == sha(lp),
                'Original P1 official-input ownership/lifecycle on the same canonical weights')
        lifecycle.lifecycle(dict(prepared,lifecycle_receipt=str(lp)),bp,lp)
    result.update(admission_context_sha256=args.context_sha256,admission_watch_sha256=args.admission_watch_sha256,
        complete_native_verifier_passed=True,complete_monitor_verifier_passed=True,
        original_startup_and_dispatch_counters_verified=True,observational_groups_retired=True,
        native_sha256=args.native_sha256,native_watch_sha256=args.watch_sha256,
        no_partial_acquisition_scored=True,initialization_accounting=native['initialization_accounting'],
        primary_dispatch_count=native['primary_dispatch_count'])
    return native,baseline,pscore


def score_body(args,e,result,pins,h,guard):
    native,baseline,pscore = admit(args,e,result,pins,h,guard)
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'All admission before numerical imports')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Qualified OFFPY NumPy scoring frame')
    values = [h['trial_output'](np,e.OUTPUT,t,pins) for t in native['trials']]
    parents = [h['trial_output'](np,P1/'native2_terminal_v1',t,pins) for t in baseline['trials']]
    for record,arrays in ((native,values),(baseline,parents)):
        require(arrays[0].tobytes() == arrays[-1].tobytes()
                and record['trials'][0]['flops']-record['trials'][-1]['flops'] == record['initialization_flops'],
                'Own complete cold/repeat bills and exact served repeat; no initialization subtraction')
    for call,rank in enumerate(VALID):
        require(all(record['trials'][call]['call'] == call and record['trials'][call]['rank'] == rank
                    and record['trials'][call]['includes_initialization'] is (call == 0) for record in (native,baseline))
                and values[call][0].tobytes() == parents[call][0].tobytes(),
                'Matched primary ownership and unchanged Gaussian first layer')
    scorer,proof = arithmetic(); result['score_arithmetic_proof'] = proof
    h['bind'](result['source_bindings'],{}); h['bind'](pins,{}); guard()
    seal = dict(truth_accessed=False,source_bindings=result['source_bindings'],input_bindings=dict(pins),
        candidate_sha256=CANDIDATE_SHA,baseline_native_sha256=P1_NATIVE_SHA,
        native_sha256=args.native_sha256,native_watch_sha256=args.watch_sha256,
        admission_context_sha256=args.context_sha256,admission_watch_sha256=args.admission_watch_sha256,
        all41_terminal_verified=True,all41_P1_verified=True,all_sources_outputs_bills_lifecycles_verified=True,
        all_monitor_gates_verified=True,repeat_excluded=True,score_arithmetic_proof=proof)
    write(OUT/'predictions_and_bills_sealed.json',seal)
    result.update(all_predictions_verified_before_truth=True,
        predictions_and_bills_seal_sha256=sha(OUT/'predictions_and_bills_sealed.json'))
    result['artifacts'][str(OUT/'predictions_and_bills_sealed.json')] = result['predictions_and_bills_seal_sha256']
    return scorer(result,OUT,pins,h,guard,np,parents,{'terminal':values},baseline,pscore,
                  {'terminal':(e.OUTPUT,native)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','native-source','watch-source','native','watch','protocol'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for k,v in vars(args).items() if k != 'run'),
            'Explicit reviewed source and complete native/watch digests')
    require((args.native_source_sha256,args.watch_source_sha256,args.protocol_sha256) ==
            (NATIVE_SHA,WATCH_SHA,PROTOCOL_SHA) and sha(__file__) == args.source_sha256,
            'Only the reviewed fixed endpoint and scorer')
    require(not OUT.exists() and not OUT.is_symlink() and not HERE.is_symlink(), 'Exclusive score; no overwrite/resume')
    OUT.mkdir(exist_ok=False)
    result = dict(schema='current_p1_partial_d6_score_v1',stage='score',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,integrity_passed=False,
        source_bindings={},input_bindings={},artifacts={},truth_bindings={},truth_accessed=False,
        ranks=list(VALID),candidate_sha256=CANDIDATE_SHA,baseline_sha256=P1_SHA,build_sha256=BUILD_SHA,
        estimator_executed=False,training_or_refit_performed=False,old_max_read_timing_gate_still_failed=True,
        official_score=False,production_qualified=False,independent_panel_qualified=False,
        automatic_next_stage=False,advances_to_independent14=False,repeat_excluded=True,
        wall_limit_s=WALL_LIMIT,rss_limit_bytes=RSS_LIMIT,started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*unused):
        require(time.monotonic()-began < WALL_LIMIT and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS_LIMIT,
                'Complete saved-output score120s/1GB')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    pins = result['input_bindings']; h = None
    try:
        e = load(HERE/'native.py','partial_d6_score_native_owner',NATIVE_SHA)
        e.environment()
        args.worker_sha256,args.watcher_sha256 = NATIVE_SHA,WATCH_SHA
        args.builder_sha256,args.build_sha256,args.candidate_sha256 = BUILDER_SHA,BUILD_SHA,CANDIDATE_SHA
        result['source_bindings'] = source_pins(args,e)
        # Authenticate every source/metadata dependency of the dtype-only helper
        # before invoking it. Its failed historical score remains untouched.
        e.bind({str(RC/'score_v1/result.json'):'ce4364aafa13c6277bfea4f5f9abda2207f44fac17db26d2a128ec777b4ce897',
                str(RC/'score_v1/completion.json'):'5b3225b5ad01ae8c7d9bcc61f4893ebc909d2228e7b251df492aae45609b9d8c'})
        h = helpers(); h['bind'](result['source_bindings'],pins)
        h['bind']({str(RC/'score_v1/result.json'):'ce4364aafa13c6277bfea4f5f9abda2207f44fac17db26d2a128ec777b4ce897',
                   str(RC/'score_v1/completion.json'):'5b3225b5ad01ae8c7d9bcc61f4893ebc909d2228e7b251df492aae45609b9d8c'},pins)
        write(OUT/'started.json',result); result['artifacts'][str(OUT/'started.json')] = sha(OUT/'started.json')
        score_body(args,e,result,pins,h,guard)
        h['bind'](result['source_bindings'],{}); h['bind'](pins,{}); h['bind'](result['artifacts'],{}); guard()
        result.update(sources_unchanged=True,hashes_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
            integrity_passed=False,advances_to_independent14=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if result['elapsed_s'] >= WALL_LIMIT or result['peak_rss_bytes'] >= RSS_LIMIT:
            result.update(passed=False,completed=False,integrity_passed=False,advances_to_independent14=False)
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','summary','mean_adjusted_ratio',
        'adjusted_wins','classification','advances_to_independent14','elapsed_s','peak_rss_bytes','error')},allow_nan=False),flush=True)
    return 0 if result['passed'] and result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
