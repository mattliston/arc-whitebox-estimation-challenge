"""Independent, separately launched native15 and offline-score supervision.

Native uses the qualified retired-admission supervisor by source inverse;
score uses one owned child. No candidate/source-construction/numerical import.
"""
import json
import os
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace
import panel_common as c


def native_watch(args):
    original = c.function_source(c.HERE/'watch_native_v2.py','main')
    edits = [
        ("    args = e.arguments('watcher')\n", '    args = PANEL_ARGS\n'),
        ("'Full fixed41-call sequence'", "'Full fixed15-call sequence'"),
        ('result[\'completed_bill_count\'] == 41', 'result[\'completed_bill_count\'] == 15'),
        ("'All41 native calls and exact old-zero lineage'", "'All15 panel calls and exact native40/P1 lineage'"),
        ("schema='current_p1_node_composition_admission_watch_v2'", "schema=e.PREFIX+'admission_watch_v2'"),
        ('                first22_output_bill_domain_byte_exact=True,first22_exact_checks=22)',
         '                all15_output_bill_domain_qualified=True,layer0_byte_exact=True)'),
        ('            observe(resources,active=False); last = time.monotonic()\n',
         '            observe(resources,active=False)  # Keep the actual completed sample as deadline origin.\n'),
        ("'Whole1900s and next released bill120s include admission/final sealing'",
         "'Whole1200s and next released bill120s include admission/final sealing'"),
    ]
    source = c.splice(original,edits)
    n = c.e.old()
    # The copied main only uses this metadata view for roster/BUDGET/lineage.
    metadata = SimpleNamespace(**dict(vars(n),VALID=c.RANKS))
    helper = SimpleNamespace(**dict(vars(c),old=lambda:metadata,HELPER=c.e.HELPER,HELPER_SHA=c.e.HELPER_SHA,
        FAIL_SHA=c.e.FAIL_SHA,datetime=c.datetime,timezone=c.timezone))
    owner = c.load(c.HERE/'watch_native_v2.py','node_panel_qualified_native_supervisor',c.WATCH_OWNER_SHA)
    namespace = dict(vars(owner),e=helper,PANEL_ARGS=args,WALL_LIMIT=1200.,ADMISSION_LIMIT=60.,PROGRESS_LIMIT=120.,
        RSS_LIMIT=8_000_000_000,SWAP_LIMIT=0,PERIOD=.2,MAX_INTERVAL=1.)
    exec(compile(source,str(__file__)+':unchanged_owned_native_supervision','exec'),namespace)
    return namespace['main']()


def score_watch(args):
    began = last = time.monotonic()
    c.environment()
    sources,inputs = c.owners(args),{}
    helper = c.load(c.e.HELPER,'node_panel_score_resource_helpers',c.e.HELPER_SHA)
    c.require(not c.SCORE_OUT.exists() and not c.SCORE_WATCH_OUT.exists() and not c.ROOT.is_symlink(),
              'Exclusive separately launched score outputs')
    c.SCORE_WATCH_OUT.mkdir(exist_ok=False)
    result = dict(schema=c.WATCH_SCHEMA,stage='score',role='terminal',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},
        child_execution_completed=False,whole_owned_tree_retired=False,operational_verdict_only=True,
        scientific_advancement=False,truth_arrays_opened=False,numerical_imports=False,
        candidate_sha256=c.CANDIDATE_SHA,build_sha256=c.BUILD_SHA,prior_verdict_reviewed=True,
        numerical_slot_clear=True,auto_launch_next_stage=False,wall_limit_s=120.,combined_rss_limit_bytes=2_000_000_000,
        swap_growth_limit_bytes=0,sample_interval_limit_s=1.,requested_sample_period_s=.2,
        maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_sample_interval_s=0.,samples=0)
    child,before = None,None
    handlers = {}
    def deadline():
        c.require(time.monotonic()-began < 120, 'Whole score120s includes admission/reap/sealing')
    def interrupted(signum,frame):
        raise RuntimeError('Independent14 score supervisor interrupted: '+str(signum))
    def observe(stream=None,active=True):
        nonlocal last
        memory = helper.memory_snapshot(child.pid if child else -1)
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        now = time.monotonic(); interval = now-last
        item = dict(elapsed_s=now-began,interval_s=interval,**memory,swap=swap,swap_growth_bytes=growth,
                    live_child_interval_gate=active)
        if stream: stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
        completed = time.monotonic()
        if active: interval = completed-last
        last = completed
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active: result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        c.require(memory['combined_rss_bytes'] < 2_000_000_000 and growth <= 0, 'Combined score2GB and zero added swap')
        c.require(not active or interval <= 1., 'Actual complete live-child sample<=1s')
        deadline(); return item
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        prior,_ = c.sealed(c.WATCH_OUT,args.acquisition_watch_sha256,inputs,tick=deadline)
        args.context_sha256 = prior['admission_context_sha256']
        args.admission_watch_sha256 = prior['admission_watch_sha256']
        context = c.read_context(args,args.context_sha256,deadline); c.retired_admission(args,context,deadline)
        c.verify_watch(args,context,inputs,deadline)
        sources.update(context['source_bindings']); inputs.update(context['input_bindings'])
        result.update(prerequisites=context['prerequisites'],acquisition_sha256=args.acquisition_sha256,
            acquisition_watch_sha256=args.acquisition_watch_sha256)
        command = c.command(args,'scorer'); result['command'] = command
        c.write(c.SCORE_WATCH_OUT/'started.json',result)
        with (c.SCORE_WATCH_OUT/'resources.jsonl').open('x') as resources,\
             (c.SCORE_WATCH_OUT/'stdout.log').open('xb') as stdout,\
             (c.SCORE_WATCH_OUT/'stderr.log').open('xb') as stderr:
            observe(resources,active=False)
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     env=os.environ.copy(),start_new_session=True)
            result.update(child_pid=child.pid,owned_process_group=child.pid)
            while True:
                code = child.poll(); sample = observe(resources)
                c.require(sample['child_pids'] or code is not None or child.poll() is not None, 'Owned score process visible')
                if code is not None:
                    c.require(not sample['child_pids'], 'No surviving score descendants')
                    child.wait(timeout=2); c.require(code == 0, 'Score diagnostic did not complete')
                    result.update(exit_code=code,whole_owned_tree_retired=True); break
                time.sleep(.2)
            rp,cp = c.SCORE_OUT/'result.json',c.SCORE_OUT/'completion.json'
            digest = c.sha(rp,deadline)
            record,end = c.sealed(c.SCORE_OUT,digest,inputs,tick=deadline)
            c.require(record['schema'] == c.SCORE_SCHEMA and record['mode'] == 'score' and
                record['integrity_passed'] and record['qualification_passed'] and record['sources_unchanged'] and
                type(record['retain_local_component']) is bool and record['all_predictions_verified_before_truth'] and
                record['acquisition_sha256'] == args.acquisition_sha256 and
                record['acquisition_watch_sha256'] == args.acquisition_watch_sha256 and
                all(record['source_bindings'].get(k) == v for k,v in c.owners(args,deadline).items()),
                'Completed sole fixed independent14 score, scientific retention is separate')
            c.inventory(record,inputs,deadline)  # Never traverse truth_bindings.
            result.update(child_result_sha256=digest,child_completion_sha256=c.sha(cp,deadline),
                child_execution_completed=True,retain_local_component=record['retain_local_component'])
            observe(resources,active=False)
        c.bind(sources,deadline); c.bind(inputs,deadline); deadline()
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); child.wait(timeout=2)
                c.require(not helper.memory_snapshot(child.pid)['child_pids'], 'Owned score group retired')
                result['whole_owned_tree_retired'] = True
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            tick = deadline if result['passed'] else None
            for folder,names in ((c.SCORE_WATCH_OUT,('started.json','resources.jsonl','stdout.log','stderr.log')),
                                 (c.SCORE_OUT,('result.json','completion.json'))):
                for name in names:
                    path = folder/name
                    if path.is_file(): result['artifacts'][str(path)] = c.sha(path,tick)
            c.bind(sources,tick); c.bind(inputs,tick)
            if result['passed']:
                c.require(not observe(active=False)['child_pids'], 'Whole score group retired through final seals')
                c.require(c.sha(c.SCORE_OUT/'result.json',deadline) == result['child_result_sha256'] and
                          c.sha(c.SCORE_OUT/'completion.json',deadline) == result['child_completion_sha256'],
                          'Final score result/completion seals')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        result['elapsed_s'] = time.monotonic()-began
        if result['elapsed_s'] >= 120: result.update(passed=False,completed=False)
        c.write(c.SCORE_WATCH_OUT/'result.json',result)
        c.write(c.SCORE_WATCH_OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=c.sha(c.SCORE_WATCH_OUT/'result.json')))
    return 0 if result['passed'] else 1


def main():
    args = c.arguments('watcher')
    if args is None: return 0
    c.environment()
    c.owners(args)
    c.require(not c.ROOT.is_symlink(), 'Fixed nonsymlink panel root')
    c.ROOT.mkdir(exist_ok=True)
    return native_watch(args) if args.stage == 'native' else score_watch(args)


if __name__ == '__main__':
    raise SystemExit(main())
