"""Two nonoverlapping owned processes, one unchanged whole-stage budget.

The admission child is fully reaped before the native child is launched.
No source-composition replay occurs in this lightweight supervisor.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback
import endpoint_v2 as e

WALL_LIMIT, ADMISSION_LIMIT, PROGRESS_LIMIT = 1900.,60.,120.
RSS_LIMIT, SWAP_LIMIT, PERIOD, MAX_INTERVAL = 8_000_000_000,0,.2,1.


def main():
    args = e.arguments('watcher')
    if args is None: return 0
    began = last_progress = time.monotonic()
    e.environment()
    e.require(args.prior_verdict_reviewed and args.numerical_slot_clear, 'Explicit reviewed prior verdict and free numerical slot')
    def deadline():
        now = time.monotonic()
        e.require(now-began < WALL_LIMIT and now-last_progress < PROGRESS_LIMIT,
                  'Whole1900s and next released bill120s include admission/final sealing')
    sources,inputs = e.owners(args,deadline),{}
    for folder in (e.ADMISSION,e.OUTPUT,e.WATCH_OUT):
        e.require(not folder.exists() and not folder.is_symlink() and not e.HERE.is_symlink(), 'Exclusive full restart paths')
    helper = e.load(e.HELPER,'node_p1_v2_resource_helpers',e.HELPER_SHA)
    n = e.old()
    census = e.load(n.CENSUS,'node_p1_v2_lifecycle_helpers',n.CENSUS_SHA)
    sources[str(n.CENSUS)] = n.CENSUS_SHA
    sequence = n.VALID+(n.VALID[0],)
    e.WATCH_OUT.mkdir(exist_ok=False)
    result = dict(schema=e.WATCH_SCHEMA,stage='native',role='terminal',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},
        candidate_sha256=n.CANDIDATES['terminal'],build_sha256=n.BUILD_SHA,failed_v1_watch_sha256=e.FAIL_SHA,
        prior_verdict_reviewed=True,numerical_slot_clear=True,auto_launch_next_stage=False,
        old_NODE_gate_still_failed=True,truth_accessed=False,truth_arrays_opened=False,numerical_imports=False,
        operational_verdict_only=True,scientific_advancement=False,child_execution_completed=False,
        whole_owned_tree_retired=False,admission_process_reaped_before_native=False,
        completed_bill_count=0,progress=[],phase_records=[],phase_commands=[],
        first_progress_elapsed_s=None,last_progress_elapsed_s=None,progress_requires_closed_lifecycle=True,
        wall_limit_s=WALL_LIMIT,admission_limit_s=ADMISSION_LIMIT,combined_rss_limit_bytes=RSS_LIMIT,
        swap_growth_limit_bytes=SWAP_LIMIT,no_completed_bill_limit_s=PROGRESS_LIMIT,
        requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_sample_interval_s=0.,samples=0,
        started_utc=e.datetime.now(e.timezone.utc).isoformat(),
        limitations=['RSS and system swap are sampled, not continuous peak guarantees.',
            'Only the current fresh owned group is signalled; the admission group is retired before native launch.',
            'No native call, scoring or model cache reset occurs during admission; PASS is acquisition integrity only.'])
    child,before,phase,phase_start,last = None,None,None,None,began
    handlers = {}

    def interrupted(signum,frame):
        raise RuntimeError('Endpoint supervisor interrupted: '+str(signum))

    def progress():
        nonlocal last_progress
        found = []
        for path in e.OUTPUT.glob('call_*_bill.json'):
            if path.name.endswith('_input_bill.json'): continue
            match = re.fullmatch(r'call_([0-9]{2})_bill\.json',path.name)
            e.require(match is not None and not path.is_symlink(), 'Only original ordered bill names')
            found.append(int(match[1]))
        e.require(sorted(found) == list(range(len(found))) and len(found) <= len(sequence), 'Full fixed41-call sequence')
        for call in range(result['completed_bill_count'],len(found)):
            bp,lp = e.OUTPUT/f'call_{call:02d}_bill.json',e.OUTPUT/f'call_{call:02d}_lifecycle.json'
            if not lp.exists(): break
            try:
                bill = e.read(bp); census.lifecycle(bill,bp,lp)
            except json.JSONDecodeError:
                break
            e.require(bill['call'] == call and bill['rank'] == sequence[call]
                      and bill['includes_initialization'] is (call == 0)
                      and type(bill['flops']) is int and 0 < bill['flops'] <= n.BUDGET,
                      'Next own cold/warm bill and released lifecycle')
            deadline()  # A late lifecycle never resets an already expired clock.
            now = time.monotonic(); last_progress = now
            result['completed_bill_count'] = call+1
            if call == 0: result['first_progress_elapsed_s'] = now-began
            result['last_progress_elapsed_s'] = now-began
            result['progress'].append(dict(call=call,rank=sequence[call],path=str(bp),sha256=e.sha(bp),
                lifecycle_path=str(lp),lifecycle_sha256=e.sha(lp),observed_elapsed_s=now-began))

    def observe(stream=None,active=True):
        nonlocal last
        memory = helper.memory_snapshot(child.pid if child is not None else -1)
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        progress()
        now = time.monotonic(); interval,last = now-last,now
        item = dict(phase=phase,elapsed_s=now-began,interval_s=interval,**memory,swap=swap,
            swap_growth_bytes=growth,completed_bill_count=result['completed_bill_count'],
            no_completed_bill_s=now-last_progress,live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active: result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        if stream:
            stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
        e.require(memory['combined_rss_bytes'] < RSS_LIMIT and growth <= SWAP_LIMIT, 'Combined8GB and zero added swap')
        e.require(not active or interval <= MAX_INTERVAL, 'Observed live-child sample interval<=1s')
        e.require(not active or phase != 'admission' or now-phase_start < ADMISSION_LIMIT, 'Independent admission60s')
        deadline()
        return item

    def supervise(command, name, resources):
        nonlocal child,phase,phase_start,last
        e.require(child is None or (child.poll() is not None and not helper.memory_snapshot(child.pid)['child_pids']),
                  'Previous owned process/group fully retired before another launch')
        phase,phase_start = name,time.monotonic()
        result['phase_commands'].append(dict(phase=name,command=command))
        with (e.WATCH_OUT/(name+'_stdout.log')).open('xb') as stdout,\
                (e.WATCH_OUT/(name+'_stderr.log')).open('xb') as stderr:
            observe(resources,active=False); last = time.monotonic()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     env=os.environ.copy(),start_new_session=True)
            item_record = dict(phase=name,pid=child.pid,process_group=child.pid,whole_owned_tree_retired=False)
            result['phase_records'].append(item_record)
            while True:
                code = child.poll(); sample = observe(resources)
                e.require(sample['child_pids'] or code is not None or child.poll() is not None, 'Live owned tree visible')
                if code is not None:
                    e.require(not sample['child_pids'], 'No surviving phase descendants')
                    child.wait(timeout=2)
                    item_record.update(exit_code=code,elapsed_s=time.monotonic()-phase_start,whole_owned_tree_retired=True)
                    e.require(code == 0, name+' did not complete successfully')
                    if name == 'admission': e.require(item_record['elapsed_s'] < ADMISSION_LIMIT, 'Complete admission/reap60s')
                    return item_record
                time.sleep(PERIOD)

    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        e.write(e.WATCH_OUT/'started.json',result)
        with (e.WATCH_OUT/'resources.jsonl').open('x') as resources:
            admitted = supervise(e.command(args,'admitter'),'admission',resources)
            rp,cp,jp = e.ADMISSION/'result.json',e.ADMISSION/'completion.json',e.ADMISSION/'context.json'
            args.context_sha256 = e.sha(jp,deadline)
            context = e.read_context(args,args.context_sha256,deadline)
            sources.update(context['source_bindings']); inputs.update(context['input_bindings'])
            result.update(admission_result_sha256=e.sha(rp,deadline),admission_completion_sha256=e.sha(cp,deadline),
                admission_context_sha256=args.context_sha256,prerequisites=context['prerequisites'])
            observe(resources,active=False)
            handoff = dict(schema='current_p1_node_composition_admission_watch_v2',passed=True,completed=True,
                whole_owned_tree_retired=True,admission_pid=admitted['pid'],
                admission_elapsed_s=admitted['elapsed_s'],whole_elapsed_s=time.monotonic()-began,
                context_sha256=args.context_sha256,reviewed_owners=e.owners(args,deadline),
                maximum_sampled_rss_bytes=result['maximum_sampled_rss_bytes'],
                maximum_swap_growth_bytes=result['maximum_swap_growth_bytes'],
                maximum_sample_interval_s=result['maximum_sample_interval_s'])
            deadline(); e.write(e.WATCH_OUT/'admission_retired.json',handoff)
            args.admission_watch_sha256 = e.sha(e.WATCH_OUT/'admission_retired.json',deadline)
            e.retired_admission(args,context); inputs.update(context['input_bindings'])
            result.update(admission_process_reaped_before_native=True,admission_watch_sha256=args.admission_watch_sha256)
            supervise(e.command(args,'worker',args.context_sha256,args.admission_watch_sha256),'native',resources)
            rp,cp = e.OUTPUT/'result.json',e.OUTPUT/'completion.json'
            record,end = e.read(rp),e.read(cp)
            result.update(child_result_sha256=e.sha(rp,deadline),child_completion_sha256=e.sha(cp,deadline))
            e.verify_native(record,end,result['child_result_sha256'],args,context)
            e.require(result['completed_bill_count'] == 41 and record['prerequisites'] == context['prerequisites'],
                      'All41 native calls and exact old-zero lineage')
            for item,trial in zip(result['progress'],record['trials']):
                e.require(e.read(item['path']) == trial and e.sha(item['path'],deadline) == item['sha256']
                          and e.sha(item['lifecycle_path'],deadline) == item['lifecycle_sha256'], 'Observed lifecycle seals unchanged')
            result.update(child_execution_completed=True,whole_owned_tree_retired=True,
                first22_output_bill_domain_byte_exact=True,first22_exact_checks=22)
            observe(resources,active=False)
        e.bind(sources,deadline); e.bind(inputs,deadline); deadline()
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); child.wait(timeout=2)
                e.require(not helper.memory_snapshot(child.pid)['child_pids'], 'Owned descendants retired after failure')
                if result['phase_records']:
                    result['phase_records'][-1].update(whole_owned_tree_retired=True,exit_code=child.returncode)
                result['whole_owned_tree_retired'] = all(x['whole_owned_tree_retired'] for x in result['phase_records'])
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            tick = deadline if result['passed'] else None
            for folder,names in ((e.WATCH_OUT,('started.json','resources.jsonl','admission_retired.json',
                'admission_stdout.log','admission_stderr.log','native_stdout.log','native_stderr.log')),
                (e.ADMISSION,('result.json','completion.json','context.json')),(e.OUTPUT,('result.json','completion.json'))):
                for name in names:
                    path = folder/name
                    if path.is_file(): result['artifacts'][str(path)] = e.sha(path,tick)
            e.bind(sources,tick); e.bind(inputs,tick)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                last_sample = observe(active=False)
                e.require(not last_sample['child_pids'] and all(x['whole_owned_tree_retired'] for x in result['phase_records']),
                          'Both owned groups retired through final sealing')
                e.require(e.sha(e.OUTPUT/'result.json',deadline) == result['child_result_sha256']
                          and e.sha(e.OUTPUT/'completion.json',deadline) == result['child_completion_sha256'],
                          'Final native result/completion immutable')
                result['final_resource_observation'] = last_sample
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-began,finished_utc=e.datetime.now(e.timezone.utc).isoformat())
        if result['elapsed_s'] >= WALL_LIMIT:
            result.update(passed=False,completed=False)
        e.write(e.WATCH_OUT/'result.json',result)
        e.write(e.WATCH_OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=e.sha(e.WATCH_OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','completed_bill_count',
        'maximum_sampled_rss_bytes','maximum_sample_interval_s','error','cleanup_error','final_sealing_error')},allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
