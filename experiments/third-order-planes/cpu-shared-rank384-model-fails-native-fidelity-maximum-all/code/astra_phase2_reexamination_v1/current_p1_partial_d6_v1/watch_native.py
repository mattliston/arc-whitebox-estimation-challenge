"""Source-inverse native40 watchdog: retired admission, then one OFFPY worker.

Unchanged 1900s/120s/8GB/zero-swap gates. Mock-certified ps/sysctl transport
has no dependency on any MPS model, training payload, gradient or target.
No execution or automatic launch occurs on import.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'current_p1_node_composition_v1/watch_native_v2.py'
PARENT_SHA = '8ffc9ae68388065d49c9e39286893f5da8f5eeabe000db625643fd340442cd28'


def derived_source():
    """No resource/model calls: authenticate and invert small harness text."""
    def require(value,message):
        if not value: raise RuntimeError(message)
    original = PARENT.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == PARENT_SHA, 'Frozen retired-process watchdog')
    start,stop = original.index('    def observe('),original.index('\n    def supervise(')
    old_observe = original[start:stop]
    new_observe = '''    def observe(stream=None,active=True):
        nonlocal last
        previous = last if active else time.monotonic()
        absolute = min(previous+MAX_INTERVAL,began+WALL_LIMIT,last_progress+PROGRESS_LIMIT)
        if active and phase == 'admission': absolute = min(absolute,phase_start+ADMISSION_LIMIT)
        snapshot = monitor.collect(absolute)
        memory = monitor.scope(snapshot['rows'],os.getpid(),child.pid if child is not None else None)
        foreign = inventory_owner.other_workers(snapshot['rows'],memory.pop('allowed'))
        swap = snapshot['swap']; growth = swap['used_bytes']-before['used_bytes']
        progress()
        now = time.monotonic()
        item = dict(phase=phase,elapsed_s=now-began,**memory,swap=swap,
            swap_growth_bytes=growth,completed_bill_count=result['completed_bill_count'],
            no_completed_bill_s=now-last_progress,live_child_interval_gate=active,
            foreign_numerical_workers=foreign,transport=snapshot['command_records'],
            one_shared_process_table=True,observational_commands_reaped=snapshot['commands_reaped'],
            cached_observation=False,observation_retry_count=0,
            log_timestamp_scope='Before flush; authoritative completed intervals are sealed in the result')
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if stream:
            stream.write(json.dumps(item,allow_nan=False)+'\\n'); stream.flush()
        e.require(not foreign and snapshot['commands_reaped'] and not monitor.PENDING,
                  'No competing numerical workers; observation groups retired')
        e.require(memory['combined_rss_bytes'] < RSS_LIMIT and growth <= SWAP_LIMIT, 'Combined8GB and zero added swap')
        e.require(not active or phase != 'admission' or now-phase_start < ADMISSION_LIMIT, 'Independent admission60s')
        deadline()
        completed,interval = monitor.complete(previous,absolute)
        last = completed
        item.update(completed_elapsed_s=completed-began,completed_interval_s=interval)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
            result['completed_sample_intervals_s'].append(interval)
        return item
'''
    edits = [
        ('import endpoint_v2 as e\n', '# Reviewed native/monitor/classifier modules supplied by the wrapper.\n'),
        ('sources,inputs = e.owners(args,deadline),{}',
         'sources,inputs = e.owners(args,deadline),e.transport_admission(deadline)'),
        ('failed_v1_watch_sha256=e.FAIL_SHA,','old_max_read_result_sha256=e.FAIL_SHA,'),
        ('old_NODE_gate_still_failed=True,','old_max_read_timing_gate_still_failed=True,'),
        ('completed_bill_count=0,progress=[],phase_records=[],phase_commands=[],',
         'completed_bill_count=0,progress=[],phase_records=[],phase_commands=[],completed_sample_intervals_s=[],\n'
         '        watcher_source_inverse=WATCH_PROOF,monitor_mock_result_sha256=e.MONITOR_TEST_RESULT_SHA,\n'
         "        monitor_scope='Target-free command transport only; no MPS model or training-state dependency',"),
        (old_observe,new_observe),
        ("e.require(child is None or (child.poll() is not None and not helper.memory_snapshot(child.pid)['child_pids']),",
         "e.require(child is None or (child.poll() is not None and not monitor.group_alive(child.pid)),"),
        ('observe(resources,active=False); last = time.monotonic()', 'observe(resources,active=False)  # Preserve actual completed-sample origin.'),
        ('                    child.wait(timeout=2)\n',
         "                    child.wait(timeout=2)\n                    e.require(not monitor.group_alive(child.pid) and not monitor.PENDING, 'Whole phase/observation groups retired')\n"),
        ("        before = helper.swap_state(); result['before_swap'] = before\n",
         "        initial = monitor.collect(min(time.monotonic()+1,began+WALL_LIMIT,last_progress+PROGRESS_LIMIT))\n"
         "        initial_scope = monitor.scope(initial['rows'],os.getpid(),None)\n"
         "        e.require(not inventory_owner.other_workers(initial['rows'],initial_scope['allowed'])\n"
         "                  and initial_scope['combined_rss_bytes'] < RSS_LIMIT and not monitor.PENDING,\n"
         "                  'Prelaunch process inventory and combined RSS; no competing numerics')\n"
         "        before = initial['swap']; result['before_swap'] = before\n"),
        ("schema='current_p1_node_composition_admission_watch_v2'",'schema=e.RETIRED_SCHEMA'),
        ("'All41 native calls and exact old-zero lineage'","'All41 native calls and exact frozen build/P1/probe lineage'"),
        ('result.update(child_execution_completed=True,whole_owned_tree_retired=True,\n'
         '                first22_output_bill_domain_byte_exact=True,first22_exact_checks=22)',
         'result.update(child_execution_completed=True,whole_owned_tree_retired=True,\n'
         '                additional_bindings_paid_once=True,same_existing_arena_qualified=True,dispatch_tariffs_qualified=True,\n'
         "                own_complete_bills=True,initialization_flops=record['initialization_flops'],\n"
         "                primary_dispatch_count=record['primary_dispatch_count'])"),
        ("        if child is not None:\n            try:\n                helper.stop_owned_group(child); child.wait(timeout=2)\n",
         "        try:\n            monitor.retire_pending(time.monotonic()+.30)\n"
         "        except BaseException:\n            result['observational_cleanup_error'] = traceback.format_exc()\n"
         "        if child is not None:\n            try:\n                helper.stop_owned_group(child); child.wait(timeout=2)\n"),
        ("                e.require(not helper.memory_snapshot(child.pid)['child_pids'], 'Owned descendants retired after failure')",
         "                e.require(not monitor.group_alive(child.pid), 'Recorded experiment group retired after failure')"),
        ("        for sig,handler in handlers.items(): signal.signal(sig,handler)\n",
         "        result['observational_groups_retired'] = not monitor.PENDING and 'observational_cleanup_error' not in result\n"
         "        if not result['observational_groups_retired']:\n"
         "            result.update(passed=False,completed=False,whole_owned_tree_retired=False)\n"
         "        for sig,handler in handlers.items(): signal.signal(sig,handler)\n"),
    ]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1, 'Unique fixed watchdog source span')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique watchdog inverse span')
        inverse = inverse.replace(after,before)
    require(inverse == original, 'Whole retired-process watchdog source inverse')
    return changed,dict(parent_sha256=PARENT_SHA,derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True,resource_limits_unchanged=True,
        next_completed_sample_absolute_deadline=True,progress_requires_closed_lifecycle=True)


def main():
    path = HERE/'native.py'
    spec = importlib.util.spec_from_file_location('partial_d6_native_endpoint',path)
    e = importlib.util.module_from_spec(spec); sys.modules[spec.name] = e; spec.loader.exec_module(e)
    source,proof = derived_source()
    # All three imports define only stdlib helpers. Their fixed hashes are
    # verified here; the full CLI owner inventory is verified again by main.
    monitor = e.load(e.MONITOR_DIR/'monitor.py','partial_d6_monitor_transport',e.MONITOR_SHA)
    inventory = e.load(e.P1/'watch.py','partial_d6_process_classifier',e.INVENTORY_SHA)
    namespace = dict(__name__='partial_d6_watch_source_owner',__file__=str(Path(__file__).resolve()),
                     e=e,monitor=monitor,inventory_owner=inventory,WATCH_PROOF=proof)
    exec(compile(source,str(__file__)+':derived','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
