"""P2 supervisor: absolute-deadline observations, fixed fresh numerical restart.

All original graph/source/resource/cleanup checks remain. No model imports.
"""
from pathlib import Path
import common as c

_version_changes = [
    ('current_p1_native_objective_p2_v1',c.RESTART_SCHEMA,1),
    ('current_p1_native_objective_p2_watch_v1',c.RESTART_WATCH_SCHEMA,1),
    ('current_p1_native_objective_p2_roster_v1',c.RESTART_ROSTER_SCHEMA,1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Frozen entrypoint suppressed; this adapter owns the sole explicit entrypoint.\n',1),
]
_frozen_watch, RESTART_WATCH_VERSION_PROOF = c.splice_source(c.ORIGINAL/'watch.py',c.ORIGINAL_WATCH_SHA,_version_changes)
exec(compile(_frozen_watch,str(Path(__file__).resolve())+':frozen_p2_watch','exec'),globals())
require(sha(HERE/'monitor.py') == c.MONITOR_SHA,'Reviewed monitor before loading its stdlib functions')
monitor = c.module(HERE/'monitor.py','p2_absolute_deadline_monitor')


def _splice_text(source, edits):
    changed = source
    for old,new in edits:
        require(changed.count(old) == 1,'Unique monitor/receipt-only substitution')
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1,'Unique monitor/receipt inverse')
        inverse = inverse.replace(new,old)
    require(inverse == source,'Every non-monitor inherited statement restored')
    return changed,dict(parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inverse_source_byte_exact=True,substitution_count=len(edits),
        numerical_worker_edited=False,resource_thresholds_changed=False)


def _nested(source, name):
    nodes = [n for n in c.ast.walk(c.ast.parse(source)) if isinstance(n,c.ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1,'One inherited nested observer')
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _bind_restart(sources, inputs, path, digest, source, tick):
    path = str(Path(path).resolve()); chosen = sources if source else inputs
    actual = sha(path,tick) if path not in sources and path not in inputs else sources.get(path,inputs.get(path))
    require(digest is None or actual == digest,'Exact restart/mock provenance owner')
    require(path not in chosen or chosen[path] == actual,'Consistent restart hash binding')
    chosen[path] = actual
    return actual


def _check_restart_child(value, stage, args):
    require(value['restart_common_inverse'] == c.RESTART_COMMON_PROOF
            and value['failed_training_watch_sha256'] == c.FAILED_WATCH_SHA
            and value['old_operational_failure_still_failed'] and value['fresh_full_restart']
            and not value['partial_payload_or_optimizer_resumed']
            and value['prefix199_authenticated_before_targets'] and value['prefix199_required'] == 199
            and value['prefix199_exact_matches'] == (199 if stage == 'train' else 0)
            and value['prefix199_byte_exact'] is (stage == 'train')
            and value['monitor_certificate']['result_sha256'] == args.monitor_test_result_sha256
            and value['restart_worker_inverse']['parent_source_sha256'] == c.ORIGINAL_WORKER_SHA
            and value['restart_worker_inverse']['inverse_source_byte_exact'], 'Fixed fresh restart, exact prefix and actual monitor prerequisite')


_observe = '''    def observe(stream=None,active=True):
        nonlocal last_sample
        start = time.monotonic()
        previous = last_sample if active else start
        deadline = min(previous+MAX_INTERVAL,began+limit,last_progress+PROGRESS_LIMIT)
        snapshot = monitor.collect(deadline)
        rows = snapshot['rows']
        scoped = monitor.scope(rows,os.getpid(),None if child is None else child.pid)
        foreign = other_workers(rows,scoped.pop('allowed'))
        swap = snapshot['swap']; growth = swap['used_bytes']-before['used_bytes']
        progress()
        observed = time.monotonic()
        item = dict(elapsed_s=observed-began,pre_log_interval_s=observed-previous,
            interval_in_stream_is_pre_log_only=True,**scoped,swap=swap,swap_growth_bytes=growth,
            completed_graph_count=result['completed_graph_count'],no_completed_graph_s=observed-last_progress,
            other_detected_workers=foreign,live_child_interval_gate=active,
            acquisition_started_elapsed_s=snapshot['started_monotonic_s']-began,
            acquisition_finished_elapsed_s=snapshot['acquired_monotonic_s']-began,
            absolute_sample_deadline_elapsed_s=deadline-began,
            command_records=snapshot['command_records'],commands_reaped=snapshot['commands_reaped'],
            reserved_completion_margin_s=snapshot['reserved_completion_margin_s'],
            one_shared_process_table=True,retry_count=0,cached_sample_used=False)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],scoped['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        # Preserve the attempted observation before any later resource gate.
        if stream is not None:
            stream.write(json.dumps(item,allow_nan=False)+'\\n'); stream.flush()
        require(snapshot['commands_reaped'] and not monitor.PENDING,'Every observational command retired')
        require(not foreign,'Another identifiable numerical worker is active; stop only our owned child')
        require(scoped['combined_rss_bytes'] < HOST_LIMIT and growth <= SWAP_LIMIT,'Unchanged independent RSS/swap limits')
        time_guard()
        # Logging/progress/resource checks are INSIDE the absolute sample bound.
        completed,interval = monitor.complete(previous,deadline)
        last_sample = completed
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
            result['completed_sample_intervals_s'].append(interval)
        item.update(completed_elapsed_s=completed-began,actual_completed_interval_s=interval)
        result['last_completed_sample_elapsed_s'] = completed-began
        return item
'''

_before_admission = ('    bind(sources,time_guard); bind(inputs,time_guard)\n'
                     '    lineage = admit_parent(sources,inputs,time_guard)')
_after_admission = ('    sources.update(c.restart_source_bindings())\n'+_before_admission+'\n'
    '    def restart_pin(path,digest,source=False):\n'
    '        return _bind_restart(sources,inputs,path,digest,source,time_guard)\n'
    '    restart_prefix = c.prefix_provenance(restart_pin)\n'
    '    monitor_certificate = c.mock_admission(args.monitor_test_result_sha256,restart_pin)')
_monitor_edits = [
    ("    parser.add_argument('--mps-slot-clear',action='store_true')",
     "    parser.add_argument('--mps-slot-clear',action='store_true')\n"
     "    parser.add_argument('--monitor-test-result-sha256')"),
    ("    required = ('source','worker','common','protocol') + (",
     "    required = ('source','worker','common','protocol','monitor_test_result') + ("),
    (_before_admission,_after_admission),
    ("        '--protocol-sha256',PROTOCOL_SHA,'--watcher-sha256',args.source_sha256]",
     "        '--protocol-sha256',PROTOCOL_SHA,'--watcher-sha256',args.source_sha256,\n"
     "        '--monitor-test-result-sha256',args.monitor_test_result_sha256]"),
    ('        parent_lineage=lineage,watch_adapter=WATCH_ADAPTER_PROOF,\n',
     '        parent_lineage=lineage,watch_adapter=WATCH_ADAPTER_PROOF,\n'
     '        restart_watch_version_inverse=RESTART_WATCH_VERSION_PROOF,monitor_source_inverse=MONITOR_SOURCE_PROOF,\n'
     '        monitor_certificate=monitor_certificate,failed_training_watch_sha256=c.FAILED_WATCH_SHA,\n'
     '        old_operational_failure_still_failed=True,fresh_full_restart=True,prefix199_exact_matches=0,\n'
     '        prefix199_authenticated_before_targets=True,completed_sample_intervals_s=[],\n'
     '        actual_sample_interval_includes_logging=True,observational_commands_retired=False,\n'),
    ('            marker(value,call_roster[\'entries\'][index])\n',
     '            marker(value,call_roster[\'entries\'][index])\n'
     "            if args.stage == 'train' and index < 199:\n"
     "                require(c.signature(value) == restart_prefix['prefix_signatures'][index], 'Independent exact old199 numerical signature')\n"
     "                require(result['prefix199_exact_matches'] == index,'Monotone independent prefix replay')\n"
     "                result['prefix199_exact_matches'] += 1\n"),
    (_nested(_derived,'observe'),_observe),
    ("        before = helper.swap_state(); result['before_swap'] = before\n"
     "        result['prelaunch_process_inventory'] = inventory()",
     "        prelaunch = monitor.collect(min(time.monotonic()+MAX_INTERVAL,began+limit,last_progress+PROGRESS_LIMIT))\n"
     "        before = prelaunch['swap']; result['before_swap'] = before\n"
     "        result['prelaunch_process_inventory'] = prelaunch['rows']"),
    ('            observe(resources,active=False)\n            last_sample = time.monotonic()\n',
     '            observe(resources,active=False)\n'),
    ("        record,end = completed_child(child_out,result['child_result_sha256'],args.stage,args,time_guard)",
     "        record,end = completed_child(child_out,result['child_result_sha256'],args.stage,args,time_guard)\n"
     '        _check_restart_child(record,args.stage,args)\n'
     "        require(result['prefix199_exact_matches'] == (199 if args.stage == 'train' else 0), 'All required independent prefix checks')"),
    ('    except BaseException:\n        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n',
     '    except BaseException:\n        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n'
     '        try:\n            monitor.retire_pending(time.monotonic()+.30)\n'
     "        except BaseException:\n            result['observational_cleanup_error'] = traceback.format_exc()\n"),
    ('        for sig,handler in handlers.items():\n',
     '        try:\n'
     "            require(not monitor.PENDING and not result.get('observational_cleanup_error'),'All owned observational commands retired')\n"
     "            result['observational_commands_retired'] = True\n"
     '        except BaseException:\n'
     "            result.update(passed=False,completed=False,whole_owned_tree_retired=False,observational_cleanup_error=traceback.format_exc())\n"
     '        for sig,handler in handlers.items():\n'),
]
_repaired_source,MONITOR_SOURCE_PROOF = _splice_text(_derived,_monitor_edits)
exec(compile(_repaired_source,str(Path(__file__).resolve())+':deadline_monitor_adapter','exec'),globals())


if __name__ == '__main__':
    raise SystemExit(main())
