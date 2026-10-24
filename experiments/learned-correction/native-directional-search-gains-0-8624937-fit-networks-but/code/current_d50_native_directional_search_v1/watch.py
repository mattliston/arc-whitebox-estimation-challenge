"""Independent stdlib-only, one-child, one-deadline ND supervisor.

Eight fixed serial stages; a completed scientific STOP ends the sequence.
Only the freshly created child process group may be terminated. No retries.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'current_bd_p_coadaptation_v1/watch.py'
HELPER_SHA = 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b'
LIMIT, RSS, SWAP = 1200., 12_000_000_000, 2 * 1024**3
STEPS = (
    ('qualify', 'cpu.py', 'qualify', 'qualification_v1'),
    ('baseline', 'native.py', 'native', 'native_baseline_v1'),
    ('propose', 'cpu.py', 'propose', 'proposal_v1'),
    ('plus', 'native.py', 'native', 'native_plus_v1'),
    ('minus', 'native.py', 'native', 'native_minus_v1'),
    ('interpolate', 'interpolate.py', 'interpolate', 'interpolation_v1'),
    ('candidate', 'native.py', 'native', 'native_candidate_v1'),
    ('judge', 'judge.py', 'judge', 'judgment_v1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--admission-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    spec = importlib.util.spec_from_file_location('nd_common_watch', HERE / 'common.py')
    c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c
    spec.loader.exec_module(c)
    c.require(all(not (HERE / x[3]).exists() for x in STEPS), 'Exclusive one-shot sequence; no retry/continuation')
    s = c.Stage(__file__, args.admission_sha256, 'watch', 'watch_v1')
    child, samples, helper = None, None, None
    before = None
    last = last_swap = s.start
    seals = {}
    s.record.update(terminated=False, stages=[], scientific_stop=False,
        whole_sequence_wall_limit_s=LIMIT, combined_live_rss_limit_bytes=RSS,
        swap_growth_limit_bytes=SWAP, requested_sample_period_s=.25,
        maximum_sampled_combined_rss_bytes=0, maximum_sampled_child_tree_rss_bytes=0,
        maximum_sampled_supervisor_rss_bytes=0, maximum_sample_interval_s=0.,
        maximum_swap_growth_bytes=0., samples=0, one_numerical_child_at_a_time=True,
        memory_covers_post_context_serialization=True,
        sampled_rss_is_not_an_absolute_peak_guarantee=True, native_score_measured=False)

    def observe(name, force_swap=False, log_sample=True):
        nonlocal last, last_swap
        now = time.monotonic()
        table = helper.process_table()
        own = [rss * 1024 for pid, parent, group, rss in table if pid == os.getpid()]
        c.require(len(own) == 1, 'Observable supervisor RSS')
        child_rss, pids = (0, []) if child is None else helper.tree_memory(child.pid, table)
        if child is not None and not pids:
            c.require(child.poll() is not None, 'Live child tree must be observed')
        combined, interval = own[0] + child_rss, now - last
        last = now
        for key, value in (('maximum_sample_interval_s', interval),
                ('maximum_sampled_combined_rss_bytes', combined),
                ('maximum_sampled_child_tree_rss_bytes', child_rss),
                ('maximum_sampled_supervisor_rss_bytes', own[0])):
            s.record[key] = max(s.record[key], value)
        s.record['samples'] += 1
        item = dict(elapsed_s=now - s.start, stage=name, combined_rss_bytes=combined,
            supervisor_rss_bytes=own[0], child_tree_rss_bytes=child_rss, pids=pids, interval_s=interval)
        if force_swap or now - last_swap >= 5:
            state = helper.swap_state()
            growth = state['used_bytes'] - before['used_bytes']
            s.record['maximum_swap_growth_bytes'] = max(s.record['maximum_swap_growth_bytes'], growth)
            s.record['latest_swap'] = state
            item.update(swap=state, swap_growth_bytes=growth)
            last_swap = now
        if log_sample:
            samples.write(json.dumps(item, allow_nan=False) + '\n'); samples.flush()
        else:
            s.record['final_resource_sample_after_hashing'] = item
        if combined >= RSS or s.record['maximum_swap_growth_bytes'] > SWAP or time.monotonic() - s.start >= LIMIT:
            s.record.update(terminated=True, reason='Whole-sequence deadline, combined RSS, or swap-growth guard')
            raise RuntimeError(s.record['reason'])

    def command(name, source, stage):
        value = [str(c.OFFPY if stage == 'native' else c.AP), '-u', str(HERE / source),
                 '--run', '--admission-sha256', args.admission_sha256]
        if source == 'cpu.py':
            value += ['--stage', stage]
        elif stage == 'native':
            value += ['--case', name]
        needed = []
        if name != 'qualify':
            needed.append(('qualification', 'qualify'))
        if name not in ('qualify', 'baseline'):
            needed.append(('baseline', 'baseline'))
        if name not in ('qualify', 'baseline', 'propose'):
            needed.append(('proposal', 'propose'))
        if name in ('interpolate', 'judge'):
            needed += [('plus', 'plus'), ('minus', 'minus')]
        if name in ('candidate', 'judge'):
            needed.append(('interpolation', 'interpolate'))
        if name == 'judge':
            needed.append(('candidate', 'candidate'))
        for flag, prior in needed:
            value += ['--' + flag + '-result-sha256', seals[prior]['result_sha256']]
        return value

    try:
        s.environment('AP')
        s.pin(HELPER, HELPER_SHA, 'source_bindings')
        helper = c.module(HELPER, 'nd_old_stdlib_watch_helpers')
        before = helper.swap_state()
        s.record['before_swap'] = before
        samples = (s.directory / 'resources.jsonl').open('x')
        for name, source, stage, folder in STEPS:
            s.recheck(); observe(name + '_before', True)
            cmd = command(name, source, stage)
            entry = dict(stage=stage, name=name, command=cmd, completed=False,
                         started_elapsed_s=time.monotonic() - s.start)
            s.record['stages'].append(entry)
            print(json.dumps(dict(stage=name, action='start', elapsed_s=entry['started_elapsed_s'])), flush=True)
            with (s.directory / (name + '.log')).open('x') as log:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                entry['child_pid'] = entry['child_process_group'] = child.pid
                while child.poll() is None:
                    observe(name)
                    time.sleep(max(0., min(.25, LIMIT - (time.monotonic() - s.start))))
                code = entry['exit_code'] = child.wait(timeout=3)
                entry['elapsed_s'] = time.monotonic() - s.start - entry['started_elapsed_s']
                observe(name + '_after', True)
                _, survivors = helper.tree_memory(child.pid, helper.process_table())
                c.require(not survivors, 'No owned numerical descendants may survive')
                rp, cp = HERE / folder / 'result.json', HERE / folder / 'completion.json'
                digest = s.pin(rp)
                completion_digest = s.pin(cp)
                result, seal = c.read(rp), c.read(cp)
                c.require(result.get('schema') == c.SCHEMA and result.get('stage') == stage
                    and result.get('admission_sha256') == args.admission_sha256
                    and result.get('completed') is True and result.get('hashes_unchanged') is True
                    and not result.get('error') and seal.get('completed') is True
                    and seal.get('passed') == result.get('passed') and seal.get('result_sha256') == digest,
                    'Complete sealed execution required before any advance')
                if stage == 'native':
                    c.require(result.get('case') == name, 'Exact native case')
                c.require(code == (0 if result['passed'] else 1), 'Exit status matches scientific verdict')
                c.require(result['passed'] or (name in ('interpolate', 'judge') and result['scientific']),
                          'Only completed interpolation/judgment may scientifically stop')
                c.require(result['validation_targets_accessed'] is False and result['MPS_used'] is False,
                          'No VALID labels or MPS at any stage')
                c.require(name == 'judge' or result['check_targets_accessed'] is False,
                          'Check targets withheld until final judgment')
                s.inventory(result, targets=True)
                receipt = dict(result_path=str(rp), result_sha256=digest,
                               completion_path=str(cp), completion_sha256=completion_digest)
                seals[name] = receipt
                entry.update(receipt, completed=True, passed=result['passed'])
                child = None
                s.recheck(); observe(name + '_sealed', True)
                print(json.dumps(dict(stage=name, action='complete', passed=result['passed'],
                                      elapsed_s=time.monotonic() - s.start)), flush=True)
                if not result['passed']:
                    s.record.update(scientific_stop=True, stop_stage=name, decision=result['decision'])
                    break
        s.record.update(stage_receipts=seals, numerical_gates_passed=True,
                        advancement_gate_passed=bool(result.get('advancement_gate_passed', False)),
                        native_score_measured='interpolate' in seals)
        s.finish(passed=bool(result.get('advancement_gate_passed', False)), scientific=True)
        observe('sequence_final', True)
    except BaseException:
        s.record.update(passed=False, completed=False, error=traceback.format_exc())
        if child is not None and helper is not None:
            try:
                helper.stop_owned_group(child)
            except BaseException:
                s.record['termination_error'] = traceback.format_exc()
    finally:
        if helper is not None and before is not None:
            try:
                after = helper.swap_state()
                s.record['after_swap'] = after
                s.record['maximum_swap_growth_bytes'] = max(s.record['maximum_swap_growth_bytes'],
                    after['used_bytes'] - before['used_bytes'])
                if s.record['maximum_swap_growth_bytes'] > SWAP or time.monotonic() - s.start >= LIMIT:
                    s.record.update(passed=False, completed=False, terminated=True, reason='Final resource guard')
            except BaseException:
                s.record.update(passed=False, completed=False, final_resource_error=traceback.format_exc())
        if samples is not None:
            samples.close()
            s.pin(s.directory / 'resources.jsonl', group='artifacts')
        for name, _, _, _ in STEPS:
            path = s.directory / (name + '.log')
            if path.exists():
                s.pin(path, group='artifacts')
        final_guard = (lambda: observe('after_final_hashing', True, False)) if helper is not None and before is not None else None
        status = s.close(final_guard=final_guard)
    print(json.dumps({k: s.record.get(k) for k in ('passed', 'completed', 'elapsed_s', 'decision',
        'maximum_sampled_combined_rss_bytes', 'maximum_swap_growth_bytes', 'terminated', 'error')},
        sort_keys=True), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
