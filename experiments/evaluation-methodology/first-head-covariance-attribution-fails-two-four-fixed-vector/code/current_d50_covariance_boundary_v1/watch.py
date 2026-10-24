"""Inert, independent supervisor for the one fixed covariance attribution.

Four serial processes share one 600-second deadline and one combined live-RSS
limit. Only a newly created child process group can be terminated. Scientific
FAIL in the final judge is a completed diagnostic, never permission to retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
AP = Path('/Users/matt/anaconda3/bin/python')
OFFPY = P2 / 'data_gate/.venv/bin/python'
HELPER = NEW / 'current_bd_p_coadaptation_v1/watch.py'
HELPER_SHA = 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b'
MODEL = NEW / 'current_d50_native_g0_forward_v1/model.py'
MODEL_SHA = '00f7a37825577ef083b8a5eac1f230183bcbae7f9b1f2a68739e4f37d52b7049'
PROTOCOL_SHA = '26b60da71938e72025c2d2a2ca9634e65aa8bd4f0e3e8827333d67b356848179'
SCHEMA = 'current_d50_covariance_boundary_v1'
LIMIT, RSS, SWAP = 600., 4_000_000_000, 2 * 1024**3
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    for name in ('watch-source', 'native-source', 'cpu-source', 'judge-source', 'model', 'protocol'):
        p.add_argument('--' + name + '-sha256')
    args = p.parse_args()
    if not args.run:
        p.print_help(); return 0
    require(all(isinstance(v, str) and len(v) == 64 for k, v in vars(args).items() if k != 'run'),
            'All reviewed SHA256 arguments required')
    require(args.model_sha256 == MODEL_SHA and args.protocol_sha256 == PROTOCOL_SHA,
            'Fixed original model and prospective protocol')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 supervisor and children')
    require(all(os.environ.get(k) == '1' for k in BLAS), 'Explicit one-thread environment')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Stdlib-only supervisor')
    pins = {str(Path(__file__).resolve()): args.watch_source_sha256,
            str(HERE / 'native.py'): args.native_source_sha256,
            str(HERE / 'cpu.py'): args.cpu_source_sha256,
            str(HERE / 'judge.py'): args.judge_source_sha256,
            str(HERE / 'PROTOCOL.md'): args.protocol_sha256,
            str(MODEL): args.model_sha256, str(HELPER): HELPER_SHA}

    def recheck():
        for path, digest in pins.items():
            require(sha(path) == digest, 'Reviewed source changed: ' + path)

    recheck()
    spec = importlib.util.spec_from_file_location('cb_reviewed_watch_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    out = HERE / 'watch_v1'
    folders = ('native_n_v1', 'cpu_v1', 'native_controls_v1', 'judgment_v1')
    require(not out.exists() and all(not (HERE / x).exists() for x in folders),
            'Exclusive one-shot sequence; no continuation or retry')
    out.mkdir()
    start = last = last_swap = time.monotonic()
    before = helper.swap_state()
    result = dict(schema=SCHEMA + '_watch', passed=False, completed=False,
        sources_unchanged=False, terminated=False, stages=[], source_bindings=pins,
        started_utc=datetime.now(timezone.utc).isoformat(), before_swap=before,
        whole_sequence_wall_limit_s=LIMIT, combined_live_rss_limit_bytes=RSS,
        swap_growth_limit_bytes=SWAP, requested_sample_period_s=.25,
        maximum_sampled_combined_rss_bytes=0, maximum_sampled_child_tree_rss_bytes=0,
        maximum_sampled_supervisor_rss_bytes=0, maximum_sample_interval_s=0.,
        maximum_swap_growth_bytes=0., samples=0, MPS_used=False,
        one_numerical_child_at_a_time=True, memory_covers_post_context_serialization=True,
        sampled_rss_is_not_an_absolute_peak_guarantee=True)
    helper.write(out / 'started.json', result)
    samples = (out / 'resources.jsonl').open('x')
    child = None

    def observe(stage, force_swap=False):
        nonlocal last, last_swap
        now = time.monotonic()
        rows = helper.process_table()
        own = [rss * 1024 for pid, parent, group, rss in rows if pid == os.getpid()]
        require(len(own) == 1, 'Supervisor RSS must be observed')
        child_rss, pids = (0, []) if child is None else helper.tree_memory(child.pid, rows)
        if child is not None and not pids:
            require(child.poll() is not None, 'Cannot observe the live child tree')
        combined = own[0] + child_rss
        interval = now - last
        last = now
        result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'], interval)
        result['maximum_sampled_combined_rss_bytes'] = max(result['maximum_sampled_combined_rss_bytes'], combined)
        result['maximum_sampled_child_tree_rss_bytes'] = max(result['maximum_sampled_child_tree_rss_bytes'], child_rss)
        result['maximum_sampled_supervisor_rss_bytes'] = max(result['maximum_sampled_supervisor_rss_bytes'], own[0])
        result['samples'] += 1
        item = dict(elapsed_s=now - start, stage=stage, combined_rss_bytes=combined,
                    supervisor_rss_bytes=own[0], child_tree_rss_bytes=child_rss, pids=pids, interval_s=interval)
        if force_swap or now - last_swap >= 5:
            state = helper.swap_state()
            growth = state['used_bytes'] - before['used_bytes']
            result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
            result['latest_swap'] = state
            item.update(swap=state, swap_growth_bytes=growth)
            last_swap = now
        samples.write(json.dumps(item, allow_nan=False) + '\n'); samples.flush()
        if combined >= RSS or result['maximum_swap_growth_bytes'] > SWAP or time.monotonic() - start >= LIMIT:
            result.update(terminated=True, reason='Whole-sequence deadline, combined RSS, or swap-growth guard')
            raise RuntimeError(result['reason'])

    def verify_record(folder, stage, exit_code):
        rp, cp = HERE / folder / 'result.json', HERE / folder / 'completion.json'
        record, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
        digest = sha(rp)
        require(record['schema'] == SCHEMA and record['stage'] == stage
            and record['completed'] and record['hashes_unchanged']
            and record['truth_accessed'] is False and not record.get('error')
            and seal['completed'] and seal['passed'] == record['passed']
            and seal['result_sha256'] == digest, 'Complete hash-sealed diagnostic stage')
        if stage == 'judge':
            require(record['numerical_gates_passed']
                and record['passed'] == record['information_gate_passed']
                and exit_code == (0 if record['passed'] else 1), 'Scientific verdict separate from execution failure')
        else:
            require(exit_code == 0 and record['passed'], 'Prior stage must pass before advance')
        source = HERE / ('cpu.py' if stage == 'cpu' else 'judge.py' if stage == 'judge' else 'native.py')
        require(record['source_bindings'][str(source)] == pins[str(source)]
            and record['source_bindings'][str(HERE / 'PROTOCOL.md')] == PROTOCOL_SHA,
            'Stage reviewed source/protocol identity')
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, expected in record[group].items():
                require(sha(path) == expected, 'Post-stage binding changed: ' + path)
        observe(stage + '_sealed', True)
        return record, dict(result_path=str(rp), result_sha256=digest,
                           completion_path=str(cp), completion_sha256=sha(cp))

    records = {}
    try:
        for stage, folder in zip(('n', 'cpu', 'controls', 'judge'), folders):
            recheck(); observe(stage + '_before', True)
            command = [str(OFFPY if stage in ('n', 'controls') else AP), '-u',
                str(HERE / ('native.py' if stage in ('n', 'controls') else stage + '.py')),
                '--run', '--source-sha256', pins[str(HERE / ('native.py' if stage in ('n', 'controls') else stage + '.py'))],
                '--protocol-sha256', PROTOCOL_SHA]
            if stage in ('n', 'controls'):
                command += ['--stage', stage]
            if stage in ('cpu', 'judge'):
                command += ['--native-source-sha256', args.native_source_sha256]
            if stage == 'cpu':
                command += ['--model-sha256', MODEL_SHA]
            if stage != 'n':
                command += ['--native-result-sha256', records['n']['result_sha256']]
            if stage in ('controls', 'judge'):
                command += ['--cpu-source-sha256', args.cpu_source_sha256,
                            '--cpu-result-sha256', records['cpu']['result_sha256']]
            if stage == 'judge':
                command += ['--controls-result-sha256', records['controls']['result_sha256']]
            entry = dict(stage=stage, command=command, completed=False,
                         started_elapsed_s=time.monotonic() - start)
            result['stages'].append(entry)
            print(json.dumps(dict(stage=stage, action='start', elapsed_s=entry['started_elapsed_s'])), flush=True)
            with (out / (stage + '.log')).open('x') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                entry['child_pid'] = entry['child_process_group'] = child.pid
                while child.poll() is None:
                    observe(stage)
                    time.sleep(max(0., min(.25, LIMIT - (time.monotonic() - start))))
                entry['exit_code'] = child.wait(timeout=3)
                entry['elapsed_s'] = time.monotonic() - start - entry['started_elapsed_s']
                observe(stage + '_after', True)
                _, remaining = helper.tree_memory(child.pid, helper.process_table())
                require(not remaining, 'Child left an owned process alive')
                record, receipt = verify_record(folder, stage, entry['exit_code'])
                records[stage] = receipt
                entry.update(receipt, completed=True, passed=record['passed'])
                recheck()
                print(json.dumps(dict(stage=stage, action='complete', passed=record['passed'],
                                      elapsed_s=time.monotonic() - start)), flush=True)
                child = None
        result.update(completed=True, passed=record['passed'],
                      information_gate_passed=record['information_gate_passed'])
        recheck(); observe('sequence_final', True)
        result['sources_unchanged'] = True
    except BaseException:
        result.update(completed=False, passed=False, error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child)
            except BaseException:
                result['termination_error'] = traceback.format_exc()
    finally:
        try:
            result['after_swap'] = helper.swap_state()
            result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],
                result['after_swap']['used_bytes'] - before['used_bytes'])
            if result['maximum_swap_growth_bytes'] > SWAP or time.monotonic() - start >= LIMIT:
                result.update(completed=False, passed=False, terminated=True, reason='Final resource guard')
            recheck()
            result['sources_unchanged'] = True
        except BaseException:
            result.update(completed=False, passed=False, final_check_error=traceback.format_exc())
        samples.close()
        result.update(elapsed_s=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        helper.write(out / 'result.json', result)
        helper.write(out / 'completion.json', dict(completed=result['completed'], passed=result['passed'],
                                                   result_sha256=sha(out / 'result.json')))
    print(json.dumps({k: result.get(k) for k in ('passed', 'completed', 'elapsed_s',
        'maximum_sampled_combined_rss_bytes', 'maximum_swap_growth_bytes', 'terminated', 'reason', 'error')},
        sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
