#!/usr/bin/env python3
"""Scheduling-only clean-boundary resume of the registered fixed48 screen.

No preparation, request edits, automatic retries, OOS, or partial scientific reads.
Root must approve the pinned old boundary and separately grant one or two slots.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from table_weightfit_v3 import HERE, O, prereg, smoke, screen

PROTOCOL = HERE / 'WEIGHTFIT_SCHEDULING_PROTOCOL_DRAFT_v5.json'
RUN = HERE / 'run_screen'
GUARD_PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
MODEL_PY = '/Users/matt/anaconda3/bin/python'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
OUTER_THREADS = THREAD_KEYS + ('VECLIB_NUM_THREADS',)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def resource_gate(new_slots=0):
    output = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', output).group(1))
    rows = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=10)
    heavy = sum(int(line.split(None, 1)[0]) >= 1_000_000 and 'python' in line.lower()
                for line in rows.splitlines() if line.strip())
    assert free >= 50 and heavy + 1 + new_slots <= 10, ('resource_gate', free, heavy, new_slots)
    return dict(free_pct=free, heavy_python_ge1GB=heavy, truth_reserve=1, new_slots=new_slots)


def source_pins(config):
    for path, digest in config['pins'].items():
        assert Path(path).name != 'key.txt'
        assert sha(path) == digest, ('source pin drift', path)


def configured(review_path):
    config, review = read(PROTOCOL), read(review_path)
    assert review_path.resolve().parent == HERE
    assert review['approved'] is True and review['reviewed_by'] == 'root'
    assert review['runner_sha256'] == sha(__file__) and review['protocol_sha256'] == sha(PROTOCOL)
    assert review['science_and_fixed48_unchanged'] is True
    assert review['old_clean_boundary_approved'] is True
    assert review['old_controller_reaped_and_owned_guards_retired'] is True
    assert review['boundary_snapshot_sha256'] == sha(config['boundary_snapshot'])
    assert config['canonical_run'] == str(RUN)
    assert config['segment_name'] == 'resume_v1'
    source_pins(config)
    prereg()  # Metadata only; preserves the exact registered scientific rule.
    assert smoke()['bitexact'], 'Previously qualified zero-C identity must remain exact'
    manifest = read(RUN / 'requests_manifest_v1.json')
    assert manifest['prereg_sha256'] == sha(HERE / 'PREREG_v2.json')
    assert len(manifest['requests']) == 48
    return config, manifest


def result_hashes(run):
    # Hash custody only. No result JSON values are parsed before clean all48.
    return {p.stem: sha(p) for p in sorted((run / 'results').glob('*.json'))}


def request_custody(run, requests, require_no_unfinished=True):
    found, queued, claimed = {}, set(), set()
    results = result_hashes(run)
    assert set(results) <= set(requests)
    assert not list((run / 'results').glob('*.error'))
    for subdir in ('queue', 'claimed'):
        for p in (run / subdir).glob('*.json'):
            rid = read(p)['id']
            assert rid in requests and rid not in found and sha(p) == requests[rid], str(p)
            found[rid] = p
            (queued if subdir == 'queue' else claimed).add(rid)
            if subdir == 'queue':
                assert p.name == rid + '.json' and rid not in results
            else:
                prefix = r'(?:000_sharedC_)?' if rid.startswith('wfit_z_') else ''
                assert re.fullmatch(prefix + re.escape(rid) + r'\.w\d+\.json', p.name)
    assert set(found) == set(requests)
    assert set(results) <= claimed
    if require_no_unfinished:
        assert claimed == set(results), 'Incomplete owned claim requires root review; no automatic retry'
    return dict(results=results, queued=sorted(queued), claimed=sorted(claimed))


def guard_clean(path, command, digest=None):
    if digest is not None:
        assert sha(path) == digest
    g = read(path)
    assert g['command'] == command
    assert g['completed'] is True and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows'] == []
    assert g['rss_backstop_bytes'] == g['footprint_trigger_bytes'] == 7_500_000_000
    assert 0 <= g['max_rss_bytes'] < 7_500_000_000 and 0 <= g['max_footprint_bytes'] < 7_500_000_000
    return g


def prior_boundary(config, manifest, require_initial=False):
    old = read(config['boundary_snapshot'])
    receipt = read(old['prior_launch_receipt']['path'])
    assert sha(old['prior_launch_receipt']['path']) == old['prior_launch_receipt']['sha256']
    assert receipt['phase'] == 'screen' and receipt['workers'] == 1
    assert receipt['completed'] is False and receipt['exit_codes'] == [0]
    assert receipt['errors'] == [] and receipt['result_count'] == 17
    assert len(receipt['runs']) == 1
    old_guard = old['prior_guard']
    assert receipt['runs'][0]['guard_record'] == old_guard['path']
    guard_clean(old_guard['path'], old_guard['command'], old_guard['sha256'])
    assert set(map(str, RUN.glob('guard_*.json'))) == {old_guard['path']}
    assert old['requests_manifest_sha256'] == sha(RUN / 'requests_manifest_v1.json')
    state = request_custody(RUN, manifest['requests'])
    assert len(old['preserved_results_sha256']) == 17 and len(old['remaining_request_ids']) == 31
    for rid, digest in old['preserved_results_sha256'].items():
        assert state['results'].get(rid) == digest, ('prior result changed', rid)
    assert set(old['preserved_results_sha256']) | set(old['remaining_request_ids']) == set(manifest['requests'])
    if require_initial:
        assert (RUN / 'STOP').exists() and not (RUN / 'LAUNCH_LOCK').exists()
        assert state['results'] == old['preserved_results_sha256']
        assert state['queued'] == old['remaining_request_ids']
        assert not [g for folder in RUN.glob('segments_v*') for g in folder.rglob('guard_*.json')]
    return old, state


def child_environment(run):
    e = dict(TRAINER_CARD='M', TRAINER_PL='1', TRAINER_F4C='1',
             TRAINER_THREADS='2', TRAINER_RELIEF='1', TRAINER_STAR_ATTACH='0',
             TRAINER_BIRTH_ATTACH='0', TRAINER_RSS_LIMIT='7500000000',
             TRAINER_GENERATED_MIN_N='15990784', TRAINER_GENERATED_ALLOW_FILE=str(run / 'allowed_seeds.txt'),
             PYTHONPATH=str(O / 'trainer'))
    e.update({key: '2' for key in THREAD_KEYS})
    return e


def cleanup_owned(processes):
    # Only exact Popen guards created by this invocation receive a signal.
    for process in processes:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
    timed_out = []
    deadline = time.monotonic() + 30
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out.append(process.pid)
    return timed_out


def retired_owned(processes, expected_guards):
    retired, hashes = True, {}
    for process, path in zip(processes, expected_guards):
        ok = False
        if process.poll() is not None and Path(path).exists():
            try:
                g = read(path)
                hashes[path] = sha(path)
                ok = g.get('whole_group_retired') is True and g.get('remaining_rows') == []
            except (OSError, ValueError, TypeError):
                pass  # Missing/invalid retirement proof retains both locks and reservation.
        retired = retired and ok
    return retired, hashes


def segment_home(config):
    return RUN / 'segments_v5' / config['segment_name']


def clean_final(config, manifest, review_path):
    prior_boundary(config, manifest)
    segment = segment_home(config)
    assert not (RUN / 'LAUNCH_LOCK').exists() and not (segment / 'SEGMENT_LOCK').exists()
    owner, ret = read(segment / 'OWNER_v1.json'), read(segment / 'RETIREMENT_v1.json')
    assert owner['runner_sha256'] == sha(__file__) and owner['protocol_sha256'] == sha(PROTOCOL)
    assert owner['review_sha256'] == sha(review_path) and owner['source_pins'] == config['pins']
    assert ret['owner_sha256'] == sha(segment / 'OWNER_v1.json')
    assert ret['completed'] is True and ret['exception'] is None and ret['errors'] == []
    assert ret['all_owned_groups_retired'] is True and ret['slot_reservation_retained'] is False
    assert ret['guard_exit_codes'] == [0] * owner['workers']
    assert len(owner['expected_guards']) == len(owner['child_commands']) == owner['workers']
    assert set(map(str, segment.glob('guard_w*_v1.json'))) == set(owner['expected_guards'])
    all_segment_guards = {str(g) for folder in RUN.glob('segments_v*') for g in folder.rglob('guard_*.json')}
    assert all_segment_guards == set(owner['expected_guards'])
    for i, (path, command) in enumerate(zip(owner['expected_guards'], owner['child_commands'])):
        started = read(segment / f'STARTED_w{i}_v1.json')
        assert started['owner_sha256'] == ret['owner_sha256'] and started['guard_record'] == path
        assert started['guard_pid'] == ret['guard_pids'][i]
        guard_clean(path, command, ret['guard_sha256'][path])
    state = request_custody(RUN, manifest['requests'])
    assert len(state['results']) == 48 and not state['queued']
    assert state['results'] == ret['result_sha256']
    source_pins(config)
    return ret


def launch(config, manifest, review_path, grant_path):
    old, _ = prior_boundary(config, manifest, require_initial=True)
    grant = read(grant_path)
    assert grant_path.resolve().parent == HERE
    assert grant['approved'] is True and grant['reviewed_by'] == 'root'
    assert grant['purpose'] == 'weighted_fit_fixed48_clean_boundary_resume'
    assert grant['protocol_sha256'] == sha(PROTOCOL) and grant['runner_sha256'] == sha(__file__)
    assert grant['review_sha256'] == sha(review_path)
    assert grant['boundary_snapshot_sha256'] == sha(config['boundary_snapshot'])
    workers = grant['numerical_slots']
    assert workers in (1, 2) and grant['segment_name'] == config['segment_name']
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0
    assert all(os.environ.get(key) == '1' for key in OUTER_THREADS)
    admission = resource_gate(workers)
    lock = RUN / 'LAUNCH_LOCK'
    lock.mkdir()  # Never remove a stale lock or repair old work automatically.
    segment = segment_home(config)
    segment.mkdir(parents=True, exist_ok=False)
    segment_lock = segment / 'SEGMENT_LOCK'
    segment_lock.mkdir()
    env = dict(os.environ)
    env.update({key: '1' for key in OUTER_THREADS})
    for key in ('TRAINER_TRUTH_PLAN', 'TRAINER_TRUTH_PLAN_SHA256'):
        env.pop(key, None)
    child_env = child_environment(RUN)
    worker = HERE / 'worker_weightfit_finite_v1.py'
    children = [['/usr/bin/env'] + [key + '=' + value for key, value in child_env.items()] +
                [MODEL_PY, '-B', str(worker), str(RUN), str(i)] for i in range(workers)]
    owner = dict(utc=stamp(), controller_pid=os.getpid(), runner_sha256=sha(__file__),
                 protocol_sha256=sha(PROTOCOL), review_sha256=sha(review_path), root_grant_sha256=sha(grant_path),
                 boundary_snapshot_sha256=sha(config['boundary_snapshot']), source_pins=config['pins'], workers=workers,
                 preserved_result_sha256=old['preserved_results_sha256'], admission=admission,
                 expected_guards=[str(segment / f'guard_w{i}_v1.json') for i in range(workers)],
                 child_commands=children, priority=dict(controller=0, guard=0, native_child=10),
                 outer_threads={key: '1' for key in OUTER_THREADS}, child_environment=child_env)
    write(segment / 'OWNER_v1.json', owner)
    processes, logs, failure = [], [], None
    ret = dict(utc=stamp(), completed=False, all_owned_groups_retired=False,
               owner_sha256=sha(segment / 'OWNER_v1.json'))

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller signal ' + str(signum))

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        (RUN / 'STOP').rename(segment / 'STOP.before_resume_v1')
        for i, child in enumerate(children):
            resource_gate(workers - i)
            log = (segment / f'worker_w{i}_v1.log').open('x')
            logs.append(log)
            command = [GUARD_PY, '-B', str(O / 'release/guard_footprint_v3.py'),
                       '--record', owner['expected_guards'][i], '--seconds', '10800',
                       '--rss-backstop', '7500000000', '--footprint-limit', '7500000000'] + child
            process = subprocess.Popen(command, cwd=O / 'trainer', env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            write(segment / f'STARTED_w{i}_v1.json', dict(guard_pid=process.pid,
                  owner_sha256=sha(segment / 'OWNER_v1.json'), guard_record=owner['expected_guards'][i]))
        next_check = time.monotonic() + 10
        while any(process.poll() is None for process in processes):
            assert not list((RUN / 'results').glob('*.error')), 'Worker error; stop for review'
            assert all(process.poll() in (None, 0) for process in processes), 'Guard failure; stop for review'
            if time.monotonic() >= next_check:
                ret['last_live_resource_gate'] = resource_gate(0)
                next_check = time.monotonic() + 10
            time.sleep(1)
        source_pins(config)
        state = request_custody(RUN, manifest['requests'])
        for rid, digest in old['preserved_results_sha256'].items():
            assert state['results'][rid] == digest
        for path, command in zip(owner['expected_guards'], children):
            guard_clean(path, command)
    except BaseException as error:
        failure = error
        (RUN / 'STOP').touch(exist_ok=True)
        ret['cleanup_wait_timed_out_guard_pids'] = cleanup_owned(processes)
    finally:
        (RUN / 'STOP').touch(exist_ok=True)
        retired, hashes = retired_owned(processes, owner['expected_guards'])
        ret.update(utc=stamp(), all_owned_groups_retired=retired,
                   guard_pids=[process.pid for process in processes],
                   guard_exit_codes=[process.poll() for process in processes], guard_sha256=hashes,
                   errors=[p.name for p in (RUN / 'results').glob('*.error')],
                   exception=repr(failure) if failure else None, result_sha256=result_hashes(RUN),
                   slot_reservation_retained=not retired)
        ret['completed'] = (failure is None and len(processes) == workers and retired and
                            ret['guard_exit_codes'] == [0] * workers and not ret['errors'] and
                            set(ret['result_sha256']) == set(manifest['requests']))
        ret['clean_boundary_hold'] = (failure is None and len(processes) == workers and retired and
                                     ret['guard_exit_codes'] == [0] * workers and not ret['errors'] and not ret['completed'])
        write(segment / 'RETIREMENT_v1.json', ret)
        for i, log in enumerate(logs):
            if i >= len(processes) or processes[i].poll() is not None:
                log.close()
        if retired:
            segment_lock.rmdir()
            lock.rmdir()
        signal.signal(signal.SIGTERM, previous_term)
    if failure is not None:
        raise failure
    if ret['completed']:
        clean_final(config, manifest, review_path)
    print(json.dumps(dict(completed=ret['completed'], clean_boundary_hold=ret['clean_boundary_hold'],
                         all_owned_groups_retired=retired, scientific_values_read=False)))
    return 0 if ret['completed'] else 2


def controller_retirement(config, path):
    assert path is not None and path.resolve().parent == HERE
    root = read(path)
    segment = segment_home(config)
    owner = read(segment / 'OWNER_v1.json')
    assert root['approved'] is True and root['reviewed_by'] == 'root'
    assert root['purpose'] == 'weighted_fit_controller_postrun_retirement'
    assert root['controller_pid'] == owner['controller_pid']
    assert root['controller_reaped'] is True and root['managed_exit_code'] == 0
    assert root['controller_absent'] is True and root['remaining_owned_rows'] == []
    assert root['owner_sha256'] == sha(segment / 'OWNER_v1.json')
    assert root['retirement_sha256'] == sha(segment / 'RETIREMENT_v1.json')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'launch', 'grade'))
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--retirement', type=Path)
    args = parser.parse_args()
    config, manifest = configured(args.review)
    if args.action == 'inspect':
        _, state = prior_boundary(config, manifest, require_initial=True)
        print(json.dumps(dict(completed_rows=len(state['results']), queued=len(state['queued']), values_read=False)))
        return 0
    if args.action == 'grade':
        clean_final(config, manifest, args.review)  # All48, exact requests, old+new clean guards, no live owners first.
        controller_retirement(config, args.retirement)
        print(json.dumps(screen(), indent=2, allow_nan=False))
        return 0
    assert args.grant, 'Separate root numerical slot grant required'
    return launch(config, manifest, args.review, args.grant)


if __name__ == '__main__':
    raise SystemExit(main())
