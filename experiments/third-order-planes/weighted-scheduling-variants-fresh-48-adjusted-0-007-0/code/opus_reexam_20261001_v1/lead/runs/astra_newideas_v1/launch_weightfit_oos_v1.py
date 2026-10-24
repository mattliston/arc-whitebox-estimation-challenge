#!/usr/bin/env python3
"""HELD guarded confirmation16, only after the original fixed48 is complete.
Source-only readiness: no automatic selection, preparation, launch, or grade.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
HP = D / 'launch_weightfit_v5.py'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(HP) == 'f05815ac980a2ea073923fbe7d77a73c81743fa48bfaddcdb55a08358073894a'
spec = importlib.util.spec_from_file_location('weighted_fixed48_owner_v5', HP)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
O = H.O
RUN = D / 'run_confirm'
PROTOCOL = D / 'WEIGHTFIT_OOS_PROTOCOL_DRAFT_v1.json'
read, write = H.read, H.write


def configured(review_path):
    cfg, review = read(PROTOCOL), read(review_path)
    assert review_path.resolve().parent == D
    assert review['approved'] is True and review['reviewed_by'] == 'root'
    assert review['runner_sha256'] == sha(__file__) and review['protocol_sha256'] == sha(PROTOCOL)
    assert review['registered_confirmation_science_unchanged'] is True
    assert cfg['canonical_run'] == str(RUN) and cfg['segment_name'] == 'oos_v1'
    H.source_pins(cfg)
    return cfg


def complete_fresh(fresh_review, fresh_retirement):
    # No table/arm values until all48 requests/results, old+new guards and
    # the managed fixed48 controller's actual retirement have been proved.
    cfg, manifest = H.configured(fresh_review)
    H.clean_final(cfg, manifest, fresh_review)
    H.controller_retirement(cfg, fresh_retirement)
    decision = H.screen()
    assert decision['complete'] is True and decision['winner'] in ('q', 'h')
    return decision


def granted(cfg, review_path, grant_path, purpose, fresh_review, fresh_retirement):
    g = read(grant_path)
    assert grant_path.resolve().parent == D and g['approved'] is True and g['reviewed_by'] == 'root'
    assert g['purpose'] == purpose and g['runner_sha256'] == sha(__file__)
    assert g['protocol_sha256'] == sha(PROTOCOL) and g['review_sha256'] == sha(review_path)
    assert g['fresh_review_sha256'] == sha(fresh_review) and g['fresh_retirement_sha256'] == sha(fresh_retirement)
    decision = complete_fresh(fresh_review, fresh_retirement)
    assert g['winner'] == decision['winner']
    return g, decision


def prepared(cfg, winner):
    p = RUN / 'OOS_PREPARATION_v1.json'
    prep = read(p)
    assert prep['winner'] == winner and prep['protocol_sha256'] == sha(PROTOCOL)
    assert prep['runner_sha256'] == sha(__file__)
    for path, digest in prep['output_sha256'].items():
        assert sha(path) == digest
    manifest = read(RUN / 'requests_manifest_v1.json')
    assert manifest['prereg_sha256'] == sha(D / 'PREREG_v2.json')
    expected = {f'wfit_{winner}_r{seed}' for seed in range(7100592, 7100608)}
    assert set(manifest['requests']) == expected
    return manifest, prep


def prepare(cfg, review, grant, fresh_review, fresh_retirement):
    g, decision = granted(cfg, review, grant, 'weighted_fit_oos16_prepare', fresh_review, fresh_retirement)
    assert g['numerical_slots'] == 0
    assert (RUN / 'STOP').exists() and (RUN / 'HELD_BY_ROOT').exists()
    assert not (RUN / 'LAUNCH_LOCK').exists() and not (RUN / 'OOS_PREPARATION_v1.json').exists()
    assert all(not list((RUN / d).iterdir()) for d in ('queue', 'claimed', 'results'))
    assert not (RUN / 'requests_manifest_v1.json').exists()
    # Preserve the exact original request construction and original truth-N checks.
    path = D / 'prepare_weightfit_v3.py'
    spec = importlib.util.spec_from_file_location('weighted_registered_prepare_v3', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main('confirm')
    outputs = [RUN / 'requests_manifest_v1.json', RUN / 'truth_manifest_v1.json',
               RUN / 'allowed_seeds.txt', D / 'screen_decision_negative_v1.json']
    write(RUN / 'OOS_PREPARATION_v1.json', dict(utc=H.stamp(), winner=decision['winner'],
          protocol_sha256=sha(PROTOCOL), runner_sha256=sha(__file__), review_sha256=sha(review),
          prepare_grant_sha256=sha(grant), fresh_review_sha256=sha(fresh_review),
          fresh_retirement_sha256=sha(fresh_retirement), numerical_evaluations_launched=0,
          output_sha256={str(p): sha(p) for p in outputs}))
    manifest, _ = prepared(cfg, decision['winner'])
    state = H.request_custody(RUN, manifest['requests'])
    assert len(state['queued']) == 16 and not state['results']


def segment_home(cfg):
    return RUN / 'segments_v1' / cfg['segment_name']


def clean_final(cfg, manifest, review_path, fresh_review, fresh_retirement):
    decision = complete_fresh(fresh_review, fresh_retirement)
    pinned, prep = prepared(cfg, decision['winner'])
    assert pinned == manifest
    segment = segment_home(cfg)
    assert not (RUN / 'LAUNCH_LOCK').exists() and not (segment / 'SEGMENT_LOCK').exists()
    owner, ret = read(segment / 'OWNER_v1.json'), read(segment / 'RETIREMENT_v1.json')
    assert owner['runner_sha256'] == sha(__file__) and owner['protocol_sha256'] == sha(PROTOCOL)
    assert owner['review_sha256'] == sha(review_path) and owner['source_pins'] == cfg['pins']
    assert owner['preparation_sha256'] == sha(RUN / 'OOS_PREPARATION_v1.json')
    assert owner['winner'] == decision['winner']
    assert ret['owner_sha256'] == sha(segment / 'OWNER_v1.json')
    assert ret['completed'] is True and ret['exception'] is None and ret['errors'] == []
    assert ret['all_owned_groups_retired'] is True and ret['slot_reservation_retained'] is False
    assert ret['guard_exit_codes'] == [0] * owner['workers']
    assert len(owner['expected_guards']) == len(owner['child_commands']) == owner['workers']
    all_guards = {str(p) for p in RUN.rglob('guard_*.json')}
    assert all_guards == set(owner['expected_guards'])
    for i, (path, command) in enumerate(zip(owner['expected_guards'], owner['child_commands'])):
        started = read(segment / f'STARTED_w{i}_v1.json')
        assert started['owner_sha256'] == ret['owner_sha256'] and started['guard_record'] == path
        assert started['guard_pid'] == ret['guard_pids'][i]
        H.guard_clean(path, command, ret['guard_sha256'][path])
    state = H.request_custody(RUN, manifest['requests'])
    assert len(state['results']) == 16 and not state['queued'] and state['results'] == ret['result_sha256']
    H.source_pins(cfg)
    return ret


def launch(cfg, review, grant, fresh_review, fresh_retirement):
    g, decision = granted(cfg, review, grant, 'weighted_fit_oos16_numerical', fresh_review, fresh_retirement)
    workers = g['numerical_slots']
    assert workers in (1, 2) and g['segment_name'] == cfg['segment_name']
    manifest, prep = prepared(cfg, decision['winner'])
    assert g['preparation_sha256'] == sha(RUN / 'OOS_PREPARATION_v1.json')
    state = H.request_custody(RUN, manifest['requests'])
    assert len(state['queued']) == 16 and not state['results'] and not state['claimed']
    assert (RUN / 'STOP').exists() and not list(RUN.rglob('guard_*.json'))
    assert os.getpriority(os.PRIO_PROCESS, 0) == 0
    assert all(os.environ.get(key) == '1' for key in H.OUTER_THREADS)
    admission = H.resource_gate(workers)
    lock = RUN / 'LAUNCH_LOCK'
    lock.mkdir()
    segment = segment_home(cfg)
    segment.mkdir(parents=True, exist_ok=False)
    segment_lock = segment / 'SEGMENT_LOCK'
    segment_lock.mkdir()
    env = dict(os.environ)
    env.update({key: '1' for key in H.OUTER_THREADS})
    for key in ('TRAINER_TRUTH_PLAN', 'TRAINER_TRUTH_PLAN_SHA256'):
        env.pop(key, None)
    child_env = H.child_environment(RUN)
    children = [['/usr/bin/env'] + [key + '=' + value for key, value in child_env.items()] +
                [H.MODEL_PY, '-B', str(D / 'worker_weightfit_finite_v1.py'), str(RUN), str(i)] for i in range(workers)]
    owner = dict(utc=H.stamp(), controller_pid=os.getpid(), runner_sha256=sha(__file__),
                 protocol_sha256=sha(PROTOCOL), review_sha256=sha(review), root_grant_sha256=sha(grant),
                 preparation_sha256=sha(RUN / 'OOS_PREPARATION_v1.json'), winner=decision['winner'],
                 fresh_review_sha256=sha(fresh_review), fresh_retirement_sha256=sha(fresh_retirement),
                 source_pins=cfg['pins'], workers=workers, admission=admission,
                 expected_guards=[str(segment / f'guard_w{i}_v1.json') for i in range(workers)],
                 child_commands=children, priority=dict(controller=0, guard=0, native_child=10),
                 outer_threads={key: '1' for key in H.OUTER_THREADS}, child_environment=child_env)
    write(segment / 'OWNER_v1.json', owner)
    processes, logs, failure = [], [], None
    ret = dict(completed=False, all_owned_groups_retired=False, owner_sha256=sha(segment / 'OWNER_v1.json'))

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller signal ' + str(signum))

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        (RUN / 'STOP').rename(segment / 'STOP.before_launch_v1')
        for i, child in enumerate(children):
            H.resource_gate(workers - i)
            log = (segment / f'worker_w{i}_v1.log').open('x')
            logs.append(log)
            command = [H.GUARD_PY, '-B', str(O / 'release/guard_footprint_v3.py'),
                       '--record', owner['expected_guards'][i], '--seconds', '5400',
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
                ret['last_live_resource_gate'] = H.resource_gate(0)
                next_check = time.monotonic() + 10
            time.sleep(1)
        H.source_pins(cfg)
        H.request_custody(RUN, manifest['requests'])
        for path, command in zip(owner['expected_guards'], children):
            H.guard_clean(path, command)
    except BaseException as error:
        failure = error
        (RUN / 'STOP').touch(exist_ok=True)
        ret['cleanup_wait_timed_out_guard_pids'] = H.cleanup_owned(processes)
    finally:
        (RUN / 'STOP').touch(exist_ok=True)
        retired, hashes = H.retired_owned(processes, owner['expected_guards'])
        ret.update(utc=H.stamp(), all_owned_groups_retired=retired,
                   guard_pids=[process.pid for process in processes], guard_exit_codes=[process.poll() for process in processes],
                   guard_sha256=hashes, errors=[p.name for p in (RUN / 'results').glob('*.error')],
                   exception=repr(failure) if failure else None, result_sha256=H.result_hashes(RUN),
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
        clean_final(cfg, manifest, review, fresh_review, fresh_retirement)
    print(json.dumps(dict(completed=ret['completed'], all_owned_groups_retired=retired,
                         confirmation_values_read=False, auto_grade=False)))
    return 0 if ret['completed'] else 2


def root_retirement(cfg, path):
    assert path is not None and path.resolve().parent == D
    root = read(path)
    segment = segment_home(cfg)
    owner = read(segment / 'OWNER_v1.json')
    assert root['approved'] is True and root['reviewed_by'] == 'root'
    assert root['purpose'] == 'weighted_fit_oos_controller_postrun_retirement'
    assert root['controller_pid'] == owner['controller_pid']
    assert root['controller_reaped'] is True and root['managed_exit_code'] == 0
    assert root['controller_absent'] is True and root['remaining_owned_rows'] == []
    assert root['owner_sha256'] == sha(segment / 'OWNER_v1.json')
    assert root['retirement_sha256'] == sha(segment / 'RETIREMENT_v1.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch', 'grade'))
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--grant', type=Path)
    p.add_argument('--fresh-review', type=Path, required=True)
    p.add_argument('--fresh-retirement', type=Path, required=True)
    p.add_argument('--retirement', type=Path)
    args = p.parse_args()
    cfg = configured(args.review)
    if args.action == 'grade':
        decision = complete_fresh(args.fresh_review, args.fresh_retirement)
        manifest, _ = prepared(cfg, decision['winner'])
        clean_final(cfg, manifest, args.review, args.fresh_review, args.fresh_retirement)
        root_retirement(cfg, args.retirement)
        from table_weightfit_v3 import confirm
        print(json.dumps(confirm(), indent=2, allow_nan=False))
        return 0
    assert args.grant
    return globals()[args.action](cfg, args.review, args.grant, args.fresh_review, args.fresh_retirement)


if __name__ == '__main__':
    raise SystemExit(main())
