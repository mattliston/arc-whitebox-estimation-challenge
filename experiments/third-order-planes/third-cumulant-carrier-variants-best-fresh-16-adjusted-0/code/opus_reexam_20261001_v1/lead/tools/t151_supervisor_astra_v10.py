#!/usr/bin/env python3
"""SOURCE ONLY until root activation: resumable T151/T152 supervisor.

Default `audit` is read-only. `run` requires explicit --old-supervisor-retired.
`control SESSION --workers 0` requests a network-boundary hold; 1..3 resumes or
resizes. Control files and audit/decision receipts are exclusive-create files.
No signals are sent to workers. Existing worker_ph153 and scientific requests
are unchanged. Existing T153 empty decision is independently checked/adopted.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

O = Path(__file__).resolve().parents[2]
P, R = O / 'physicist', O / 'physicist/t151_carrier'
WORKER = P / 'worker_ph153.py'
SOURCE_SHA256 = {'physicist/worker_ph153.py': '6c087a94b4ef94e8acfc2bf34949eae7503f356eabedbaefe73b6090ca1aed0c', 'physicist/otr_cut_runtime_ph153.py': 'e49e561cf161bceea7237d126d76b3b93657c43087dd64e10f3e343d7313099b', 'lead/tools/t151_allowed_astra_v4.txt': '8fbdb2ffda646cdfef7864b355fc8893f95e592fea46121db59ddee49dcc6714', 'trainer/otr_common.py': '5e87f0b5f768de0e9bdaec63ac49b4af31c10f52cb6d01e00931d11ba83ae8a2'}
PY = '/Users/matt/anaconda3/bin/python'
FRESH, CONF = list(range(7100608, 7100624)), list(range(7100592, 7100608))
DIALS = {151: (('sp', 'sm'), ('dp', 'dm'), ('kp',)), 152: (('m',), ('x',)), 153: (('r3',),)}
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def stamp():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive(path, content):
    with path.open('x') as stream:
        stream.write(content)


def frozen(path, data):
    if path.exists():
        assert json.loads(path.read_text()) == data, 'Immutable receipt mismatch: ' + path.name
    else:
        exclusive(path, json.dumps(data, indent=2, allow_nan=False) + '\n')


def event(session, kind, **data):
    obj = dict(time=stamp(), event=kind, **data)
    name = time.time_ns()
    exclusive(session / 'events' / (str(name) + '.json'), json.dumps(obj, indent=2) + '\n')
    print(json.dumps(obj), flush=True)


def team(message):
    # date -u is obtained in this same process immediately before append.
    with (O / 'TEAM_LOG.md').open('a') as stream:
        stream.write('[astra] ' + stamp() + ' ' + message + '\n')


def request_map():
    found = {}
    for directory in ('queue', 'claimed'):
        for path in sorted((R / directory).glob('*.json')):
            try:
                data = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            rid = data['id']
            if rid in found:
                old_path, old_data = found[rid]
                assert old_data == data and not old_path.exists(), 'Duplicate request id: ' + rid
            found[rid] = (path, data)
    return found


def record(rid, reference=False):
    path = ((P / 't141_k4l_ext/results') if reference else (R / 'results')) / (rid + '.json')
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    assert data['id'] == rid and data['mode'] == 'eval' and data['threads'] == 2, rid
    assert len(data['mse_layers']) == 16 and all(math.isfinite(x) and x > 0 for x in data['mse_layers']), rid
    assert math.isfinite(data['mse']) and data['mse'] > 0, rid
    if 'final_prediction' in data:
        assert all(math.isfinite(x) for x in data['final_prediction']), rid
    return data


def ids(k, arm, seeds, confirmation=False):
    return [f'ph{k}{"x" if confirmation else ""}{arm}_r{seed}' for seed in seeds]


def mean_log(k, arm, seeds, confirmation=False):
    import numpy as np
    values = []
    for seed, rid in zip(seeds, ids(k, arm, seeds, confirmation)):
        candidate = record(rid)
        base = record(f'ph141_r{seed}_c', True) if confirmation else record(f'ph151_ref_r{seed}')
        if candidate is None or base is None:
            return None
        assert candidate['rank'] == base['rank'] == seed, rid
        values.append((candidate['mse_layers'][-1], base['mse_layers'][-1]))
    values = np.asarray(values)
    return float(np.log(values[:, 0] / values[:, 1]).mean())


POLICY_FOLDER = O / 'lead/runs/gain_floor_policy_20261013_v1'
ADDENDUM_SHA256 = 'd91844c5eb99fba8949233b364a29894b3ee2f1213756fb5a4e1e255bfbcbb5f'

def load_policy():
    assert sha(POLICY_FOLDER / 'PROSPECTIVE_ADDENDUM_v1.json') == ADDENDUM_SHA256
    spec = json.loads((POLICY_FOLDER / 'ACTIVATION_v1.json').read_text())
    assert spec['status'] == 'root_registered_prospectively'
    assert spec['addendum_sha256'] == ADDENDUM_SHA256
    assert set(spec['eligible']) <= {'T151', 'T152', 'metric', 'weighted'}
    for name, expected in spec['legacy_decision_sha256'].items():
        path = O / name
        assert (sha(path) if path.exists() else None) == expected, 'Legacy decision changed after registration: ' + name
    for item in spec['eligible']:
        assert spec['complete_at_registration'][item] is False, 'Completed experiments require independent continuation'
    for k in (151, 152):
        name = f'physicist/t{k}_confirm.txt'
        if f'T{k}' in spec['eligible']:
            assert name in spec['legacy_decision_sha256'] and spec['legacy_decision_sha256'][name] is None
    return spec

def prospective(k):
    return f'T{k}' in load_policy()['eligible']

def decision_path(k):
    return P / (f't{k}_confirm_astra_v10.txt' if prospective(k) else f't{k}_confirm.txt')

def passes_mean(k, mean):
    return mean < 0 if prospective(k) else mean <= -0.0015

def screen_rows_complete(k):
    required = [f'ph151_ref_r{seed}' for seed in FRESH]
    required += [rid for dial in DIALS[k] for arm in dial for rid in ids(k, arm, FRESH)]
    return all((R / 'results' / (rid + '.json')).exists() for rid in required)


def screen(k):
    if not screen_rows_complete(k):
        return dict(complete=False)
    means = {arm: mean_log(k, arm, FRESH) for dial in DIALS[k] for arm in dial}
    if any(x is None for x in means.values()):
        return dict(complete=False, means=means)
    winners = []
    for dial in DIALS[k]:
        winner = min(dial, key=lambda arm: means[arm])  # original stable tie order
        if passes_mean(k, means[winner]):
            winners.append(winner)
    return dict(complete=True, means=means, winners=winners)


def audit():
    load_policy()  # prospective registration is required before any statistics
    for name, expected in SOURCE_SHA256.items():
        assert sha(O / name) == expected, 'Pinned source changed: ' + name
    errors = sorted(p.name for p in (R / 'results').glob('*.error'))
    assert not errors, 'Worker .error present: ' + ','.join(errors)
    zero, base = record('ph153z_r7100584'), record('ph141_r7100584_c', True)
    assert zero is not None and base is not None, 'T153 no-op records incomplete'
    for key in ('mse', 'mse_layers', 'final_prediction'):
        assert key in zero and key in base and zero[key] == base[key], 'T153 no-op mismatch: ' + key
    state = {k: screen(k) for k in DIALS}
    assert state[153]['complete'] and state[153]['winners'] == [], 'Only audited rejected T153 may be resumed by v10'
    assert (P / 't153_confirm.txt').exists() and (P / 't153_confirm.txt').read_text().split() == [], 'T153 decision must already be empty'
    for k, value in state.items():
        path = decision_path(k)
        if path.exists():
            assert value['complete'] and path.read_text().split() == value['winners'], 'Existing decision disagrees with complete screen: ' + str(k)
            value['existing_decision_sha256'] = sha(path)
        if value['complete']:
            value['confirmation'] = {}
            for arm in value['winners']:
                complete = all((R / 'results' / (rid + '.json')).exists() for rid in ids(k, arm, CONF, True))
                mean = mean_log(k, arm, CONF, True) if complete else None
                value['confirmation'][arm] = dict(mean=mean, complete=mean is not None,
                    passed=(mean is not None and mean < 0 and passes_mean(k, (value['means'][arm] + mean) / 2)))
    return state


def publish_science(session, state):
    requests = request_map()
    for k in (151, 152, 153):
        value = state[k]
        if not value['complete']:
            continue
        receipt = session / f'screen{k}.json'
        first = not receipt.exists()
        fixed = dict(experiment=k, fresh=FRESH, means=value['means'], winners=value['winners'],
                     threshold=(0.0 if prospective(k) else -0.0015), strict=prospective(k),
                     rule='per-dial minimum, original tie order; all16 required; prospective negative rule only for registered unfinished experiments')
        frozen(receipt, fixed)
        decision = decision_path(k)
        if decision.exists():
            assert decision.read_text().split() == value['winners']
        else:
            exclusive(decision, ' '.join(value['winners']))
        for arm in value['winners']:
            source = requests[f'ph{k}{arm}_r{FRESH[0]}'][1]
            for seed, rid in zip(CONF, ids(k, arm, CONF, True)):
                req = copy.deepcopy(source)
                req.update(id=rid, rank=seed)
                if rid in requests:
                    assert requests[rid][1] == req, 'Confirmation request changed: ' + rid
                else:
                    assert record(rid) is None, 'Result without preserved request: ' + rid
                    # Atomic publish: a worker never sees a partially written JSON file.
                    tmp = R / 'queue' / ('.astra_v10_' + uuid.uuid4().hex)
                    exclusive(tmp, json.dumps(req))
                    try:
                        os.link(tmp, R / 'queue' / (rid + '.json'))
                    finally:
                        tmp.unlink()
        if first:
            event(session, 'screen_adopted_or_decided', **fixed)
            team(f'Hunter supervisor v10 audited complete experiment {k} fresh16; selected arms {value["winners"] or "none"}; registered prospective or legacy criterion and out-of-selection sign rule applied. Immutable receipt {receipt.relative_to(O)}.')
        confirmation = value.get('confirmation', {})
        if all(item['complete'] for item in confirmation.values()):
            path = session / f'complete{k}.json'
            first_done = not path.exists()
            final = dict(experiment=k, winners=value['winners'], confirmation=confirmation,
                         all32={arm: (value['means'][arm] + item['mean']) / 2 for arm, item in confirmation.items()})
            frozen(path, final)
            if first_done:
                event(session, 'experiment_complete', **final)
                team(f'Hunter supervisor v10 completed audited experiment {k}; confirmation and all32 receipt {path.relative_to(O)}. No partial result was used.')


def no_existing_workers():
    import psutil
    conflicts = []
    for process in psutil.process_iter(['pid', 'name']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            args = process.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if str(R) in args and any(Path(x).name in ('worker_ph153.py', 'worker_sc8.py') for x in args):
            conflicts.append(process.pid)
    assert not conflicts, 'Existing pool worker PIDs remain: ' + str(conflicts)


def admission():
    pressure = subprocess.check_output(['memory_pressure'], text=True)
    percent = int(re.findall(r'(\d+)%', pressure)[-1])
    listing = subprocess.check_output(['ps', '-axo', 'rss=,comm='], text=True)
    heavy = sum(1 for line in listing.splitlines() if re.match(r'\s*\d+\s+', line) and
                int(line.split()[0]) > 1048576 and 'python' in line.lower())
    vm = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', vm).group(1))
    free = sum(int(re.search(r'Pages ' + term + r':\s*(\d+)', vm).group(1)) for term in ('free', 'inactive')) * page
    # Reserve one slot for the GPU truth producer even when its RSS is small.
    return percent >= 55 and heavy + 1 + 1 <= 10 and free >= 8 * 1073741824


def worker_command(session, index, serial):
    env = ['-u', 'TRAINER_TRUTH_PLAN', '-u', 'TRAINER_TRUTH_PLAN_SHA256',
           'TRAINER_CARD=M', 'TRAINER_PL=1', 'TRAINER_F4C=1', 'TRAINER_THREADS=2',
           'TRAINER_RELIEF=1', 'TRAINER_STAR_ATTACH=0', 'TRAINER_BIRTH_ATTACH=0',
           'TRAINER_RSS_LIMIT=7500000000', 'TRAINER_GENERATED_MIN_N=15990784',
           'TRAINER_GENERATED_ALLOW_FILE=' + str(O / 'lead/tools/t151_allowed_astra_v4.txt'),
           'PYTHONPATH=' + str(O / 'trainer')]
    env += [key + '=2' for key in THREAD_KEYS]
    return ['/usr/bin/nice', '-n', '10', PY, '-B', str(O / 'release/guard_footprint_v3.py'), '--record',
            str(session / f'guard_w{index}_{serial}.json'), '--seconds', '14400',
            '--rss-backstop', '7500000000', '/usr/bin/env', *env,
            PY, '-B', str(WORKER), str(R), str(index)]


def run(args):
    assert args.old_supervisor_retired, 'Root must retire previous supervisor first'
    assert not (P / 'STOP_T151').exists(), 'Manual experiment stop remains'
    no_existing_workers()
    audit()  # no mutation before all existing decisions/control checks
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    for directory in ('events', 'control'):
        (session / directory).mkdir(exist_ok=True)
    assert not (session / 'source_pins_v9.json').exists(), 'Use a new v10 session; prior receipts remain immutable'
    frozen(session / 'source_pins_v10.json', SOURCE_SHA256)
    frozen(session / 'prospective_policy_v10.json', load_policy())
    lock = (R / '.astra_supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = R / 'STOP'
    if stop.exists():
        assert args.resume_stop, 'Existing STOP requires explicit --resume-stop'
        original = stop.read_bytes()
        frozen(session / f'adopted_stop_{time.time_ns()}.json', dict(sha256=sha(stop), bytes=original.hex()))
        assert stop.read_bytes() == original
        stop.unlink()
    desired, active = args.workers, {}
    serial = max([int(p.stem.rsplit('_', 1)[1]) for p in session.glob('worker_w*_*.log')] or [0])
    seen = {p.name for p in (session / 'control').glob('*.json')}
    own_stop = None
    draining, failed, finished, held_announced = False, None, False, False
    event(session, 'started', desired_workers=desired, worker_sha256=sha(WORKER), supervisor_sha256=sha(Path(__file__)))
    try:
        while True:
            for command in sorted((session / 'control').glob('*.json')):
                if command.name in seen:
                    continue
                spec = json.loads(command.read_text())
                assert isinstance(spec['workers'], int) and 0 <= spec['workers'] <= 3
                desired = spec['workers']; seen.add(command.name)
                event(session, 'control_received', desired_workers=desired, command=command.name, reason=spec.get('reason'))
                if len(active) != desired:
                    draining = True; held_announced = False
            if (P / 'STOP_T151').exists() or (stop.exists() and own_stop is None):
                desired = 0; draining = True
                failed = 'external hold; STOP preserved'  # graceful, never remove someone else's stop
            if not failed:
                try:
                    state = audit()
                    publish_science(session, state)
                    finished = all((session / f'complete{k}.json').exists() for k in DIALS)
                    if finished:
                        desired = 0; draining = True
                except Exception as error:
                    failed = repr(error); desired = 0; draining = True
                    event(session, 'audit_failure', detail=failed)
            for index, (process, stream) in list(active.items()):
                code = process.poll()
                if code is not None:
                    stream.close(); del active[index]
                    event(session, 'worker_guard_exit', index=index, code=code, draining=draining)
                    if not draining or code != 0:
                        failed = f'guard {index} exited {code}; no automatic error-as-success'
                        desired = 0; draining = True
            if draining:
                if not stop.exists():
                    own_stop = (str(session) + '\n').encode()
                    with stop.open('xb') as stream:
                        stream.write(own_stop)
                    event(session, 'boundary_drain_requested', remaining_workers=sorted(active), next_workers=desired)
                if not active:
                    no_existing_workers()
                    if not held_announced:
                        event(session, 'boundary_drained', next_workers=desired, failure=failed, finished=finished)
                        held_announced = True
                    if failed or finished:
                        return 0 if finished or failed == 'external hold; STOP preserved' else 2
                    if desired:
                        assert own_stop is not None and stop.read_bytes() == own_stop, 'STOP owner changed'
                        stop.unlink(); own_stop = None; draining = False; held_announced = False
                    else:
                        time.sleep(2)
                        continue
            if not draining and len(active) < desired:
                index = next(i for i in range(desired) if i not in active)
                # No abandoned higher-index claim may silently starve after a resize.
                pending = [p for p in (R / 'claimed').glob('*.json') if
                           not (R / 'results' / (json.loads(p.read_text())['id'] + '.json')).exists()]
                assert all(int(re.search(r'\.w(\d+)\.json$', p.name).group(1)) < desired for p in pending), 'Pending orphan needs a higher worker count before resizing'
                if admission():
                    serial += 1
                    stream = (session / f'worker_w{index}_{serial}.log').open('x')
                    env = os.environ.copy(); env.update({key: '1' for key in THREAD_KEYS})
                    process = subprocess.Popen(worker_command(session, index, serial), cwd=O / 'trainer', env=env,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[index] = process, stream
                    event(session, 'worker_guard_started', index=index, pid=process.pid)
                    time.sleep(20)
            time.sleep(2)
    finally:
        # An interrupted supervisor stops only this queue at a network boundary.
        # It never kills a process. Guards retain their original resource backstop.
        if active and not stop.exists():
            exclusive(stop, str(session) + '\n')
            event(session, 'supervisor_exit_boundary_hold', remaining_workers=sorted(active))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('audit')
    execute = sub.add_parser('run')
    execute.add_argument('--session', type=Path, required=True)
    execute.add_argument('--workers', type=int, choices=range(1, 4), required=True)
    execute.add_argument('--old-supervisor-retired', action='store_true')
    execute.add_argument('--resume-stop', action='store_true')
    control = sub.add_parser('control')
    control.add_argument('session', type=Path)
    control.add_argument('--workers', type=int, choices=range(4), required=True)
    control.add_argument('--reason', required=True)
    args = parser.parse_args()
    if args.command in (None, 'audit'):
        state = audit()
        for k, value in list(state.items()):
            if not value['complete']:
                state[k] = dict(complete=False, expected_screen_pairs=16*sum(map(len,DIALS[k])))
        print(json.dumps(dict(source_pins=SOURCE_SHA256, experiments=state), indent=2, allow_nan=False)); return 0
    if args.command == 'control':
        assert (args.session / 'control').is_dir()
        path = args.session / 'control' / f'{time.time_ns()}.json'
        exclusive(path, json.dumps(dict(time=stamp(), workers=args.workers, reason=args.reason)) + '\n')
        print(path); return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
