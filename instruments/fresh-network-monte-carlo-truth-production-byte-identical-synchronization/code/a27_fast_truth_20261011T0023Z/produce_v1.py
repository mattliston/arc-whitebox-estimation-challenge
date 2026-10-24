#!/usr/bin/env python3
"""Versioned balanced-half reference production; default is source-only verify.

Uses the original A13 generator and MC implementation without editing them.
Every truth and completion receipt is published exclusively. Staging artifacts
are retained. The legacy producer must finish and exit before --run is allowed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    require(plan['schema'] == 'analyst_accelerated_truth_plan_v1', 'Wrong plan schema')
    require(plan['batch'] == 32768, 'Batch size changed')
    require([e['seed'] for e in plan['entries']] == list(range(7100136, 7100200)), 'Seed manifest changed')
    for p, digest in plan['source_sha256'].items():
        require(sha(p) == digest, 'Source changed: ' + p)
    require(str(Path(__file__).resolve()) in plan['source_sha256'], 'Producer not pinned')
    for e in plan['entries']:
        k = e['seed'] - 7100000
        expected_h = 978 if k <= 153 else 488 if k <= 167 else 244
        require(e['half_batches'] == expected_h and expected_h % 2 == 0, 'Unbalanced/wrong halves')
        require(e['N_half'] == expected_h * plan['batch'] and e['N'] == 2 * e['N_half'], 'Wrong sample count')
        expected_role = 'sizing' if k <= 151 else 'acceptance' if k <= 167 else 'final'
        require(e['role'] == expected_role, 'Wrong seed role')
        require(e['kind'] == ('existing_full' if k <= 153 else 'produce'), 'Wrong producer ownership')
        if e['kind'] == 'produce':
            require(Path(e['path']).parent == path.parent / 'truths', 'Truth outside versioned directory')
    return plan, sha(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def publish_json(path, value):
    path = Path(path)
    staging = HERE / 'staging' / (path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex)
    with staging.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(staging, path)  # Atomic no-replace publication; staging is retained.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(HERE / 'plan.json'))
    parser.add_argument('--run', action='store_true', help='Explicitly activate the reviewed plan')
    args = parser.parse_args()
    plan, plan_sha = load_plan(args.plan)
    if not args.run:
        print(json.dumps(dict(verified=True, plan_sha256=plan_sha, networks=len(plan['entries']),
                              production_networks=sum(e['kind'] == 'produce' for e in plan['entries']),
                              data_read=False, torch_imported=False)))
        return
    require(Path(plan['legacy_stop']).is_file(), 'Legacy clean STOP is missing')
    require(not any(alive(p) for p in plan['legacy_pids']), 'Legacy producer still alive; preserve its network')
    require(Path(plan['entries'][17]['path']).is_file(), 'Full seed153 is not finished')
    for directory in ('truths', 'receipts', 'staging'):
        (HERE / directory).mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(plan['core_path']).parent))
    spec = importlib.util.spec_from_file_location('a27_original_mc_truth', plan['core_path'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    np = core.np
    require(core.BATCH == plan['batch'], 'Environment changed core batch size')
    last_state = [0.0]

    def state(message, force=False):
        if force or time.time() - last_state[0] > 55:
            with (HERE / 'state.jsonl').open('a') as f:
                f.write(json.dumps(dict(utc=utc(), message=message)) + '\n')
            last_state[0] = time.time()
    core.state = state  # Only this new process; original STATE is never changed.

    def receipt(e):
        path = Path(e['path'])
        with np.load(path, allow_pickle=False) as z:
            require(int(z['seed']) == e['seed'] and int(z['N']) == e['N'], 'Truth identity/N mismatch')
            require(int(z['N_half']) == e['N_half'] and int(z['batch']) == plan['batch'], 'Truth half/batch mismatch')
            require(z['mc_post'].shape == (16, 1024), 'Wrong truth shape')
            require(all(np.isfinite(z[k]).all() for k in ('mc_post', 'mc_post_A', 'mc_post_B', 'mc_post_var')), 'Nonfinite truth')
            weights_sha = str(z['weights_sha256'])
        digest = sha(path)
        if e['sha256'] is not None:
            require(digest == e['sha256'], 'Existing full truth changed')
        return dict(schema='analyst_truth_completion_v1', plan_sha256=plan_sha,
                    seed=e['seed'], path=str(path), N=e['N'], N_half=e['N_half'],
                    half_batches=e['half_batches'], batch=plan['batch'], sha256=digest,
                    weights_sha256=weights_sha, source_sha256=plan['source_sha256'],
                    kind=e['kind'], utc=utc())

    state('started reviewed accelerated reference plan', True)
    for e in plan['entries']:
        if (HERE / 'STOP').exists():
            state('clean stop before seed%d' % e['seed'], True)
            return
        path, done = Path(e['path']), Path(e['completion_receipt'])
        if done.exists():
            saved = json.loads(done.read_text())
            current = receipt(e)
            for key in current:
                if key != 'utc':
                    require(saved[key] == current[key], 'Completed receipt changed: ' + key)
            continue
        if e['kind'] == 'existing_full':
            publish_json(done, receipt(e))
            continue
        require(not path.exists(), 'Unreceipted truth exists; preserve and review before recovery')
        core.HALF_BATCHES = e['half_batches']
        core.yield_gpu('before accelerated seed%d' % e['seed'])
        weights = core.loader.weights(e['seed'])
        weights_sha = core.loader.weights_sha256(weights)
        result, seconds = core.run_network(weights, 'seed%d' % e['seed'], e['seed'],
                                           'seed%d N%d' % (e['seed'], e['N']))
        require(int(result['N']) == e['N'] and int(result['N_half']) == e['N_half'], 'Core changed sample count')
        staging = HERE / 'staging' / (path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex)
        with staging.open('xb') as f:
            np.savez(f, **result, seed=np.int64(e['seed']), weights_sha256=np.array(weights_sha),
                     generation_plan_sha256=np.array(plan_sha),
                     core_source_sha256=np.array(plan['source_sha256'][plan['core_path']]))
            f.flush()
            os.fsync(f.fileno())
        os.link(staging, path)
        completed = receipt(e)
        publish_json(done, completed)
        event = dict(event='network', utc=utc(), seed=e['seed'], N=e['N'],
                     seconds=round(seconds, 3), path=str(path), sha256=completed['sha256'])
        with (HERE / 'prod.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
        del weights, result
    state('complete: acceptance154-167 and final168-199 references published', True)


if __name__ == '__main__':
    main()
