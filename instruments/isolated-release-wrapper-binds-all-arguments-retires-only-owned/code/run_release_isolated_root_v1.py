"""Root-only isolated invocation of the unchanged release script, once.

The final manifest is sealed only after the exact candidate runtime verdict.
This program never submits an archive and never edits research controllers.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT_PID = 77740
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    ap.add_argument('--manifest-sha', required=True)
    args = ap.parse_args()
    manifest = HERE/'RELEASE_ISOLATED_PLAN_v1.json'
    assert sha(manifest) == args.manifest_sha
    plan = json.loads(manifest.read_text())
    assert plan['one_unchanged_release_authorized'] and not plan['automatic_fire']
    assert plan['command'][0] == 'bash' and len(plan['command']) == 6
    assert plan['command'][2] == plan['candidate']
    assert plan['command'][4] == plan['qualification_output']
    assert sha(__file__) == plan['runner_sha256']

    def verify():
        for path, digest in plan['source_pins'].items():
            assert sha(path) == digest, ('changed pin', path)

    verify()
    verdict = json.loads(Path(plan['runtime_verdict']).read_text())
    assert verdict['successful'] and verdict['release_source_sha256'] == plan['candidate_sha256']
    assert verdict['material_measured_margin_accepted']
    hp = Path(plan['isolation_helper'])
    assert sha(hp) == plan['isolation_helper_sha256']
    spec = importlib.util.spec_from_file_location('frozen_release_isolation_helpers', hp)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    record = HERE/'ISOLATION_v1.json'
    log = HERE/'release_min.log'
    assert not record.exists() and not log.exists()
    assert not Path(plan['qualification_output']).exists()
    assert not Path(plan['archive_directory']).exists()
    initial = h.processes()
    assert ROOT_PID in initial and 'run_queue_astra_v6.sh' in initial[ROOT_PID]['command']
    h.deny_external_heavy(initial, h.descendants(initial, ROOT_PID))
    if args.check:
        print(json.dumps(dict(utc=utc(), status='READ_ONLY_RELEASE_READY_NO_EXECUTION',
              manifest_sha256=sha(manifest), runner_sha256=sha(__file__),
              candidate_sha256=sha(plan['candidate']), command=plan['command'], no_fire=True)))
        return 0

    rec = dict(start_utc=utc(), status='ISOLATING', manifest_sha256=sha(manifest),
               runner_sha256=sha(__file__), source_pins=plan['source_pins'],
               command=plan['command'], no_fire=True)
    paused, active, exitcode = {}, None, 1
    began = time.monotonic()

    def save():
        record.write_text(json.dumps(rec, indent=2)+'\n')

    def interrupt(signum, frame):
        raise KeyboardInterrupt('signal %s' % signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def pause(pid, before):
        paused[pid] = before
        os.kill(pid, signal.SIGSTOP)

    try:
        pause(ROOT_PID, initial[ROOT_PID])
        table = h.processes()
        judges = [pid for pid, row in table.items() if row['ppid'] == ROOT_PID
                  and 'run_current32_' in row['command'] and '_fable_v1.py ' in row['command']]
        for pid in judges:
            pause(pid, table[pid])
        rec['paused'] = paused
        save()
        print(utc(), 'DRAINING', len(judges), flush=True)
        deadline = time.monotonic()+210
        while True:
            table = h.processes()
            tree = h.descendants(table, ROOT_PID)
            h.deny_external_heavy(table, tree)
            remaining = {pid for pid in tree-set(paused) if not table[pid]['state'].startswith('Z')}
            if not remaining:
                break
            assert time.monotonic() < deadline, remaining
            time.sleep(2)
        rec.update(drained_utc=utc(), free_pct_before=h.pressure())
        save()
        assert rec['free_pct_before'] >= 60
        verify()
        table = h.processes()
        h.deny_external_heavy(table, h.descendants(table, ROOT_PID))
        assert not Path(plan['qualification_output']).exists()
        assert not Path(plan['archive_directory']).exists()
        with log.open('x') as stream:
            active = subprocess.Popen(plan['command'], cwd=h.O, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            rec.update(release_pid=active.pid, release_start_utc=utc())
            save()
            print(rec['release_start_utc'], 'UNCHANGED_RELEASE_MIN_START', active.pid, flush=True)
            try:
                rc = active.wait(timeout=900)
            finally:
                if active.poll() is None:
                    os.killpg(active.pid, signal.SIGINT)
                    active.wait(timeout=30)
                active = None
        rec.update(release_end_utc=utc(), release_returncode=rc, log_sha256=sha(log))
        save()
        assert rc == 0, ('release_min failed; no retry', rc, str(log))
        verify()
        rec['status'] = 'UNCHANGED_RELEASE_SCRIPT_COMPLETED_ROOT_ARTIFACT_REVIEW_REQUIRED'
        exitcode = 0
    except BaseException as error:
        rec.update(status='FAILED', error=repr(error))
        raise
    finally:
        cleanup, resumed, already_stopped = [], [], []
        for pid, before in reversed(list(paused.items())):
            try:
                now = h.processes().get(pid)
                if now and now['started'] == before['started'] and now['command'] == before['command']:
                    if before['state'].startswith('T'):
                        already_stopped.append(pid)
                    else:
                        os.kill(pid, signal.SIGCONT)
                        resumed.append(pid)
            except ProcessLookupError:
                pass
            except BaseException as error:
                cleanup.append(dict(pid=pid, error=repr(error)))
        rec.update(end_utc=utc(), pause_elapsed_s=time.monotonic()-began, resumed=resumed,
                   previously_stopped_preserved=already_stopped, cleanup_errors=cleanup, exitcode=exitcode)
        try:
            rec['free_pct_after'] = h.pressure()
        except Exception as error:
            rec['free_pct_after_error'] = repr(error)
        try:
            save()
        finally:
            print(rec['end_utc'], 'RESEARCH_RESTORED', resumed, 'prior_stops', already_stopped,
                  'errors', cleanup, 'exitcode', exitcode, flush=True)
    return exitcode


if __name__ == '__main__':
    raise SystemExit(main())
