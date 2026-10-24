"""Root-only, once-only invocation of unchanged release_min_v1.sh. Never fires.

The final v2 manifest must be sealed after the complete runtime verdict. Ownership
tracking and cleanup concern this release invocation only. They do not alter the
candidate, an evaluation guard, or any research controller. An uncertain process
retirement leaves research paused with a durable failure record for root review.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
O = HERE.parents[4]
PROJECT = O.parents[4]
ROOT_PID = 77740
RELEASE_MIN = O/'lead/runs/fable_oct19_l13rank_v1/release_min_v1.sh'
CANDIDATE = HERE.parent/'candidate_first128_paid_nocensus_release_v1.py'
CANDIDATE_SHA = 'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'
LAST_ARCHIVE = O/'release/runs/astra_edge8_hidden_complement90_4685e145_v1/submission.tar.gz'
ISOLATION_HELPER = O/'lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/qualify_runtime_companion_isolated_v3.py'
FIXED_PINS = {
    str(RELEASE_MIN): 'ec26493b98f4cae46e2f20d2745a2af96be0def430500770678e315c0f566258',
    str(O/'lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py'): '6523c5520595531d3ce25af6e73f6f894f7f17bd0967c9d44f1f7973e204241c',
    str(O/'lead/runs/astra_oct18_accuracy_v1/official_setup_five_v2.py'): '479a0551a8e8b586573553bd999149c959ab8bd39902172a77e1f8e8f8b558cc',
    str(O/'release/stage_a_package.py'): 'efc40e1c451392e45ec79ce3abadb3c138408a921ad8b643ca4430ffbe49dd8e',
    str(O/'release/common.py'): 'f3410b655c1833e1a8d8e437f791850394af599a1a5c998d91d358a3cdcc5b15',
    str(O/'fire/prefire_check_v1.py'): '02f63acb927da7208aafc019f38830d857efbb4aa499310695fd004ed7f0b701',
    str(LAST_ARCHIVE): '3848cfaed85175d0f0f817e99234201cc5736bad51e071eabc553b719145947a',
    str(CANDIDATE): CANDIDATE_SHA,
    str(ISOLATION_HELPER): 'cc6f457db1297877a4b2c09b0d95489d5cd40d26e749ca5151b098d4f8911b84',
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def identity(pid, row):
    return (pid, row['started'], row['command'])


def process_table():
    """No environment arguments; retain PGID as well as the existing identity fields."""
    table = {}
    raw = subprocess.check_output(
        ['/bin/ps', '-axo', 'pid=,ppid=,pgid=,lstart=,stat=,command='], text=True, timeout=5)
    for line in raw.splitlines():
        fields = line.split(None, 9)
        if len(fields) == 10:
            table[int(fields[0])] = dict(ppid=int(fields[1]), pgid=int(fields[2]),
                started=' '.join(fields[3:8]), state=fields[8], command=fields[9])
    assert table and os.getpid() in table, 'Incomplete process inventory'
    return table


class OwnedRelease:
    """Track observed ancestry and continuously anchored new process groups.

    Signals target individually rechecked (PID, start, command) identities only.
    A group number by itself never authorizes a signal. A new, unowned project
    process or a member left in an observed group makes final cleanup uncertain.
    Dependencies are injectable for root-reviewed metadata-only fixtures.
    """
    def __init__(self, baseline, owner_pid, release_pid, table_fn=process_table,
                 signal_fn=os.kill, now_fn=time.monotonic, sleep_fn=time.sleep):
        self.baseline = {identity(pid, row) for pid, row in baseline.items()}
        self.baseline_groups = {row['pgid'] for row in baseline.values()}
        self.owner_pid, self.release_pid = owner_pid, release_pid
        self.table_fn, self.signal_fn = table_fn, signal_fn
        self.now_fn, self.sleep_fn = now_fn, sleep_fn
        self.owned, self.groups, self.signals = {}, set(), []
        self.uncertainty, self.root_seen = [], False

    def observe(self, table, root_running):
        root = table.get(self.release_pid)
        if root_running and root and root['ppid'] == self.owner_pid and root['pgid'] == self.release_pid:
            self.owned.setdefault(identity(self.release_pid, root), dict(pid=self.release_pid, **root, via='Popen root'))
            self.root_seen = True
        while True:
            live_ids = {identity(pid, row) for pid, row in table.items()} & self.owned.keys()
            parents = {item[0] for item in live_ids}
            anchored = {table[pid]['pgid'] for pid in parents} - self.baseline_groups
            added = 0
            for pid, row in table.items():
                key = identity(pid, row)
                if key in self.baseline or key in self.owned or pid == self.owner_pid:
                    continue
                if row['ppid'] in parents or row['pgid'] in anchored:
                    via = 'observed parent' if row['ppid'] in parents else 'live owned group anchor'
                    self.owned[key] = dict(pid=pid, **row, via=via)
                    added += 1
            if not added:
                break
        self.groups.update(row['pgid'] for key, row in self.owned.items()
                           if key in {identity(pid, r) for pid, r in table.items()}
                           and row['pgid'] not in self.baseline_groups)
        return table

    def sample(self, root_running):
        try:
            return self.observe(self.table_fn(), root_running)
        except BaseException as error:
            self.uncertainty.append('process inventory failed: '+repr(error))
            raise

    def live(self, table):
        return {pid: row for pid, row in table.items()
                if identity(pid, row) in self.owned and not row['state'].startswith('Z')}

    def unknown(self, table):
        result = []
        for pid, row in table.items():
            key = identity(pid, row)
            if pid == self.owner_pid or key in self.baseline or key in self.owned or row['state'].startswith('Z'):
                continue
            if str(PROJECT) in row['command'] or row['pgid'] in self.groups:
                result.append(dict(pid=pid, **row))
        return result

    def checked_signal(self, pid, row, sig, root_running):
        current = self.sample(root_running).get(pid)
        key = identity(pid, row)
        if current and identity(pid, current) == key and key in self.owned and not current['state'].startswith('Z'):
            try:
                self.signal_fn(pid, sig)
                self.signals.append(dict(pid=pid, started=row['started'], command=row['command'], signal=int(sig)))
            except ProcessLookupError:
                pass

    def retire(self, root_running_fn):
        """Bounded cleanup, then two clean inventories. Never touches unknown jobs."""
        table = self.sample(root_running_fn())
        initially_live = list(self.live(table))
        if not self.root_seen:
            self.uncertainty.append('release root exited before its identity was observed')
        for sig, grace in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            if not self.live(table):
                break
            # Freeze the owned tree before terminating parents, and discover any
            # child born since the previous snapshot. This is cleanup only.
            freeze_end = self.now_fn()+3
            while True:
                live = self.live(table)
                moving = {pid: row for pid, row in live.items() if not row['state'].startswith('T')}
                if not moving:
                    break
                for pid, row in moving.items():
                    self.checked_signal(pid, row, signal.SIGSTOP, root_running_fn())
                table = self.sample(root_running_fn())
                if self.now_fn() >= freeze_end:
                    self.uncertainty.append('owned process tree did not become stopped during cleanup')
                    break
            live = self.live(table)
            # Children first; every signal rechecks exact identity. No killpg.
            def depth(pid):
                seen = set()
                while pid in live and pid not in seen:
                    seen.add(pid)
                    pid = live[pid]['ppid']
                return len(seen)
            order = sorted(live, key=lambda pid: (-depth(pid), pid))
            for pid in order:
                self.checked_signal(pid, live[pid], sig, root_running_fn())
            if sig != signal.SIGKILL:
                table = self.sample(root_running_fn())
                for pid, row in self.live(table).items():
                    if row['state'].startswith('T'):
                        self.checked_signal(pid, row, signal.SIGCONT, root_running_fn())
            deadline = self.now_fn()+grace
            while True:
                table = self.sample(root_running_fn())
                if not self.live(table) or self.now_fn() >= deadline:
                    break
                self.sleep_fn(.25)
        checks = []
        for index in range(2):
            table = self.sample(root_running_fn())
            checks.append(dict(owned_survivors=[dict(pid=pid, **row) for pid, row in self.live(table).items()],
                               unknown_project_or_group_processes=self.unknown(table)))
            if index == 0:
                self.sleep_fn(.5)
        proven = self.root_seen and not self.uncertainty and all(
            not check['owned_survivors'] and not check['unknown_project_or_group_processes'] for check in checks)
        return dict(proven=proven, initially_live_after_command=initially_live, checks=checks,
                    uncertainty=self.uncertainty, owned=list(self.owned.values()),
                    observed_new_groups=sorted(self.groups), individually_checked_signals=self.signals,
                    broad_group_signals_used=False)


def validate_plan(plan, manifest_sha):
    assert plan['one_unchanged_release_authorized'] is True and plan['automatic_fire'] is False
    assert plan['root_pid'] == ROOT_PID and plan['runner_sha256'] == sha(__file__)
    assert plan['candidate'] == str(CANDIDATE) and plan['candidate_sha256'] == CANDIDATE_SHA
    assert plan['release_min'] == str(RELEASE_MIN)
    assert plan['reference_archive'] == str(LAST_ARCHIVE)
    assert plan['reference_archive_sha256'] == FIXED_PINS[str(LAST_ARCHIVE)]
    assert plan['isolation_helper'] == str(ISOLATION_HELPER)
    assert plan['isolation_helper_sha256'] == FIXED_PINS[str(ISOLATION_HELPER)]
    tag = plan['tag']
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag) and tag not in ('.', '..')
    assert plan['qualification_output'] == str(HERE/'qualification_v2')
    assert plan['archive_directory'] == str(O/'release/runs'/tag)
    expected = ['bash', str(RELEASE_MIN), str(CANDIDATE), tag,
                plan['qualification_output'], str(LAST_ARCHIVE)]
    assert plan['command'] == expected, 'Complete argv must be the unchanged release invocation'
    required = dict(FIXED_PINS)
    required[str(Path(__file__).resolve())] = plan['runner_sha256']
    required[plan['runtime_verdict']] = plan['runtime_verdict_sha256']
    required[plan['isolation_helper']] = plan['isolation_helper_sha256']
    assert all(plan['source_pins'].get(path) == digest for path, digest in required.items())
    for path, digest in plan['source_pins'].items():
        assert sha(path) == digest, ('changed pin', path)
    verdict = json.loads(Path(plan['runtime_verdict']).read_text())
    assert verdict['successful'] is True and verdict['material_measured_margin_accepted'] is True
    assert verdict['release_source_sha256'] == CANDIDATE_SHA
    assert sha(plan['runtime_verdict']) == plan['runtime_verdict_sha256']
    assert manifest_sha and sha(CANDIDATE) == CANDIDATE_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--manifest-sha', required=True)
    args = parser.parse_args()
    manifest = HERE/'RELEASE_ISOLATED_PLAN_v2.json'
    assert sha(manifest) == args.manifest_sha
    plan = json.loads(manifest.read_text())
    validate_plan(plan, args.manifest_sha)
    helper_path = Path(plan['isolation_helper'])
    spec = importlib.util.spec_from_file_location('frozen_release_isolation_helpers_v2', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert Path(helper.O) == O
    record, log = HERE/'ISOLATION_v2.json', HERE/'release_min_v2.log'
    assert not record.exists() and not log.exists()
    assert not Path(plan['qualification_output']).exists() and not Path(plan['archive_directory']).exists()
    initial = process_table()
    assert ROOT_PID in initial and 'run_queue_astra_v6.sh' in initial[ROOT_PID]['command']
    helper.deny_external_heavy(initial, helper.descendants(initial, ROOT_PID))
    if args.check:
        print(json.dumps(dict(utc=utc(), status='READ_ONLY_RELEASE_READY_NO_EXECUTION',
            manifest_sha256=args.manifest_sha, runner_sha256=sha(__file__), command=plan['command'], no_fire=True)))
        return 0

    rec = dict(start_utc=utc(), status='ISOLATING', manifest_sha256=args.manifest_sha,
               runner_sha256=sha(__file__), source_pins=plan['source_pins'], command=plan['command'],
               no_fire=True, paused={}, ownership_retirement_proven=False)
    with record.open('x') as stream:
        json.dump(rec, stream, indent=2)
        stream.write('\n')
    paused, active, tracker, exitcode = {}, None, None, 1
    began, cleaning, release_rc = time.monotonic(), False, None

    def save():
        temporary = HERE/'ISOLATION_v2.writing'
        temporary.write_text(json.dumps(rec, indent=2)+'\n')
        temporary.replace(record)

    def interrupt(signum, frame):
        rec.setdefault('received_signals', []).append(int(signum))
        if not cleaning:
            raise KeyboardInterrupt('signal %s' % signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def pause(pid, row):
        current = process_table().get(pid)
        assert current and identity(pid, current) == identity(pid, row), ('pause identity changed', pid)
        paused[pid] = row
        rec['paused'] = paused
        save()
        os.kill(pid, signal.SIGSTOP)

    try:
        pause(ROOT_PID, initial[ROOT_PID])
        table = process_table()
        judges = [pid for pid, row in table.items() if row['ppid'] == ROOT_PID
                  and 'run_current32_' in row['command'] and '_fable_v1.py ' in row['command']]
        for pid in judges:
            pause(pid, table[pid])
        print(utc(), 'DRAINING', len(judges), flush=True)
        deadline = time.monotonic()+210
        while True:
            table = process_table()
            tree = helper.descendants(table, ROOT_PID)
            helper.deny_external_heavy(table, tree)
            remaining = {pid for pid in tree-set(paused) if not table[pid]['state'].startswith('Z')}
            if not remaining:
                break
            assert time.monotonic() < deadline, remaining
            time.sleep(2)
        rec.update(drained_utc=utc(), free_pct_before=helper.pressure())
        save()
        assert rec['free_pct_before'] >= 60
        assert sha(manifest) == args.manifest_sha
        validate_plan(plan, args.manifest_sha)
        baseline = process_table()
        helper.deny_external_heavy(baseline, helper.descendants(baseline, ROOT_PID))
        assert not Path(plan['qualification_output']).exists() and not Path(plan['archive_directory']).exists()
        with log.open('x') as stream:
            active = subprocess.Popen(plan['command'], cwd=O, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            tracker = OwnedRelease(baseline, os.getpid(), active.pid)
            rec.update(release_pid=active.pid, release_start_utc=utc(), status='UNCHANGED_RELEASE_RUNNING')
            save()
            print(rec['release_start_utc'], 'UNCHANGED_RELEASE_MIN_START', active.pid, flush=True)
            deadline = time.monotonic()+900
            saved_owned, next_checkpoint = -1, 0.
            while True:
                tracker.sample(active.poll() is None)
                if len(tracker.owned) != saved_owned or time.monotonic() >= next_checkpoint:
                    rec.update(ownership_checkpoint_utc=utc(), owned_observations=list(tracker.owned.values()),
                               observed_new_groups=sorted(tracker.groups))
                    save()
                    saved_owned, next_checkpoint = len(tracker.owned), time.monotonic()+5
                release_rc = active.poll()
                if release_rc is not None:
                    break
                assert time.monotonic() < deadline, 'Outer release lifetime timeout; no retry'
                time.sleep(.5)
        rec.update(release_end_utc=utc(), release_returncode=release_rc, log_sha256=sha(log))
        save()
        assert release_rc == 0, ('release_min failed; no retry', release_rc, str(log))
        assert sha(manifest) == args.manifest_sha
        validate_plan(plan, args.manifest_sha)
        exitcode = 0
    except BaseException as error:
        rec.update(status='FAILED', error=repr(error))
    finally:
        cleaning = True
        rec['cleanup_started_utc'] = utc()
        save()
        retirement = dict(proven=active is None, no_release_process_created=active is None)
        if active is not None and tracker is None:
            retirement['error'] = 'Release created before ownership tracker initialized; identity uncertain'
        if tracker is not None:
            try:
                retirement = tracker.retire(lambda: active.poll() is None)
                if active.poll() is not None:
                    active.wait(timeout=1)
                else:
                    retirement['proven'] = False
                    retirement['error'] = 'Popen remains running after cleanup inventories'
            except BaseException as error:
                retirement = dict(proven=False, error=repr(error), uncertainty=tracker.uncertainty,
                    owned=list(tracker.owned.values()), individually_checked_signals=tracker.signals)
        rec['ownership_retirement'] = retirement
        rec['ownership_retirement_proven'] = retirement['proven']
        resumed, already_stopped, restore_errors = [], [], []
        if retirement['proven']:
            # Reverse order leaves the root dispatcher last. Never resume a PID
            # with changed identity, or a process that was already stopped.
            for pid, before in reversed(list(paused.items())):
                try:
                    now = process_table().get(pid)
                    if now and identity(pid, now) == identity(pid, before):
                        if before['state'].startswith('T'):
                            already_stopped.append(pid)
                        else:
                            os.kill(pid, signal.SIGCONT)
                            resumed.append(pid)
                except BaseException as error:
                    restore_errors.append(dict(pid=pid, error=repr(error)))
                    break
        else:
            exitcode = 1
            rec['status'] = 'FAILED_CLEANUP_UNCERTAIN_RESEARCH_LEFT_PAUSED_ROOT_REVIEW_REQUIRED'
        if restore_errors:
            exitcode = 1
            rec['status'] = 'FAILED_RESEARCH_RESTORE_ROOT_REVIEW_REQUIRED'
        if retirement.get('initially_live_after_command'):
            exitcode = 1
            if retirement['proven']:
                rec['status'] = 'FAILED_RELEASE_LEFT_DESCENDANTS_CLEANED_NO_RETRY'
        if exitcode == 0:
            rec['status'] = 'UNCHANGED_RELEASE_SCRIPT_COMPLETED_ROOT_ARTIFACT_REVIEW_REQUIRED'
        rec.update(end_utc=utc(), pause_elapsed_s=time.monotonic()-began, resumed=resumed,
            previously_stopped_preserved=already_stopped, restore_errors=restore_errors, exitcode=exitcode,
            research_resumption_withheld=not retirement['proven'],
            not_resumed=[pid for pid in paused if pid not in resumed and pid not in already_stopped])
        try:
            rec['free_pct_after'] = helper.pressure()
        except Exception as error:
            rec['free_pct_after_error'] = repr(error)
        save()
        print(rec['end_utc'], rec['status'], 'resumed', resumed, 'prior_stops', already_stopped,
            'retirement_proven', retirement['proven'], 'exitcode', exitcode, flush=True)
    return exitcode


if __name__ == '__main__':
    raise SystemExit(main())
