"""Synthetic process tables only. Never runs a candidate, release command or signal."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from unittest import mock

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'run_release_isolated_root_v2.py'
EXPECTED_SHA = '75f0c976d29da6ea94c72281376fbdeccb2a7cc6bd2e9a6b8de0cb45c5ddcd45'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE) == EXPECTED_SHA
stamp = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
spec = importlib.util.spec_from_file_location('release_wrapper_v2_metadata_fixture', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(ppid, pgid, command, started='synthetic-old', state='S'):
    return dict(ppid=ppid, pgid=pgid, command=command, started=started, state=state)


BASELINE = {
    100: row(1, 100, str(SOURCE)),
    77740: row(1, 77740, str(m.PROJECT)+'/run_queue_astra_v6.sh', state='T'),
    900: row(1, 900, '/Applications/synthetic_external'),
}


class World:
    def __init__(self):
        self.rows = copy.deepcopy(BASELINE)
        self.signals, self.clock, self.fail_next, self.ignore_exit = [], 0., False, set()

    def table(self):
        self.clock += .01
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError('SYNTHETIC process inventory failure')
        return copy.deepcopy(self.rows)

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def kill(self, pid, sig):
        assert pid not in BASELINE, 'Fixture attempted signal to baseline/external identity'
        if pid not in self.rows:
            raise ProcessLookupError(pid)
        self.signals.append(dict(pid=pid, signal=int(sig), identity=list(m.identity(pid, self.rows[pid]))))
        if sig == signal.SIGSTOP:
            self.rows[pid]['state'] = 'T'
        elif sig == signal.SIGCONT:
            self.rows[pid]['state'] = 'S'
        elif pid not in self.ignore_exit:
            del self.rows[pid]

    def add(self, pid, ppid, pgid, name):
        self.rows[pid] = row(ppid, pgid, str(m.PROJECT)+'/SYNTHETIC/'+name, 'synthetic-new-'+str(pid))

    def root_running(self):
        return 200 in self.rows


def tracker(world, register=True):
    # Explicitly inject EVERY side-effect/time dependency. No real process API.
    result = m.OwnedRelease(copy.deepcopy(BASELINE), 100, 200,
        table_fn=world.table, signal_fn=world.kill, now_fn=world.now, sleep_fn=world.sleep)
    if register:
        result.observe(world.table(), world.root_running())
    return result


def root(world):
    world.add(200, 100, 200, 'bash_release_min')


def forbid(*args, **kwargs):
    raise AssertionError('Real process or timing API is forbidden in this metadata fixture')


results = []
with mock.patch.object(m.os, 'kill', forbid), mock.patch.object(m.subprocess, 'Popen', forbid), \
     mock.patch.object(m.subprocess, 'check_output', forbid), mock.patch.object(m.time, 'sleep', forbid):
    w = World(); root(w); t = tracker(w); del w.rows[200]
    proof = t.retire(w.root_running)
    assert proof['proven'] and not w.signals and not proof['initially_live_after_command']
    results.append(dict(case='ordinary_tracked_completion', passed=True, proof=proof))

    w = World(); root(w)
    w.add(300, 200, 300, 'new_session_setup')
    w.add(301, 300, 300, 'setup_worker')
    w.add(400, 301, 400, 'nested_new_session')
    t = tracker(w)
    del w.rows[200]
    w.rows[300]['ppid'] = 1
    w.rows[400]['ppid'] = 1
    w.add(302, 1, 300, 'newly_observed_reparented_same_group_worker')
    proof = t.retire(w.root_running)
    assert proof['proven'] and set(proof['initially_live_after_command']) == {300, 301, 302, 400}
    assert {s['pid'] for s in w.signals} == {300, 301, 302, 400}
    assert not any(pid in w.rows for pid in (200, 300, 301, 302, 400))
    results.append(dict(case='early_bash_exit_nested_sessions_and_reparented_child', passed=True, proof=proof))

    w = World(); w.add(500, 1, 500, 'unobserved_orphan')
    t = tracker(w, register=False); proof = t.retire(w.root_running)
    assert not proof['proven'] and not w.signals and proof['uncertainty']
    assert proof['checks'][0]['unknown_project_or_group_processes'][0]['pid'] == 500
    results.append(dict(case='root_exited_before_initial_observation', passed=True, proof=proof))

    w = World(); root(w); w.add(300, 200, 300, 'old_owned_worker'); t = tracker(w)
    del w.rows[200]
    w.rows[300] = row(1, 300, '/Applications/unrelated_reused_pid', 'synthetic-reused-start')
    proof = t.retire(w.root_running)
    assert not proof['proven'] and not w.signals and 300 in w.rows
    assert proof['checks'][0]['unknown_project_or_group_processes'][0]['pid'] == 300
    results.append(dict(case='pid_reuse_not_signalled', passed=True, proof=proof))

    w = World(); root(w); t = tracker(w); del w.rows[200]
    w.add(888, 1, 888, 'unowned_new_project_process')
    proof = t.retire(w.root_running)
    assert not proof['proven'] and not w.signals and 888 in w.rows
    results.append(dict(case='unknown_project_process_holds_restoration', passed=True, proof=proof))

    w = World(); root(w); t = tracker(w); w.fail_next = True
    try:
        t.sample(True)
        raise AssertionError('Expected synthetic inventory failure')
    except RuntimeError as error:
        assert 'SYNTHETIC' in str(error)
    del w.rows[200]
    proof = t.retire(w.root_running)
    assert not proof['proven'] and proof['uncertainty'] and not w.signals
    results.append(dict(case='inventory_failure_stays_uncertain_after_recovery', passed=True, proof=proof))

    w = World(); root(w); w.add(300, 200, 300, 'nonretiring_worker'); t = tracker(w)
    del w.rows[200]; w.rows[300]['ppid'] = 1; w.ignore_exit.add(300)
    proof = t.retire(w.root_running)
    assert not proof['proven'] and 300 in w.rows
    assert {int(signal.SIGINT), int(signal.SIGTERM), int(signal.SIGKILL)} <= {x['signal'] for x in w.signals}
    assert all(x['pid'] == 300 for x in w.signals)
    assert all(check['owned_survivors'][0]['pid'] == 300 for check in proof['checks'])
    results.append(dict(case='survivor_after_all_three_signals_holds_restoration', passed=True, proof=proof))

    synthetic_path = '/SYNTHETIC_IN_MEMORY_ONLY_NO_RUNTIME_AUTHORITY'
    synthetic_digest = 'a'*64
    wrapper_sha = EXPECTED_SHA
    pins = dict(m.FIXED_PINS)
    pins[str(SOURCE)] = wrapper_sha
    pins[synthetic_path] = synthetic_digest
    tag = 'SYNTHETIC_TAG_NOT_A_REAL_RELEASE'
    plan = dict(one_unchanged_release_authorized=True, automatic_fire=False, root_pid=77740,
        runner_sha256=wrapper_sha, candidate=str(m.CANDIDATE), candidate_sha256=m.CANDIDATE_SHA,
        release_min=str(m.RELEASE_MIN), reference_archive=str(m.LAST_ARCHIVE),
        reference_archive_sha256=m.FIXED_PINS[str(m.LAST_ARCHIVE)], tag=tag,
        qualification_output=str(HERE/'qualification_v2'), archive_directory=str(m.O/'release/runs'/tag),
        isolation_helper=str(m.ISOLATION_HELPER), isolation_helper_sha256=m.FIXED_PINS[str(m.ISOLATION_HELPER)],
        runtime_verdict=synthetic_path, runtime_verdict_sha256=synthetic_digest, source_pins=pins)
    plan['command'] = ['bash', str(m.RELEASE_MIN), str(m.CANDIDATE), tag,
                       plan['qualification_output'], str(m.LAST_ARCHIVE)]
    verdict = dict(SYNTHETIC_FIXTURE_ONLY_NOT_A_VERDICT=True, successful=True,
                   material_measured_margin_accepted=True, release_source_sha256=m.CANDIDATE_SHA)
    real_path = m.Path

    class MemoryVerdict:
        def read_text(self):
            return json.dumps(verdict)

    def path_stub(value):
        return MemoryVerdict() if str(value) == synthetic_path else real_path(value)

    def sha_stub(value):
        return pins[str(value)]

    rejected = []
    with mock.patch.object(m, 'Path', path_stub), mock.patch.object(m, 'sha', sha_stub):
        m.validate_plan(copy.deepcopy(plan), 'synthetic-manifest-not-on-disk')
        mutations = []
        for index in range(6):
            changed = copy.deepcopy(plan); changed['command'][index] += '_CHANGED'
            mutations.append(('argv_'+str(index), changed, None))
        for key in ('candidate', 'candidate_sha256', 'reference_archive', 'reference_archive_sha256',
                    'archive_directory', 'isolation_helper', 'runtime_verdict_sha256'):
            changed = copy.deepcopy(plan); changed[key] += '_CHANGED'
            mutations.append((key, changed, None))
        changed = copy.deepcopy(plan); changed['tag'] = '../escape'
        mutations.append(('unsafe_tag', changed, None))
        changed = copy.deepcopy(plan); changed['source_pins'][str(m.RELEASE_MIN)] = 'b'*64
        mutations.append(('changed_original_release_pin', changed, None))
        for key in ('successful', 'material_measured_margin_accepted', 'release_source_sha256'):
            mutations.append(('verdict_'+key, copy.deepcopy(plan), key))
        for name, changed, verdict_key in mutations:
            before = copy.deepcopy(verdict)
            if verdict_key:
                verdict[verdict_key] = False if verdict_key != 'release_source_sha256' else 'wrong-source'
            try:
                m.validate_plan(changed, 'synthetic-manifest-not-on-disk')
            except (AssertionError, KeyError):
                rejected.append(name)
            else:
                raise AssertionError('Invalid metadata binding accepted: '+name)
            finally:
                verdict.clear(); verdict.update(before)
        assert len(rejected) == 18
    results.append(dict(case='all_six_argv_and_identity_verdict_bindings', passed=True,
                        rejected_mutations=rejected, in_memory_only=True, no_actual_manifest=True))

# The fixture never enters main(). Static check anchors the actual restoration
# loop beneath the retirement predicate and the original stopped-state check.
tree = ast.parse(SOURCE.read_text())
restore_if = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "retirement['proven']")
assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == 'kill' for n in ast.walk(restore_if))
assert any(isinstance(n, ast.If) and ast.unparse(n.test) == "before['state'].startswith('T')"
           for n in ast.walk(restore_if))
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'killpg' for n in ast.walk(tree))
assert sha(SOURCE) == EXPECTED_SHA
report = dict(utc=stamp, status='EIGHT_SYNTHETIC_METADATA_FIXTURES_PASS_NO_REAL_PROCESS_ACTION',
    source=str(SOURCE), source_sha256=EXPECTED_SHA, fixture_sha256=sha(__file__),
    cases=results, all_eight_pass=len(results) == 8 and all(x['passed'] for x in results),
    real_process_apis_explicitly_forbidden_during_tests=True,
    process_table_signal_clock_sleep_dependencies_all_injected=True,
    source_imported='wrapper only; no candidate imports', main_called=False,
    restoration_static_predicate_verified=True, no_real_signals=True, no_native=True,
    no_actual_release_or_check=True, no_actual_manifest_or_verdict=True, no_fire=True)
destination = HERE/'SYNTHETIC_FIXTURE_RESULTS_v2.json'
assert not destination.exists()
destination.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(path=str(destination), sha256=sha(destination), all_eight_pass=report['all_eight_pass'])))
