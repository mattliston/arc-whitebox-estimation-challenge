# New immutable micro+COPY_ALL+edge release qualification; permanently disabled inherited governor.
"""Quiet paired production-boundary check, followed by unchanged release gates.

Temporarily pause the identified research tree; always resume it. No submission.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

K = Path(__file__).resolve().parent
O = K.parents[2]
PY = Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
ROOT_PID = 77740
TAG = 'astra_micro_copyall_ecut_nogov_4b8baf51_v1'
OUT = K / 'runtime_micro_copyall_ecut_nogov_quiet_v1'
Q = K / 'qual_micro_copyall_ecut_nogov_4b8baf51_v1'
RECORD = K / 'MICRO_COPYALL_ECUT_NOGOV_RELEASE_ISOLATION_v1.json'
GUARD = O / 'lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py'
BOUNDARY = O / 'lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_run_dual_v3.py'
SOURCES = [
    ('bank', K / 'candidate_rel_gram_pair_kclm_copy_astra.py', 'fd8739582ffdd04decc5fe6bbb2f99fb0b3dc02a2ee786d684bd21bbf52d974d'),
    ('micro', K / 'candidate_rel_micro_copyall_ecut_nogov_astra.py', '4b8baf51fe5ced80f46907aea7ab4461124b993d17a178d73c9c75230a6100b7'),
]

def stamp():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def processes():
    rows = {}
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,lstart=,command='], text=True).splitlines():
        p = line.split(None, 7)
        if len(p) == 8:
            rows[int(p[0])] = dict(ppid=int(p[1]), started=' '.join(p[2:7]), command=p[7])
    return rows

def pressure():
    text = subprocess.check_output(['memory_pressure'], text=True)
    return int(text.rsplit('System-wide memory free percentage:', 1)[1].strip().strip('%'))

def interrupted(signum, frame):
    raise KeyboardInterrupt('signal %s' % signum)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
assert not RECORD.exists() and not OUT.exists() and not Q.exists()
for _, source, digest in SOURCES:
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
assert hashlib.sha256(BOUNDARY.read_bytes()).hexdigest() == '3d829d2a28268829ad208b7b71514f1c708211e74d04c5e6b8dacb003baa60bb'
audit = json.loads((K / 'MICRO_COPYALL_ECUT_NOGOV_BUILD_AUDIT_v1.json').read_text())
assert audit['variants'][1]['sha256'] == SOURCES[1][2]
assert audit['no_rescue_counts']['rows'] == 32 and audit['no_rescue_counts']['all_calls_zero'] and audit['no_rescue_counts']['all_control_null']
paired = audit['parent_science_result']
assert paired['reference_submission_id'] == 337393 and paired['all32']['adjusted_pct'] < 0 and paired['heldout25']['adjusted_pct'] < 0
score = paired['frozen_v4a_qualification']
assert score['science_bar_cleared'] and score['all32']['adjusted_pct'] <= -.35 and score['heldout25']['adjusted_pct'] < 0
initial = processes()
assert ROOT_PID in initial and 'run_queue_astra_v6.sh' in initial[ROOT_PID]['command']
extra_heavy = [r for p, r in initial.items() if '/run_current32_aux30_astra_v1.py ' in r['command'] and r['command'].startswith(str(PY))]
assert not extra_heavy, extra_heavy
OUT.mkdir()
rec = dict(start_utc=stamp(), sources=[dict(label=a, path=str(b), sha256=c) for a,b,c in SOURCES],
           science_all32=paired['all32'], science_held25=paired['heldout25'], reference_submission=337393, frozen_v4a_qualification=score,
           timing_caveat='Paused research wall times are not causal performance evidence; numerical and billed-operation results are unaffected.',
           scope='Two-rank quiet production-boundary runtime comparison, then original release_min gates. No submission.')
paused = {}
active = None
began = time.monotonic()
exitcode = 1

def save():
    RECORD.write_text(json.dumps(rec, indent=2) + '\n')

def run(command, log, timeout):
    global active
    with log.open('x') as stream:
        active = subprocess.Popen(command, cwd=O, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return active.wait(timeout=timeout)
        finally:
            if active.poll() is None:
                active.send_signal(signal.SIGINT)
                active.wait(timeout=25)
            active = None

try:
    # Pause dispatchers only. Let their existing guarded groups finish and release
    # their memory before starting a production-boundary measurement.
    os.kill(ROOT_PID, signal.SIGSTOP)
    paused[ROOT_PID] = initial[ROOT_PID]
    rows = processes()
    judges = [pid for pid, row in rows.items() if row['ppid'] == ROOT_PID
              and 'run_current32_' in row['command'] and '_fable_v1.py ' in row['command']]
    for pid in judges:
        os.kill(pid, signal.SIGSTOP)
        paused[pid] = rows[pid]
    print(stamp(), 'DRAINING_ACTIVE_GROUPS', len(judges), flush=True)
    drain_end = time.monotonic() + 180
    while True:
        rows = processes()
        descendants = {ROOT_PID}
        while True:
            more = {p for p,r in rows.items() if r['ppid'] in descendants}
            if not more - descendants:
                break
            descendants |= more
        active_descendants = descendants - set(paused)
        if not active_descendants:
            break
        assert time.monotonic() < drain_end, active_descendants
        time.sleep(2)
    rec['drained_utc'] = stamp()
    rec.update(paused=paused, free_pct_before=pressure(), runs=[])
    save()
    assert rec['free_pct_before'] >= 60, rec['free_pct_before']
    print(stamp(), 'RESEARCH_PAUSED', len(paused), 'free_pct', rec['free_pct_before'], flush=True)
    for label, source, digest in SOURCES:
        out = OUT / label
        out.mkdir()
        command = [str(PY), '-B', str(GUARD), '--record', str(out/'GUARD_v1.json'), '--seconds', '125',
                   '--footprint-limit', '7500000000', '--rss-backstop', '7500000000',
                   str(PY), '-B', str(BOUNDARY), str(source), '35,200', str(out), '--short-setup']
        print(stamp(), 'BOUNDARY_START', label, flush=True)
        status = run(command, out/'run.log', 140)
        rec['runs'].append(dict(label=label, command=command, status=status))
        save()
        assert status == 0, (label, status)
        guard = json.loads((out/'GUARD_v1.json').read_text())
        assert guard['completed'] and guard['whole_group_retired'] and not guard['remaining_rows']
        native_rows = [json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
        assert len(native_rows) == 2
        assert all(row.get('fast_rescue_stats', {}).get('calls') == 0 for row in native_rows)
        print(stamp(), 'BOUNDARY_DONE', label, 'seconds', guard['elapsed_s'], flush=True)
    command = ['bash', str(O/'lead/runs/fable_oct19_l13rank_v1/release_min_v1.sh'), str(SOURCES[1][1]), TAG, str(Q),
               str(O/'release/runs/astra_micro_kclm_copyall_4a40feb6_v1/submission.tar.gz')]
    rec['release_command'] = command
    save()
    status = run(command, OUT/'release_min.log', 100)
    rec['release_gate_returncode'] = status
    assert status == 0, status
    exitcode = 0
finally:
    rows = processes()
    resumed = []
    for pid, before in reversed(list(paused.items())):
        now = rows.get(pid)
        if now and now['started'] == before['started'] and now['command'] == before['command']:
            try:
                os.kill(pid, signal.SIGCONT)
                resumed.append(pid)
            except ProcessLookupError:
                pass
    rec.update(end_utc=stamp(), pause_elapsed_s=time.monotonic()-began, resumed=resumed, free_pct_after=pressure(), exitcode=exitcode)
    save()
    print(rec['end_utc'], 'RESEARCH_RESUMED', len(resumed), 'exitcode', exitcode, flush=True)
raise SystemExit(exitcode)
