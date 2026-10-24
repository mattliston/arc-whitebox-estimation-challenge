"""Retire one unpromising dispatcher after its active guarded group completes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

K = Path(__file__).resolve().parent
PID = 2343
SOURCE = 'candidate_ntc64_q192_nogov_astra_sci_v1.py'
OUT = K / 'science32_ntc64_q192_nogov_astra_m15'
RECORD = OUT / 'ABORTED_PRIORITY_v1.json'
RESERVE = K / 'queue_v14.txt.reserve_gate_coadapt'


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def processes():
    result = {}
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,lstart=,command='], text=True).splitlines():
        p = line.split(None, 7)
        if len(p) == 8:
            result[int(p[0])] = dict(ppid=int(p[1]), started=' '.join(p[2:7]), command=p[7])
    return result


def interrupted(signum, frame):
    raise KeyboardInterrupt(signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
assert not RECORD.exists() and not RESERVE.exists()
initial = processes()[PID]
assert initial['ppid'] == 77740 and 'run_current32_m15c_fable_v1.py' in initial['command'] and SOURCE in initial['command']
RESERVE.write_text('One primary slot reserved for gate coadaptation fresh screen.\n')
retired = False
try:
    os.kill(PID, signal.SIGSTOP)
    print(utc(), 'NT_DISPATCHER_STOPPED_DRAINING_CURRENT_GROUP', flush=True)
    end = time.monotonic() + 240
    while True:
        rows = processes()
        assert rows.get(PID) == initial, 'Dispatcher identity changed'
        if not any(r['ppid'] == PID for r in rows.values()):
            break
        assert time.monotonic() < end, 'Active group did not finish; leave it intact'
        time.sleep(2)
    parts = []
    for part in sorted(OUT.glob('candidate_part[0-9][0-9]')):
        guard = json.loads((part / 'GUARD_v1.json').read_text())
        assert guard['completed'] and guard['whole_group_retired'] and not guard['remaining_rows']
        parts.append(part.name)
    # No worker remains. Kill only the stopped dispatcher; retain every capture.
    os.kill(PID, signal.SIGKILL)
    retired = True
    record = dict(utc=utc(), action='Retired only the paused dispatcher after clean drain of every guarded worker',
                  pid=PID, process=initial, preserved_complete_parts=parts,
                  candidate_sha256='c25a89a1e05615730563bd472d2e70b6445781c56d3eff5e485541ea53033f4a',
                  status='Unpromising first seven ranks; NOT a completed32 rejection',
                  reference_submission=337393, first7_adjusted_pct=1.6273100531906226,
                  first7_se=.6948162551111087, first7_lower=1, first7_n=7,
                  first7_raw_pct=.3649345272033041, first7_bill_pct=1.2581640548237643,
                  reason='Even hypothetical removal of the full NT incremental bill leaves first7 +.638% versus best. Prioritize fresh coadaptation.',
                  pending_reservation=str(RESERVE))
    with RECORD.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(record['utc'], 'NT_RETIRED', parts, flush=True)
finally:
    if not retired:
        if processes().get(PID) == initial:
            os.kill(PID, signal.SIGCONT)
        RESERVE.unlink(missing_ok=True)
