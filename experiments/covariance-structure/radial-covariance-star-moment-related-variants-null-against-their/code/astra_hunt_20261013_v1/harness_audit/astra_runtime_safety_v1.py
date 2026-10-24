"""Stdlib-only process and per-layer memory gate for the private torch worker."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

ASTRA_ROOT = Path(__file__).resolve().parents[1]
LANE_LOCK = (ASTRA_ROOT / 'numerical_lane.lock').open('a')
fcntl.flock(LANE_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice >= 10 required'
threads = os.environ.get('TRAINER_THREADS')
assert threads in ('1', '2'), threads
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == threads, (key, os.environ.get(key))
RECORD = Path(os.environ['ASTRA_PRESSURE_RECORD']).resolve()
RECORD.relative_to(ASTRA_ROOT)
assert not RECORD.exists(), 'Pressure record must be fresh'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'


def check_pressure(layer=None):
    """Check before every layer; while pressure is below the floor, do no arithmetic."""
    while True:
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            output = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
            match = re.search(r'System-wide memory free percentage: (\d+)%', output)
            assert match is not None, 'Unparseable memory pressure'
            pct = int(match.group(1))
            entry = dict(time_utc=now, pid=os.getpid(), layer=layer, free_pct=pct)
        except (OSError, subprocess.SubprocessError, AssertionError) as error:
            pct = None
            entry = dict(time_utc=now, pid=os.getpid(), layer=layer, error=repr(error))
        with RECORD.open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        if pct is not None and pct >= 50:
            return
        print('ASTRA pressure gate waiting:', entry, flush=True)
        time.sleep(10)


check_pressure()
