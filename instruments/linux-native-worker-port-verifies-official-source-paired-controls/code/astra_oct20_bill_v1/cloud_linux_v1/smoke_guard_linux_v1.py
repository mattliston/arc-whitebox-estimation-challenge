"""Tiny Linux acceptance check: normal exit and timeout kill of an owned grandchild.

No estimator/SDK import. Creates only a1MiB allocation and short-lived sleepers.
Run on a quiet VM before scientific predictions; not yet run on Linux locally.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common_linux_v1 import utc, write_new

p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
p.add_argument('--state-dir', type=Path, required=True); p.add_argument('--cgroup-root', type=Path)
a = p.parse_args(); assert sys.platform == 'linux'
a.out.mkdir(parents=True, exist_ok=False)
guard = Path(__file__).with_name('guard_linux_v1.py')
base = [sys.executable, '-B', str(guard), '--state-dir', str(a.state_dir), '--max-active', '1']
if a.cgroup_root:
    base += ['--cgroup-root', str(a.cgroup_root)]
normal = a.out / 'normal.json'; timeout = a.out / 'timeout.json'
c1 = subprocess.run(base + ['--receipt', str(normal), '--timeout', '10', '--', sys.executable, '-c',
                           'x=bytearray(1048576); print(len(x))'], check=False)
child_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)"
c2 = subprocess.run(base + ['--receipt', str(timeout), '--timeout', '1', '--', sys.executable, '-c', child_code], check=False)
r1 = json.loads(normal.read_text()); r2 = json.loads(timeout.read_text())
assert c1.returncode == 0 and r1['successful'] and r1['whole_group_retired']
assert c2.returncode != 0 and r2['reason'] == 'group_timeout' and r2['whole_group_retired']
write_new(a.out / 'PASS.json', dict(utc=utc(), status='PASS', tests=['normal_exit', 'timeout_retires_owned_grandchild'],
                                  limitations=['memory/OOM limits not stress-tested by this tiny probe']))
print('LINUX_GUARD_SMOKE_PASS')
