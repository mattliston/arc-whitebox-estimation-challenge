"""Launch the root-authorized fixed teacher fit under the original v7 guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/arcworker/arc-runs')
PORT = HERE.parents[2] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import thread_env
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

p = argparse.ArgumentParser()
p.add_argument('--selftest', action='store_true')
a = p.parse_args()
assert sys.platform == 'linux'
script = HERE / 'fit_birth_live_v2.py'
guard = PORT / 'guard_linux_v7.py'
assert sha(script) == 'a8926febd6fa8b58010597b7067d6efdb10c22aa4de8a65ea01df6ace6bc2601'
assert sha(guard) == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
protocol = json.loads((HERE / 'FIT_PROTOCOL_live_v2.json').read_text())
assert protocol['source_sha256'] == sha(script) and not protocol['truth_access']
if not a.selftest:
    assert json.loads((RUNS / 'birth_consumer_train4_v1/COMPLETE.json').read_text())['successful']
    assert json.loads((RUNS / 'birth_consumer_fit_selftest_v1/SELFTEST_v1.json').read_text())['status'] == 'PASS'
    g = json.loads((RUNS / 'birth_consumer_fit_selftest_v1_GUARD.json').read_text())
    assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
name = 'birth_consumer_fit_selftest_v1' if a.selftest else 'birth_consumer_fit_live_v2'
out = RUNS / name
assert not out.exists()
judge = '/home/arcworker/arc-envs/judge/bin/python'
command = [judge, '-B', str(guard), '--receipt', str(out) + '_GUARD.json',
           '--state-dir', str(RUNS / 'admission'), '--max-active', '36',
           '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--',
           judge, '-B', str(script), '--out', str(out)]
if a.selftest:
    command.append('--selftest')
env = dict(os.environ, **thread_env(2), ARC_BLAS_THREADS='2')
log = Path(str(out) + '.log')
with log.open('x') as f:
    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=f,
                            stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(utc=utc(), pid=proc.pid, command=command, out=str(out), log=str(log),
               script_sha256=sha(script), guard_sha256=sha(guard),
               protocol_sha256=sha(HERE / 'FIT_PROTOCOL_live_v2.json'), root_authorized=True,
               truth_access=False, native_candidate_launches=0)
path = HERE / ('SELFTEST_LAUNCH_v1.json' if a.selftest else 'FIT_LIVE_LAUNCH_v2.json')
with path.open('x') as f:
    json.dump(receipt, f, indent=2)
    f.write('\n')
print(json.dumps(receipt, indent=2))
