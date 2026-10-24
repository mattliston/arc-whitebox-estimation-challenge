"""Qualify an accepted birth fit with the official SDK under the original guard."""
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
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

assert sys.platform == 'linux'
script = HERE / 'build_birth_live_v3.py'
checker = HERE / 'check_import_v2.py'
guard = PORT / 'guard_linux_v7.py'
assert sha(script) == '4c156649645ad784c0562292ab95ef59abf65b3918c8fa4a7ec6c11570e2a4c8'
assert sha(checker) == 'ea7c443f5dd57580cc45e480066d066ee9465a41ae27365aacb513434640ae55'
assert sha(guard) == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
assert read(RUNS / 'birth_consumer_fit_live_v2/FIT_RESULT_v1.json')['status'] == 'PASS_TEACHER_FIDELITY_GATE'
g = read(RUNS / 'birth_consumer_fit_live_v2_GUARD.json')
assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
assert not (HERE / 'native_v1').exists()
out = RUNS / 'birth_consumer_build_v1'
judge = '/home/arcworker/arc-envs/judge/bin/python'
server = '/home/arcworker/arc-envs/server/bin/python'
command = [judge, '-B', str(guard), '--receipt', str(out) + '_GUARD.json',
           '--state-dir', str(RUNS / 'admission'), '--max-active', '36',
           '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--',
           server, '-B', str(script)]
env = dict(os.environ, **thread_env(2), ARC_BLAS_THREADS='2')
log = Path(str(out) + '.log')
with log.open('x') as f:
    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=f,
                            stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(utc=utc(), pid=proc.pid, command=command, log=str(log),
               script_sha256=sha(script), checker_sha256=sha(checker), guard_sha256=sha(guard),
               fit_sha256=sha(RUNS / 'birth_consumer_fit_live_v2/FIT_RESULT_v1.json'),
               root_authorized=True, native_candidate_launches=0, truth_access=False)
path = HERE / 'BUILD_LIVE_LAUNCH_v3.json'
with path.open('x') as f:
    json.dump(receipt, f, indent=2)
    f.write('\n')
print(json.dumps(receipt, indent=2))

