"""One authorized missing-metadata setup measurement; retain original census."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
H = D.parent
P = H.parent
candidate = Path(sys.argv[1]).resolve()
assert len(sys.argv) == 4 and candidate == D/'candidate.py'
assert sys.argv[2] == '35,100' and sys.argv[3] == D.name+'_setup_only_v1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == '9ea3a975561d6fae2ae8806b9812d81fe7cadc23845f86cce14b80696d0bf7d6'
server_python = P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
client_python = P/'data_gate/.client-venv/bin/python'
socket = Path('/private/tmp')/f'codex_p2_read7_setup_{os.getpid()}.sock'
assert not socket.exists()
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
client_script = D/'setup_only_harness.py'
def pin(p):
    raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
with (D/'setup_only_server.log').open('x') as server_log:
    server = subprocess.Popen([str(server_python), '-m', 'flopscope_server', '--url', 'ipc://'+str(socket), '--timeout', '1800'],
        env=env, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if server.poll() is not None: raise RuntimeError('server exited before startup')
            if socket.exists(): break
            time.sleep(.05)
        else: raise RuntimeError('server did not create its socket')
        env.update(FLOPSCOPE_SERVER_URL='ipc://'+str(socket), E2_SERVER_PID=str(server.pid))
        with (D/'setup_only.log').open('x') as client_log:
            proc = subprocess.run([str(client_python), '-B', '-u', str(client_script), str(candidate), sys.argv[2], sys.argv[3]],
                env=env, stdout=client_log, stderr=subprocess.STDOUT)
        if proc.returncode: raise SystemExit(proc.returncode)
    finally:
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait(timeout=10)
        if socket.exists(): socket.unlink()
record = dict(candidate=pin(candidate), original_harness=pin(H/'census_boundary_setup_snapshot_v1.py'),
    harness=pin(client_script), launcher=pin(Path(__file__).resolve()), setup_log=pin(D/'setup_only.log'),
    client_exit=proc.returncode, server_exit=server.returncode, server_retired=True,
    prediction_calls=0, original_census_unchanged=True, reason='Original setup stdout was lost by root filtering; one setup-only measurement.')
with (D/'setup_only_launcher.json').open('x') as f:
    json.dump(record, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps(record), flush=True)
