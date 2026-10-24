"""TAG opus-release: one worker session at the production boundary = one flopscope server (OpenBLAS, 1 thread) + one numpy-free
client that runs setup() and then the given networks. Run it under the guard (stage_c_boundary.py / stage_d_serving.py do that).
Derived from H/run_boundary_census.py and H/run_handle_worker_v3.py.
usage: boundary_run.py <candidate.py> <ranks> <outdir> [--sites] [--short-setup | --setup-slowdown X] [--handle-cap N]"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,'/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
from common import R, THREADS, sha
D = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1')
CLIENT_PY = D / "boundary310_v1/bin/python"
SERVER_PY = D / "server310_openblas_v1/bin/python"

candidate = Path(sys.argv[1]).resolve()
ranks = sys.argv[2]
out = Path(sys.argv[3]).resolve()
extra = sys.argv[4:]
cap = extra[extra.index('--handle-cap') + 1] if '--handle-cap' in extra else '100000'   # the platform's cap is 100,000
if '--handle-cap' in extra:
    del extra[extra.index('--handle-cap'):extra.index('--handle-cap') + 2]              # the rest goes to the client
socket = Path('/private/tmp')/f'opus_release_{os.getpid()}.sock'
assert not socket.exists()
url = 'ipc://' + str(socket)
receipt = out/'server.json'
env = dict(os.environ, **THREADS, FLOPSCOPE_MAX_ARRAY_COUNT=cap, RELEASE_SERVER_RECEIPT=str(receipt),
           RELEASE_SERVER_MARKS=str(out/'server_marks.jsonl'))
with (out/'server.log').open('x') as log:
    server = subprocess.Popen([str(SERVER_PY), '-B', str(R/'server_light.py'), '--url', url, '--timeout', '1800'],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(200):
            if server.poll() is not None:
                raise RuntimeError('server exited before startup')
            if socket.exists():
                break
            time.sleep(.05)
        else:
            raise RuntimeError('server did not create its socket')
        client_env = dict(env, FLOPSCOPE_SERVER_URL=url, RELEASE_SERVER_PID=str(server.pid))
        client = subprocess.run([str(CLIENT_PY), '-B', '-u', '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/boundary_client_fusion_v1.py', str(candidate), ranks, str(out),
                                 *extra], env=client_env)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)
        if socket.exists():
            socket.unlink()
record = dict(candidate=str(candidate), sha256=sha(candidate), client_exit=client.returncode, server_exit=server.returncode,
              handle_cap=int(cap), server=json.loads(receipt.read_text()) if receipt.exists() else None)
(out/'launcher.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record), flush=True)
sys.exit(client.returncode)
