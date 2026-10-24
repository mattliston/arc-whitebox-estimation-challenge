"""[opus-callform] start one server_light (OpenBLAS, 1 thread) on an ipc socket, run mb_client_v1.py with the release CLIENT_PY, stop it.
usage: mb_run_v1.py <out.json> [N]"""
import os, subprocess, sys, time
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'release'))
from common import CLIENT_PY, R, SERVER_PY, THREADS
out = Path(sys.argv[1]).resolve(); N = sys.argv[2] if len(sys.argv) > 2 else '2000'
sock = Path('/private/tmp')/f'opus_callform_{os.getpid()}.sock'; assert not sock.exists()
url = 'ipc://' + str(sock)
env = dict(os.environ, **THREADS, FLOPSCOPE_MAX_ARRAY_COUNT='100000', RELEASE_SERVER_RECEIPT=str(out.with_suffix('.server.json')),
           RELEASE_SERVER_MARKS=str(out.with_suffix('.marks.jsonl')))
with open(out.with_suffix('.server.log'), 'x') as log:
    server = subprocess.Popen([str(SERVER_PY), '-B', str(R/'server_light.py'), '--url', url, '--timeout', '900'], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(200):
            if sock.exists(): break
            time.sleep(.05)
        client = subprocess.run([str(CLIENT_PY), '-B', '-u', str(Path(__file__).resolve().parent/'mb_client_v1.py'), str(out), N], env=dict(env, FLOPSCOPE_SERVER_URL=url))
    finally:
        server.terminate()
        try: server.wait(timeout=15)
        except subprocess.TimeoutExpired: server.kill()
        if sock.exists(): sock.unlink()
sys.exit(client.returncode)
