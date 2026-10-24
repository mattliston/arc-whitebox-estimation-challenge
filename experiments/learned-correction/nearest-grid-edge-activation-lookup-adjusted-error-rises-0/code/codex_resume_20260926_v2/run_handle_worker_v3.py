"""Serial guarded real-server handle census, with explicit child retirement."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
candidate = Path(sys.argv[1]).resolve()
ranks, tag = sys.argv[2:4]
assert Path(tag).name == tag
out = HERE/tag
out.mkdir()
server_python = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
client_python = P2/'data_gate/.client-venv/bin/python'
socket = Path('/private/tmp')/f'codex_p2_handles_{os.getpid()}.sock'
assert not socket.exists()
url = 'ipc://'+str(socket)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
           FLOPSCOPE_MAX_ARRAY_BYTES='4294967296', FLOPSCOPE_MAX_ARRAY_COUNT='70000',
           CODEX_HANDLE_RECEIPT=str(out/'server_handles.json'))
with (out/'server.log').open('x') as log:
    server = subprocess.Popen([str(server_python),str(HERE/'handle_server_v2.py'),
              '--url',url,'--timeout','1200'], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if server.poll() is not None: raise RuntimeError('handle server exited')
            if socket.exists(): break
            time.sleep(.05)
        else: raise RuntimeError('handle server socket missing')
        child_env = dict(env, FLOPSCOPE_SERVER_URL=url)
        result = subprocess.run([str(client_python),'-B','-u',str(HERE/'handle_client_v2.py'),
                    str(candidate),ranks,str(out)], env=child_env, timeout=900)
        if result.returncode: raise RuntimeError(f'handle client failed: {result.returncode}')
    finally:
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=10)
        if socket.exists(): socket.unlink()
receipt = dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
               client_exit=result.returncode, server_exit=server.returncode, server_retired=True,
               instrumentation='ArrayStore.put/free/clear counters plus metadata-only root snapshots; no retained array references',
               handles=json.loads((out/'server_handles.json').read_text()))
assert receipt['handles']['peak_handles'] < 70000
(out/'launcher.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='handles'} | dict(peak_handles=receipt['handles']['peak_handles'])), flush=True)
