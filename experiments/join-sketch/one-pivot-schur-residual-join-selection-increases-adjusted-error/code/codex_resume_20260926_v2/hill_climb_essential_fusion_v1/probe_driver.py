"""Run one bounded client/server census, with explicit child retirement.

Invoke this driver through guard.py so the server and client share its RSS,
swap-growth, and timeout limits. No concurrent candidate pair is launched.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

STUDY=Path(__file__).resolve().parent
HERE=STUDY.parent
P2=HERE.parent
candidate=Path(sys.argv[1]).resolve()
ranks=sys.argv[2]
tag=sys.argv[3]
assert Path(tag).name==tag
source=STUDY/'probe_client.py'
client_script=source
payload=source.read_bytes()
mode=sys.argv[4]
assert mode in ('normal','skip_phase2')
server_python=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
client_python=P2/'data_gate/.client-venv/bin/python'
socket=Path('/private/tmp')/f'codex_p2_boundary_{os.getpid()}.sock'
assert not socket.exists()
url='ipc://'+str(socket)
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
assert not (HERE/(tag+'_census.json')).exists()
with (HERE/(tag+'_server.log')).open('x') as log:
    server=subprocess.Popen([str(server_python),'-m','flopscope_server','--url',url,'--timeout','1800'],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('server exited before startup')
            if socket.exists():break
            time.sleep(.05)
        else:raise RuntimeError('server did not create its socket')
        env.update(FLOPSCOPE_SERVER_URL=url,E2_SERVER_PID=str(server.pid))
        proc=subprocess.run([str(client_python),'-B','-u',str(client_script),str(candidate),ranks,tag,mode],env=env)
        if proc.returncode:raise SystemExit(proc.returncode)
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill();server.wait(timeout=10)
        if socket.exists():socket.unlink()
record=dict(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    source=str(source),census_sha256=hashlib.sha256(payload).hexdigest(),server_exit=server.returncode,
    client_exit=proc.returncode,server_retired=True)
with (HERE/(tag+'_launcher.json')).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
