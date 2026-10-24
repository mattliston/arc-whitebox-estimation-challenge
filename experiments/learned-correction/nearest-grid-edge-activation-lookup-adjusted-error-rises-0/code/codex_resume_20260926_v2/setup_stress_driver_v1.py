"""Three fresh setup trials with two bounded, low-memory CPU load processes.

Run this whole process group under guard.py, serially with all estimator jobs.
Every client/server and load child is explicitly retired. The load processes
hold one64KiB byte buffer each; no second large estimator is launched.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;P2=HERE.parent
candidate=Path(sys.argv[1]).resolve();tag=sys.argv[2]
assert Path(tag).name==tag
out=HERE/tag;out.mkdir()
server_python=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
client_python=P2/'data_gate/.client-venv/bin/python'
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
load_code='import hashlib,time\nb=b"x"*65536\nend=time.monotonic()+120\nwhile time.monotonic()<end: hashlib.sha256(b).digest()\n'
workers=[];records=[]
def retire(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
try:
    for _ in range(2):
        workers.append(subprocess.Popen([sys.executable,'-c',load_code],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
    for repeat in range(3):
        socket=Path('/private/tmp')/f'codex_p2_setup_{os.getpid()}_{repeat}.sock'
        assert not socket.exists()
        url='ipc://'+str(socket)
        with (out/f'server_{repeat}.log').open('x') as log:
            server=subprocess.Popen([str(server_python),'-m','flopscope_server','--url',url,'--timeout','60'],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if server.poll() is not None:raise RuntimeError('setup server exited')
                    if socket.exists():break
                    time.sleep(.05)
                else:raise RuntimeError('setup socket did not appear')
                assert all(p.poll() is None for p in workers),'controlled load stopped early'
                child_env=dict(env,FLOPSCOPE_SERVER_URL=url)
                target=out/f'setup_{repeat}.json'
                result=subprocess.run([str(client_python),'-B',str(HERE/'setup_only_client_v1.py'),str(candidate),str(target)],env=child_env,timeout=30)
                if result.returncode:raise RuntimeError(f'setup client exit {result.returncode}')
                record=json.loads(target.read_text())
                record.update(repeat=repeat,load_processes=2,system_load=os.getloadavg())
                records.append(record)
                (out/'summary.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    controlled_load='two64KiB-buffer CPU processes; one fresh setup at a time',records=records),indent=2)+'\n')
            finally:
                retire(server)
                if socket.exists():socket.unlink()
finally:
    for worker in workers:retire(worker)
assert len(records)==3
print('ALL_SETUP_TRIALS_COMPLETE',flush=True)
