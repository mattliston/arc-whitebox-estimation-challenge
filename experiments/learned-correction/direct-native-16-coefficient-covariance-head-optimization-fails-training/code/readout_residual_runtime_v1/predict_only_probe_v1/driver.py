"""Own and retire one ordinary server/client pair in the outer guarded group."""
import os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent.parent;P2=D.parent
candidate=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();ranks=sys.argv[3]
assert not out.exists()
socket=Path(f'/tmp/phase2-plain-{os.getpid()}.sock');assert not socket.exists()
url='ipc://'+str(socket)
backend=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
client=P2/'data_gate/.client-venv/bin/python'
with out.with_suffix('.server.log').open('x') as log:
    server=subprocess.Popen([str(backend),'-m','flopscope_server','--url',url,'--timeout','600'],stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+10
        while not socket.exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        env=dict(os.environ,FLOPSCOPE_SERVER_URL=url)
        done=subprocess.run([str(client),'-B','-W','ignore','-u',str(HERE/'client.py'),str(candidate),str(out),ranks],env=env,timeout=500)
        if done.returncode:raise SystemExit(done.returncode)
    finally:
        server.terminate()
        try:server.wait(timeout=5)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        socket.unlink(missing_ok=True)
