"""One owned isolated client/server pair, no model."""
import os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[3]
sys.path.insert(0,str(O/'release'))
from common import CLIENT_PY,SERVER_PY,R,THREADS
assert not (D/'CLIENT_TINY_v1.json').exists()
sock=Path('/private/tmp')/('astra_terminalscale_'+str(os.getpid())+'.sock');url='ipc://'+str(sock)
env=dict(os.environ,**THREADS,FLOPSCOPE_MAX_ARRAY_COUNT='10000',RELEASE_SERVER_RECEIPT=str(D/'SERVER_TINY_v1.json'))
with (D/'SERVER_TINY_v1.log').open('x')as log:
    server=subprocess.Popen([str(SERVER_PY),'-B',str(R/'server_light.py'),'--url',url,'--timeout','60'],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(400):
            if sock.exists():break
            if server.poll()is not None:raise RuntimeError('server startup failed')
            time.sleep(.025)
        client=subprocess.run([str(CLIENT_PY),'-B',str(D/'client_tiny_v1.py')],env=dict(env,FLOPSCOPE_SERVER_URL=url),timeout=50)
    finally:
        server.terminate()
        try:server.wait(timeout=5)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        if sock.exists():sock.unlink()
sys.exit(client.returncode)
