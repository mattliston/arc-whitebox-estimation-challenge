"""Owns exactly one isolated client/server pair, retired in finally."""
import os, subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[3]
sys.path.insert(0,str(O/'release'))
from common import CLIENT_PY,SERVER_PY,R,THREADS
name=sys.argv[1];factory=Path(sys.argv[2]).resolve();out=D/(name+'.json');assert not out.exists()
sock=Path('/private/tmp')/('astra_bini_actual_'+str(os.getpid())+'.sock');assert not sock.exists();url='ipc://'+str(sock)
env=dict(os.environ,**THREADS,FLOPSCOPE_MAX_ARRAY_COUNT='100000',RELEASE_SERVER_RECEIPT=str(D/(name+'_server.json')),RELEASE_SERVER_MARKS=str(D/(name+'_marks.jsonl')))
with (D/(name+'_server.log')).open('x')as log:
 server=subprocess.Popen([str(SERVER_PY),'-B',str(R/'server_light.py'),'--url',url,'--timeout','120'],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(400):
   if sock.exists():break
   if server.poll()is not None:raise RuntimeError('server startup failed')
   time.sleep(.025)
  client=subprocess.run([str(CLIENT_PY),'-B',str(D/'client_leaf_v2.py'),str(out),str(factory)],env=dict(env,FLOPSCOPE_SERVER_URL=url,RELEASE_SERVER_PID=str(server.pid)),timeout=100)
 finally:
  server.terminate()
  try:server.wait(timeout=5)
  except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
  if sock.exists():sock.unlink()
sys.exit(client.returncode)
