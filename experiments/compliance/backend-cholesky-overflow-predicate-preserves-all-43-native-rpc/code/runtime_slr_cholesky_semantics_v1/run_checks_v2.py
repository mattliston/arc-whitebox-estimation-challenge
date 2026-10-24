"""One guarded semantic check; no setup/model/weights/truth."""
import json,os,subprocess,sys,tempfile,time
from pathlib import Path
H=Path(__file__).resolve().parent
assert os.environ.get('ARC_GUARDED_PGID')==str(os.getpgrp())
sp='/home/arcworker/arc-envs/server/bin/python';cp='/home/arcworker/arc-envs/client/bin/python'
subprocess.run([sp,'-B',str(H/'check_slr_v2.py'),'--mode','native'],check=True)
with tempfile.TemporaryDirectory(prefix='arc-slr-') as td,(H/'RPC_SERVER_v2.log').open('x') as log:
 socket=Path(td)/'s';url='ipc://'+str(socket);server=subprocess.Popen([sp,'-B','-m','flopscope_server','--url',url,'--timeout','60'],stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(600):
   if server.poll() is not None:raise RuntimeError('server exited')
   if socket.exists():break
   time.sleep(.05)
  else:raise RuntimeError('server socket timeout')
  subprocess.run([cp,'-B',str(H/'check_slr_v2.py'),'--mode','rpc'],env=dict(os.environ,FLOPSCOPE_SERVER_URL=url),check=True)
 finally:
  if server.poll() is None:server.terminate()
  try:server.wait(timeout=8)
  except subprocess.TimeoutExpired:server.kill();server.wait(timeout=8)
print(json.dumps(dict(successful=True,native_and_rpc=True,owned_server_retired=True)))
