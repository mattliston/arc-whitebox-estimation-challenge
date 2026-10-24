"""Serial controller for the single fixed full80 endpoint. Each segment has its own guard."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, os
D=Path(__file__).resolve().parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
assert len(sys.argv)==2 and pin(D/'protocol.json')['sha256']==sys.argv[1]
protocol=json.loads((D/'protocol.json').read_text())
review=json.loads((D/'source_review.json').read_text())
assert review['passed'] and review['protocol']==pin(D/'protocol.json')
for p in protocol['pins']:
 assert pin(p['path'])==p
def status(**kw):
 v=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw)
 tmp=D/'controller_status.tmp';tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,D/'controller_status.json')
status(state='starting',completed_updates=0)
for i in range(20):
 gpath=D/('segment_%02d_guard.json'%i)
 result=D/('segment_%02d'%i)/'completed.json'
 cmd=[protocol['light'],'-B',protocol['guard']['path'],'--record',str(gpath),'--seconds','600','--rss-limit','7500000000',protocol['torch_runtime'],'-B',str(D/'train_segment.py'),sys.argv[1],str(i)]
 if gpath.exists() or result.exists():
  assert gpath.exists() and result.exists(),'Incomplete previous segment: inspect, never silently rerun'
  g=json.loads(gpath.read_text());r=json.loads(result.read_text())
  assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7500000000
  assert r['optimizer_updates']==4*(i+1) and r['checkpoint']==pin(r['checkpoint']['path'])
  continue
 status(state='training',segment=i,completed_updates=4*i,guard=str(gpath),command=cmd)
 print(json.dumps(dict(event='segment_start',segment=i,first_update=4*i+1,last_update=4*(i+1))),flush=True)
 with (D/('segment_%02d.log'%i)).open('x') as log:
  rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:
  status(state='stopped',segment=i,completed_updates=4*i,returncode=rc)
  raise RuntimeError('Segment failed; inspect its guard and log')
 g=json.loads(gpath.read_text());r=json.loads(result.read_text())
 assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7500000000
 assert g['command']==[protocol['torch_runtime'],'-B',str(D/'train_segment.py'),sys.argv[1],str(i)]
 assert r['optimizer_updates']==4*(i+1) and r['checkpoint']==pin(r['checkpoint']['path'])
 status(state='between_segments',segment=i,completed_updates=4*(i+1))
 print(json.dumps(dict(event='segment_complete',segment=i,updates=4*(i+1),seconds=g['elapsed_s'],rss=g['max_rss_bytes'],swap=g['max_swap_growth_bytes'])),flush=True)
once(D/'training_complete.json',dict(completed=True,updates=80,protocol=pin(D/'protocol.json'),endpoint=pin(D/'endpoint.npz'),dials=pin(D/'endpoint_dials.json'),segments=[pin(D/('segment_%02d_guard.json'%i)) for i in range(20)],selected_endpoint=80,validation_read=False))
status(state='completed',completed_updates=80)
print(json.dumps(dict(event='training_complete',updates=80)),flush=True)

