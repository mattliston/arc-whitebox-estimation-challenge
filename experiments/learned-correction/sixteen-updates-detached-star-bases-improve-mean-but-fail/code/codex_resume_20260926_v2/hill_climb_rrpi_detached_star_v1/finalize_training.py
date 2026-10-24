"""Verify the two root-run segments and write factual completion receipts."""
from pathlib import Path
import hashlib,json,math,sys
D=Path(__file__).resolve().parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert len(sys.argv)==2 and pin(D/'protocol.json')['sha256']==sys.argv[1]
pp=pin(D/'protocol.json');p=read(D/'protocol.json')
assert p['updates']==8 and p['batch']==4 and p['schedule_steps']==120
for r in p['pins']:assert pin(r['path'])==r
guardpins=[];completionpins=[];step_pins=[];prev=None;records=[]
for i in range(2):
 gp=pin(D/f'segment_{i:02d}_guard.json');g=read(gp['path'])
 assert g['completed'] and g['exit_code']==0 and g['error'] is None
 assert g['whole_group_retired'] and g['remaining_rows']==[]
 assert g['command']==[p['torch_runtime'],'-B',str(D/'train_segment.py'),sys.argv[1],str(i)]
 assert 0<g['elapsed_s']<=600 and 0<g['max_rss_bytes']<g['rss_trigger_bytes']==7500000000
 assert g['max_swap_growth_bytes']==0
 cp=pin(D/f'segment_{i:02d}/completed.json');c=read(cp['path'])
 assert c['completed'] and c['start']==4*i and c['optimizer_updates']==4*(i+1)
 assert c['protocol']==pp and c['trainer']==pin(D/'train_segment.py') and c['baseline']==p['baseline']
 assert c['resumed_from']==prev and c['training_ranks']==p['training_ranks']
 assert pin(c['checkpoint']['path'])==c['checkpoint'];prev=c['checkpoint']
 sp=pin(D/f'segment_{i:02d}/steps.jsonl');rows=[json.loads(x) for x in Path(sp['path']).read_text().splitlines()]
 updates=[x for x in rows if x['event']=='update'];nets=[x for x in rows if x['event']=='network_backward']
 assert [x['update'] for x in updates]==list(range(4*i+1,4*i+5)) and len(nets)==16
 assert all(math.isfinite(x['mse']) and x['mse']>0 for x in nets)
 assert [[y['rank'] for y in x['rows']] for x in updates]==c['segment_ranks']
 assert c['endpoint_selected'] is (i==1)
 records.extend(nets);guardpins.append(gp);completionpins.append(cp);step_pins.append(sp)
oldpin=p['initial_forward_check']['source'];assert pin(oldpin['path'])==oldpin
old=[json.loads(x) for x in Path(oldpin['path']).read_text().splitlines()]
old0=[x for x in old if x['event']=='network_backward' and x['update']==1]
new0=[x for x in records if x['update']==1]
assert len(old0)==len(new0)==4
assert [(x['rank'],x['mse']) for x in old0]==[(x['rank'],x['mse']) for x in new0]
assert len(records)==32 and len({x['rank'] for x in records})==32
last=read(completionpins[-1]['path'])
assert last['endpoint']==pin(D/'endpoint.npz') and last['dials']==pin(D/'endpoint_dials.json')
once(D/'initial_forward_check.json',dict(passed=True,scope='Initial four network MSEs only, not an all-output byte comparison or accuracy gain',protocol=pp,old_steps=oldpin,new_steps=step_pins[0],rows=[dict(rank=x['rank'],mse=x['mse']) for x in new0],exact_mse_equal=True))
once(D/'controller_status.json',dict(state='completed',completed_updates=8,execution_owner='root_manual_serial_tool_sessions',segments=guardpins,protocol=pp))
once(D/'training_complete.json',dict(completed=True,updates=8,selected_endpoint=8,protocol=pp,endpoint=last['endpoint'],dials=last['dials'],segments=guardpins,completions=completionpins,steps=step_pins,initial_forward_check=pin(D/'initial_forward_check.json'),validation_read=False,executed_network_examples=32,training_pool_size=160,execution_owner='root_manual_serial_tool_sessions',finalizer=pin(__file__)))
print(json.dumps(dict(training_complete=pin(D/'training_complete.json'),initial_forward_check=pin(D/'initial_forward_check.json'))))
