"""opus-release: b709 captured-setup native/production parity (2b108 grader logic; exits from launch_native35_100_replays_v1)."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def put(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
rows=[];evidence={};ids=set();groups=set();source=None;ctl=[]
for mode in ('normal','slow'):
 Z=D/('native35_100_replay_'+mode+'_v1');N=Z/'native';S=D/'boundary/numerical_v1'/('final_'+mode)
 actual=read(Z/'ACTUAL_EXIT_v1.json');launch=read(Z/'LAUNCH_v1.json');owner=read(Z/'OWNER_v1.json');g=read(Z/'guard.json');p=read(Z/'PROTOCOL_v1.json')
 assert actual['actual_exit_observed']and actual['exit_code']==0 and actual['controller_pid']==launch['controller_pid']==owner['controller_pid']
 ctl.append(owner['controller_pid']);ids.update((owner['controller_pid'],owner['guard_pid'],g['pid']));groups.add(g['pid'])
 assert g['completed']and g['exit_code']==0 and g.get('error')is None and g['whole_group_retired']and g['remaining_rows']==[]
 assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
 for path,h in p['pins'].items():assert sha(path)==h
 manifest=read(N/'manifest.json');assert manifest['sha256']==p['candidate_sha256']==read(S/'manifest.json')['sha256'];source=manifest['sha256']
 assert manifest['setup']['captured_replay']['all_setup_counts_exact']
 a=[json.loads(x)for x in(N/'rows.jsonl').read_text().splitlines()];b=[json.loads(x)for x in(S/'rows.jsonl').read_text().splitlines()];assert [x['rank']for x in a]==[x['rank']for x in b]==[35,100]
 for x,y in zip(a,b):
  rank=x['rank'];native=np.load(N/f'pred_{rank}.npy');prod=np.frombuffer((S/f'pred_{rank}.bin').read_bytes(),dtype=np.float32).reshape(16,1024).astype(np.float64)
  same=native.shape==prod.shape==(16,1024)and np.isfinite(native).all()and native.tobytes()==prod.tobytes();bill=x['flops']==y['flops']
  rows.append(dict(mode=mode,rank=rank,all16_prediction_bytes_equal=bool(same),native_flops=x['flops'],production_flops=y['flops'],FLOPs_equal=bill));assert same and bill,rows[-1]
 for file in (Z/'ACTUAL_EXIT_v1.json',Z/'LAUNCH_v1.json',Z/'OWNER_v1.json',Z/'guard.json',Z/'PROTOCOL_v1.json',Z/'NATIVE_COMPLETE_v1.json',N/'manifest.json',N/'rows.jsonl',S/'manifest.json',S/'rows.jsonl',N/'pred_35.npy',N/'pred_100.npy',S/'pred_35.bin',S/'pred_100.bin'):evidence[str(file)]=sha(file)
 live=[tuple(map(int,x.split()))for x in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()];assert not[x for x in live if x[0]in ids or x[1]in ids or x[2]in groups]
ret=D/'NATIVE35_100_CAPTURED_SETUP_RETIREMENT_v1.json';put(ret,dict(all_actual_exits_observed=True,all_owned_PID_PPID_PGID_absent=True,owned_ids=sorted(ids),owned_groups=sorted(groups),controller_pids=ctl,exit_codes=[0,0],evidence_sha256=evidence));evidence[str(ret)]=sha(ret)
receipt=D/'NATIVE35_100_CAPTURED_SETUP_PARITY_v1.json';put(receipt,dict(passed=True,candidate_sha256=source,all16_prediction_bytes_equal=True,all_FLOPs_equal=True,normal_and_slow_actual_setup_states_exact=True,all_actual_retired=True,ranks=[35,100],modes=['normal','slow'],rows=rows,source_unchanged=True,retirement_path=str(ret),retirement_sha256=sha(ret),evidence_sha256=evidence));print(json.dumps(dict(passed=True,receipt=str(receipt),sha256=sha(receipt),rows=rows)))
