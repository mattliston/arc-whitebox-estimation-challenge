from pathlib import Path
import hashlib,json,math,statistics,subprocess
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((D/'PROTOCOL_REGISTERED_astra_v1.json').read_text())
owners=[];evaluations=[];ids=set();pins={}
for i in range(4):
    h=D/f'run_pz1_pilot_s{i}_v1';r=json.loads((h/'READY_v1.json').read_text());assert r['all_guards_retired'] is True
    assert r['protocol_sha256']==sha(D/'PROTOCOL_REGISTERED_astra_v1.json')
    ids.add(r['controller_pid']);evaluations.extend(r['evaluations'])
    own=[json.loads(z)for z in (h/'ownership.jsonl').read_text().splitlines()]
    assert len(own)==2 and len(r['evaluations'])==2
    for a,e in zip(own,r['evaluations']):
        assert a['controller_pid']==r['controller_pid'];ids.add(a['owned_guard_pid'])
        g=json.loads(Path(e['guard']).read_text());ids.add(g['pid'])
        assert g['completed'] is True and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] is True and not g['remaining_rows']
    owners.append(dict(shard=i,controller=r['controller_pid'],ready_sha256=sha(h/'READY_v1.json')))
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(row)&ids for row in live)
assert sorted(x['seed']for x in evaluations)==c['phase_seeds']['pilot']
ret=dict(approved=True,reviewed_by='root',all_actual_exits_observed=True,actual_exit_codes=[0]*4,sessions=[17978,86342,17897,77974],tool_chunks=json.loads((D/'ACTUAL_PILOT_EXITS_astra_v1.json').read_text()),all_owned_absent=True,owned_ids=sorted(ids),owners=owners)
(D/'PILOT_RETIREMENT_astra_v1.json').open('x').write(json.dumps(ret,indent=2)+'\n')
def read(e):
    for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256']
    p=Path(e['certificate']);pp=p.with_name(p.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256']
    a=json.loads(p.read_text());return a,a['rows'][0],np.load(pp)[str(e['seed'])]
rows=[]
for e in sorted(evaluations,key=lambda z:z['seed']):
    a,ar,ap=read(e);assert a['sha256']==c['candidates']['ON']['sha256']
    row=dict(seed=e['seed'],comparisons={})
    for role,key in [('QR','stack_control_reuse'),('e406','reference_control_reuse')]:
        control=next(x for x in c[key]if x['seed']==e['seed']);b,br,bp=read(control)
        assert a['setup']['pb_single_sha256']==b['setup']['pb_single_sha256']
        assert a['setup']['pb_single_count']==b['setup']['pb_single_count']
        assert ap.shape==bp.shape==(16,1024) and np.isfinite(ap).all() and ap[:1].tobytes()==bp[:1].tobytes()
        mse=100*math.log(ar['mse_final']/br['mse_final']);bill=100*math.log(max(ar['flops']/2**41,.1)/max(br['flops']/2**41,.1))
        row['comparisons'][role]=dict(MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill,ops_delta=ar['ops']-br['ops'])
    rows.append(row)
means={role:{k:statistics.mean(r['comparisons'][role][k]for r in rows)for k in ('MSE_log_pct','bill_log_pct','net_log_pct','ops_delta')}for role in ('QR','e406')}
ses={role:statistics.stdev(r['comparisons'][role]['net_log_pct']for r in rows)/8**.5 for role in means}
passed=all(means[role]['net_log_pct']<0 for role in means)
out=dict(passed=passed,all_actual_retired=True,source=c['candidates']['ON'],n=8,means=means,SE_pct=ses,rows=rows,all_implementation_pass=True,remaining16_authorized=passed)
p=D/'PILOT_DECISION_astra_v1.json';p.open('x').write(json.dumps(out,indent=2)+'\n')
now=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
with (O/'TEAM_LOG.md').open('a')as f:f.write('\n[astra] '+now+' Fixedprefixz1 4aa5 atopQR pilot8 '+('PASS'if passed else'FAIL')+': '+json.dumps(means)+'. All4actualexit0/all20ownedabsent; retirement'+sha(D/'PILOT_RETIREMENT_astra_v1.json')[:8]+', decision'+sha(p)[:8]+'. '+('16ONcontinuation authorized.'if passed else'No remaining/delivery.')+' GPUpaused.\n')
print(json.dumps({k:v for k,v in out.items()if k!='rows'}));print('decision',sha(p))
