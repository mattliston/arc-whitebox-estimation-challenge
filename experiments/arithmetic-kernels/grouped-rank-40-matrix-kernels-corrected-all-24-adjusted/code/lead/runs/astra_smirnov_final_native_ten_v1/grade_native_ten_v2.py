"""Same frozen dual adjusted signs, after actual owned terminal/guard barriers."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();phase='parity'
p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());owners=[];evaluations=[];ids=set();pins={};terminals=[]
# Root-authorized exact source-derived metadata correction; original launch protocol remains immutable.
costp=D.parent/'astra_smirnov_current24_virtual_v1/FINAL24_BILL_TRANSFER_CORRECTED_v16.json'
assert sha(costp)=='cc981cb4d7eedf88b05915af22da2f5a252473da6d547f96c12b4a336bb65eab'
correction=json.loads(costp.read_text());assert correction['candidate_sha256']==c['candidates']['ON']['sha256']
assert correction['aggregate']['fixed_deferred_first_read_flops']==276903104
for item in correction['source_metadata']+[correction['previous_receipt'],correction['derivation']]:
 assert sha(item['path'])==item['sha256'];pins[item['path']]=item['sha256']
route=D/'FIXED_LAYER1_R7_ROUTE_v1.json';assert sha(route)=='b3929df61bcfeed3a4c28c1380df4019c391dc90704f6fc01fb3a7aec07baa6b'
assert json.loads(route.read_text())['all24_first_read_is_fixed_full1024x2048x1024']
pins[str(route)]=sha(route);pins[str(Path(__file__).resolve())]=sha(__file__)
for old in c['final_cost_rows']:
 new=next(r for r in correction['rows']if r['seed']==old['seed'])
 assert new['science_flops']==old['science_flops'] and new['final_flops']==old['final_flops']+276903104
c['final_cost_rows']=correction['rows'];c['final_cost_metadata']=dict(path=str(costp),sha256=sha(costp))

for i in range(2):
 h=D/f'run_final_{phase}_s{i}_v1';r=json.loads((h/'READY_v1.json').read_text());t=json.loads((h/'ACTUAL_MANAGED_EXIT_v1.json').read_text());m=json.loads((h/'MANAGED_SESSION_v1.json').read_text())
 assert r['all_guards_retired'] is True and r['protocol_sha256']==sha(p)
 assert t['actual_exit_observed'] and t['exit_code']==0 and t['terminal_session_present'] is False and t['managed_session_id']==m['managed_session_id']
 assert m['controller_pid']==r['controller_pid'];ids.add(r['controller_pid']);terminals.append(t)
 es=r['evaluations'];assert [e['seed']for e in es]==c['phase_seeds'][phase][i::2]
 own=[json.loads(z)for z in (h/'ownership.jsonl').read_text().splitlines()];assert len(own)==len(es)
 for a,e in zip(own,es):
  assert a['controller_pid']==r['controller_pid'] and a['seed']==e['seed'];ids.add(a['owned_guard_pid']);g=json.loads(Path(e['guard']).read_text());ids.add(g['pid'])
  assert g['completed'] is True and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] is True and not g['remaining_rows']
  assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
 evaluations.extend(es)
 owners.append(dict(shard=i,controller=r['controller_pid'],ready_sha256=sha(h/'READY_v1.json')))
 for n in ('READY_v1.json','ACTUAL_MANAGED_EXIT_v1.json','MANAGED_SESSION_v1.json','PLAN_v1.json','ownership.jsonl'):pins[str(h/n)]=sha(h/n)
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(row)&ids for row in live)
assert sorted(x['seed']for x in evaluations)==c['phase_seeds'][phase]
for e in evaluations:
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
 p0=Path(e['certificate']);pp=p0.with_name(p0.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];pins[str(pp)]=e['prediction_npz_sha256']
ret=dict(approved=True,reviewed_by='root',executed_by='fresh_algebra',authority='Root delegated fixed ten exact helper transfer and exact metadata24 cost repricing, two slots; actual exit and whole-group retirement before grade.',candidate_sha256=c['candidates']['ON']['sha256'],protocol_sha256=sha(p),all_actual_exits_observed=True,actual_terminals=terminals,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,owned_ids=sorted(ids),owners=owners,evaluations=evaluations,evidence_sha256=pins)
retp=D/(phase.upper()+'_RETIREMENT_astra_v1.json')
original_ret=json.loads(retp.read_text())
assert original_ret['candidate_sha256']==ret['candidate_sha256'] and original_ret['owned_ids']==ret['owned_ids'] and original_ret['evaluations']==ret['evaluations']
assert original_ret['all_actual_exits_observed'] and original_ret['all_owned_PID_PPID_PGID_absent'] and original_ret['all_guards_clean']

assert len(evaluations)==10
rows=[]
def cert(rec):
 for key in ('certificate','guard'):assert sha(rec[key])==rec[key+'_sha256']
 z=Path(rec['certificate']);pp=z.with_name(z.stem+'_pred.npz');assert sha(pp)==rec['prediction_npz_sha256']
 return json.loads(z.read_text()),pp
for e in sorted(evaluations,key=lambda x:x['seed']):
 control=next(x for x in c['stack_control_reuse']if x['seed']==e['seed'])
 a,apath=cert(e);b,bpath=cert(control)
 with np.load(apath,allow_pickle=False)as z:ap=z[str(e['seed'])].copy()
 with np.load(bpath,allow_pickle=False)as z:bp=z[str(e['seed'])].copy()
 assert a['sha256']==c['candidates']['ON']['sha256'] and b['sha256']==c['candidates']['stack']['sha256']
 assert ap.shape==bp.shape==(16,1024) and np.isfinite(ap).all() and ap.tobytes()==bp.tobytes()
 ar,br=a['rows'][0],b['rows'][0]
 assert ar['mse_layers']==br['mse_layers']
 assert ar['weights_before_sha256']==ar['weights_after_sha256']==br['weights_before_sha256']
 assert a['setup']['pb_single_sha256']==b['setup']['pb_single_sha256']
 price=next(x for x in c['final_cost_rows']if x['seed']==e['seed'])
 assert ar['flops']==price['final_flops'] and br['flops']==price['science_flops']
 assert ar['flops']-br['flops']==price['delta_flops']
 rows.append(dict(seed=e['seed'],all16_byte_identical=True,all_layerMSE_equal=True,final_flops=ar['flops'],science_flops=br['flops'],delta_flops=ar['flops']-br['flops'],exact_metadata_bill_verified=True,charged_ops_delta=ar['ops']-br['ops'],candidate=e,reference=control))
repriced=[]
for e in sorted(c['stack_control_reuse'],key=lambda x:x['seed']):
 a,_=cert(e);ar=a['rows'][0];price=next(x for x in c['final_cost_rows']if x['seed']==e['seed'])
 assert a['sha256']==c['candidates']['stack']['sha256'] and ar['flops']==price['science_flops']
 row=dict(seed=e['seed'],science_flops=ar['flops'],final_flops=price['final_flops'],comparisons={})
 for role,key,sourcekey in [('PRIMARY','primary_control_reuse','primary'),('RPE','reference_control_reuse','reference')]:
  control=next(x for x in c[key]if x['seed']==e['seed']);b,_=cert(control);br=b['rows'][0]
  assert b['sha256']==c['candidates'][sourcekey]['sha256']
  mse=100*math.log(ar['mse_final']/br['mse_final'])
  bill=100*math.log(max(price['final_flops']/2**41,.1)/max(br['flops']/2**41,.1))
  row['comparisons'][role]=dict(MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill)
 repriced.append(row)
assert [r['seed']for r in repriced]==list(range(7103248,7103272))
def summary(rs):return {role:{k:statistics.mean(r['comparisons'][role][k]for r in rs)for k in ('MSE_log_pct','bill_log_pct','net_log_pct')}for role in ('PRIMARY','RPE')}
means=summary(repriced);oos=summary([r for r in repriced if r['seed']in c['identity_groups']['oos']])
assert all(z[role]['net_log_pct']<0 for z in (means,oos)for role in ('PRIMARY','RPE'))
for path,digest in c['source_pins'].items():assert sha(path)==digest
out=dict(passed=True,approved=True,reviewed_by='root',executed_by='fresh_algebra',purpose='smirnov_final_exact_native_ten_and_cost_transfer',candidate_sha256=c['candidates']['ON']['sha256'],reference_sha256=c['candidates']['stack']['sha256'],all_actual_retired=True,all_16_prediction_bytes_equal=True,all_layerMSE_equal=True,cost_transfer_accepted=True,actual_ten_exact_metadata_bills_verified=True,all24_and_OOS_both_adjusted_negative=True,n=10,rows=rows,repriced24=repriced,repriced_means=means,repriced_OOS8=oos,retirement_path=str(retp),retirement_sha256=sha(retp),protocol_sha256=sha(p),evidence_sha256={**pins,str(p):sha(p),str(retp):sha(retp),c['final_cost_metadata']['path']:c['final_cost_metadata']['sha256']})
result=D/'NATIVE_TEN_PARITY_v1.json';result.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ('rows','repriced24','evidence_sha256')}));print('decision',sha(result),'retirement',sha(retp))
