"""Fixed coherent RTP4 diagnostic: current native cost, same-target error, and causal onset."""
import hashlib,json,math
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;T=D.parent
B=json.loads((T/'BINDING_v1.json').read_text());C=Path(B['native_reference'])
O=T.parents[3];anchor=1.4468565569762618e-9
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=lambda p:{x['rank']:x for x in map(json.loads,(p/'rows.jsonl').read_text().splitlines())}
base=rows(C);out=[]
for spec in B['data']:
 r=spec['rank'];N=D/f'native_{r}_v1';g=json.loads((N/'GUARD_v1.json').read_text());nr=rows(N)[r]
 assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
 assert g['max_rss_bytes']<=7.5e9 and g['min_pressure_free_pct']>=50
 assert nr['nonfinite']==0 and not nr['fast_rescue_stats']['calls']
 st=nr['rtp4_state'];assert st['retired'] and len(st['reads']) > 0
 n=np.fromfile(N/f'pred_{r}.bin',dtype='<f4').reshape(16,1024).astype('d')
 c=np.fromfile(C/f'pred_{r}.bin',dtype='<f4').reshape(16,1024).astype('d')
 s=np.asarray(json.loads((T/f'forward_{r}_v2/results/currentgap_{r}.json').read_text())['prediction'],dtype='d')
 t=np.load(spec['truth'],allow_pickle=False)['mc_post'].astype('d')
 nm=np.mean((n-t)**2,axis=1);cm=np.mean((c-t)**2,axis=1)
 a=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{r:04d}_A32M.npz',allow_pickle=False)['mc_post'].astype('d')
 b=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{r:04d}_B32M.npz',allow_pickle=False)['mc_post'].astype('d')
 un=np.mean((n-a)*(n-b),axis=1);uc=np.mean((c-a)*(c-b),axis=1)
 bill=nr['flops']/base[r]['flops'];raw=un[-1]/uc[-1];adj=raw*bill
 rec=dict(rank=r,truth688_raw_pct=100*(nm[-1]/cm[-1]-1),crosshalf_raw_pct=100*(raw-1),bill_pct=100*(bill-1),adjusted_pct=100*(adj-1),projected_adjusted=anchor*adj,conditional_samebill_score=anchor*raw,layer_truth688_raw_pct=(100*(nm/cm-1)).tolist(),layer_crosshalf_raw_pct=(100*(un/uc-1)).tolist(),layer_discrepancy_vs_surrogate_before=np.sqrt(np.mean((c-s)**2,axis=1)).tolist(),layer_discrepancy_vs_surrogate_after=np.sqrt(np.mean((n-s)**2,axis=1)).tolist(),layer_change_rms=np.sqrt(np.mean((n-c)**2,axis=1)).tolist(),restoration=st,actual_requests=nr['requests'],reference_requests=base[r]['requests'],cold_reference_matching=(r==35),bill_comparability='Both candidate ranks cold; reference35cold and200warm. Any cold-tax in200bill is conservative for efficacy and does not establish runtime.',flops=nr['flops'],reference_flops=base[r]['flops'],guard={k:g[k] for k in ['max_rss_bytes','min_pressure_free_pct','completed','whole_group_retired']},prediction_sha=sha(N/f'pred_{r}.bin'))
 out.append(rec)
geo=lambda k:math.exp(sum(math.log(1+x[k]/100)for x in out)/len(out))
rr=geo('crosshalf_raw_pct');bb=geo('bill_pct');aa=rr*bb
receipt=dict(candidate_sha=sha(D/'candidate_95c_rtp4_HELD_v2.py'),anchor=anchor,rows=out,geometric_raw_pct=100*(rr-1),geometric_bill_pct=100*(bb-1),geometric_adjusted_pct=100*(aa-1),projected_adjusted=anchor*aa,truth688_geometric_raw_pct=100*(geo('truth688_raw_pct')-1),scientific_scope='FixedRTP4coherentdiagnostic2; no gainbanked or heldoutconfirmation.')
with(D/'RESULT_v1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items()if k!='rows'}))
for x in out:print(json.dumps({k:x[k] for k in ['rank','truth688_raw_pct','crosshalf_raw_pct','bill_pct','adjusted_pct','projected_adjusted','restoration','guard']}))
