import json,statistics
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[2];O=R.parents[1]
N=D/'native_control_split_fixed_v1'
Q=R/'astra_oct18_lead_v1/compact10_repair_qualification_v1'
assert json.loads((N/'COMPLETE_v1.json').read_text())['all_groups_retired']
rows=lambda p:{r['rank']:r for r in map(json.loads,(p/'rows.jsonl').read_text().splitlines())if'rank'in r}
def prediction(directory,rank):return np.fromfile(directory/f'pred_{rank}.bin',dtype='<f4').reshape(16,1024).astype('d')
def grade(candidate,control,rank):
 a=rows(candidate)[rank];b=rows(control)[rank]
 p=prediction(candidate,rank);c=prediction(control,rank)
 A=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{rank:04d}_A32M.npz')['mc_post'].astype('d')
 B=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{rank:04d}_B32M.npz')['mc_post'].astype('d')
 u=np.mean((p-A)*(p-B),axis=1);v=np.mean((c-A)*(c-B),axis=1)
 raw=float(u[-1]/v[-1]);bill=a['flops']/b['flops']
 return dict(rank=rank,raw_pct=100*(raw-1),bill_pct=100*(bill-1),adjusted_pct=100*(raw*bill-1),extra_flops=a['flops']-b['flops'],requests=a['requests'],control_requests=b['requests'],requests_delta=a['requests']-b['requests'],
    wall_s=a['wall_s'],control_wall_s=b['wall_s'],residual_instructions=a['residual_instructions'],control_residual_instructions=b['residual_instructions'],
    nonfinite=a['nonfinite'],rescue=a.get('fast_rescue_stats'),covsplit_state=a.get('oct18_covsplit_state'),
    prediction_rms=float(np.sqrt(np.mean((p-c)**2))),final_prediction_rms=float(np.sqrt(np.mean((p[-1]-c[-1])**2))),layer_raw_pct=(100*(u/v-1)).tolist())
records=[]
guards={}
for name in('control200','candidate35','candidate200'):
 g=json.loads((N/name/'GUARD_v1.json').read_text());assert g['completed']and g['whole_group_retired']and not g['remaining_rows']
 guards[name]={k:g.get(k)for k in('completed','whole_group_retired','max_rss_bytes','max_footprint_bytes','min_pressure_free_pct')}
for rank in(35,200):records.append(grade(N/f'candidate{rank}',Q/'science32/candidate32'if rank==35 else N/'control200',rank))
mean={k:statistics.mean(r[k]for r in records)for k in('raw_pct','bill_pct','adjusted_pct','requests_delta')}
cold=rows(N/'control200')[200];warm=rows(Q/'candidate_normal')[200]
control_receipt=dict(source_sha256='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf',directory=str(N/'control200'),rank=200,
    cold_requests=cold['requests'],warm_requests=warm['requests'],extra_cold_requests=cold['requests']-warm['requests'],cold_flops=cold['flops'],warm_flops=warm['flops'],cold_bill_tax_pct=100*(cold['flops']/warm['flops']-1),
    prediction_byte_equal_to_warm=(N/'control200/pred_200.bin').read_bytes()==(Q/'candidate_normal/pred_200.bin').read_bytes(),guard=guards['control200'])
with(N/'COLD200_CONTROL_v1.json').open('x')as f:json.dump(control_receipt,f,indent=2);f.write('\n')
result=dict(status='FIXED_TWO_NETWORK_FIRST_PRODUCT_SPLIT_SCREEN',candidate_sha256='84e46bfbe4afd81387eaa432ade5fa82563ce5bf26dbf707b9927cfb395195be',baseline_submission=336935,baseline_adjusted=1.4468565569762618e-9,rows=records,mean=mean,conditional_projected_adjusted=1.4468565569762618e-9*(1+mean['adjusted_pct']/100),guards=guards,controls=control_receipt,
    caveats=['Rank200 uses a newly measured matching fresh control. Rank35 uses existing current95c cold science control.','Two fixed networks do not validate population gain or runtime eligibility.','Current kernel geometry is unchanged by source policy; floating-point ordering changes and efficacy is measured.'])
with(N/'RESULT_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(mean=mean,projection=result['conditional_projected_adjusted'],controls=control_receipt,guards=guards)))
for row in records:print(json.dumps({k:row[k]for k in('rank','raw_pct','bill_pct','adjusted_pct','extra_flops','requests_delta','wall_s','covsplit_state')}))
# Re-price the already-finished symmetric margin component against matchingcold200.
P=D.parent/'fusion_strip5_sym_v1/native_fixed2_v1'
sym=[grade(P/f'rank{rank}',Q/'science32/candidate32'if rank==35 else N/'control200',rank)for rank in(35,200)]
rec=dict(status='MATCHED_COLD_RECALIBRATION_EXISTING_COMPONENT',candidate_sha256='a41895cf5d6186e299a30cec33b8792f3610aa96aef5d90f924d15acf1117851',rows=sym,mean={k:statistics.mean(r[k]for r in sym)for k in('raw_pct','bill_pct','adjusted_pct','requests_delta')},caveat='No new componentmodelrun. Matchingcold200control removes earlier cold-versus-warm bill/request bias; hostedcushion remains unqualified.')
with(N/'SYMSTRIP_MATCHED_CONTROL_v1.json').open('x')as f:json.dump(rec,f,indent=2);f.write('\n')
print(json.dumps(dict(symstrip_matched_mean=rec['mean'])))
