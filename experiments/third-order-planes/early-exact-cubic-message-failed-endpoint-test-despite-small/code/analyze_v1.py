import json,math,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[2];O=R.parents[2]
N=D/'native2_v1';Q=R/'astra_oct18_lead_v1/compact10_repair_qualification_v1'
C=Q/'science32/candidate32';T=R/'astra_oct18_accuracy_v1/post336935_training_audit_v1'
g=json.loads((N/'GUARD_v1.json').read_text());assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'];assert g['min_pressure_free_pct']>=50 and g['max_rss_bytes']<=7.5e9
rows=lambda p:{x['rank']:x for x in map(json.loads,(p/'rows.jsonl').read_text().splitlines())}
a,b,rt=rows(N),rows(C),rows(Q/'candidate_normal');grade=json.loads((N/'GRADE_STORE_v1.json').read_text());out=[]
for x in grade:
 r=x['rank'];n=np.fromfile(N/f'pred_{r}.bin',dtype='<f4').reshape(16,1024).astype('d');c=np.fromfile(C/f'pred_{r}.bin',dtype='<f4').reshape(16,1024).astype('d');s=np.asarray(json.loads((T/f'forward_{r}_v2/results/currentgap_{r}.json').read_text())['prediction'],dtype='d')
 x.update(requests_vs_same_sequence= a[r]['requests']-rt[r]['requests'],wall_s=a[r]['wall_s'],restores=a[r]['oct18_w3early_state'],rtt=a[r]['rtt_state'],layer_rms_vs_parent=np.sqrt(np.mean((n-c)**2,axis=1)).tolist(),layer_rms_vs_surrogate_before=np.sqrt(np.mean((c-s)**2,axis=1)).tolist(),layer_rms_vs_surrogate_after=np.sqrt(np.mean((n-s)**2,axis=1)).tolist(),nonfinite=a[r]['nonfinite'],rescue=a[r]['fast_rescue_stats'])
 out.append(x)
mean=lambda k:sum(x[k]for x in out)/len(out)
r=dict(status='FIXED_FIRST2_DIAGNOSTIC_NOT_BANKED',parent_sha256='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf',candidate_sha256=hashlib.sha256((D/'candidate_95c_w3early_HELD_v1.py').read_bytes()).hexdigest(),mean_raw_pct=mean('raw_pct'),mean_bill_pct=mean('bill_pct'),mean_adjusted_pct=mean('adjusted_pct'),projected_adjusted=1.4468565569762618e-9*(1+mean('adjusted_pct')/100),rows=out,guard={k:g[k]for k in ['completed','whole_group_retired','max_rss_bytes','min_pressure_free_pct']})
with(N/'RESULT_v1.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in r.items()if k!='rows'}))
for x in out:print(json.dumps({k:x[k]for k in ['rank','raw_pct','bill_pct','adjusted_pct','requests_vs_same_sequence','restores','layer_rms_vs_surrogate_before','layer_rms_vs_surrogate_after']}))
