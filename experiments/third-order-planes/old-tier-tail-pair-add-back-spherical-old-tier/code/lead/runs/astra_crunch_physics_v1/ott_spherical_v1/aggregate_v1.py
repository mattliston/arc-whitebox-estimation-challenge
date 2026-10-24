from pathlib import Path
import json,math,statistics
D=Path(__file__).resolve().parent
rows=[]
for rank in(378,390,400,424):
 pair=[]
 for side in('control','candidate'):
  g=json.loads((D/f'guard_{rank}_{side}_v1.json').read_text());assert g['completed']and g['whole_group_retired']and g['exit_code']==0
  x=json.loads((D/f'res_{rank}_{side}_v1.json').read_text());pair.append(x)
 a,b=pair;op_a=sum(v['calls']for v in a['operations'].values());op_b=sum(v['calls']for v in b['operations'].values())
 rows.append(dict(rank=rank,parent_U=a['U'],candidate_U=b['U'],raw_pct=100*(b['U']/a['U']-1),parent_flops=a['flops'],candidate_flops=b['flops'],bill_pct=100*(b['flops']/a['flops']-1),adjusted_pct=100*(b['U']/a['U']*b['flops']/a['flops']-1),operation_calls_delta=op_b-op_a,parent_operation_calls=op_a,candidate_operation_calls=op_b,ott_shapes_equal=a['ott_shapes']==b['ott_shapes'],parent_ott_shapes=a['ott_shapes'],candidate_ott_shapes=b['ott_shapes'],formula_credit=a['sphere_formula_credit'],actual_flop_credit=a['flops']-b['flops'],source_parent=a['source_sha'],source_candidate=b['source_sha']))
def stats(key):
 v=[x[key]for x in rows];return dict(mean=statistics.mean(v),SE=statistics.stdev(v)/math.sqrt(len(v)),wins=sum(x<0 for x in v))
A=sum(x['parent_U']for x in rows);B=sum(x['candidate_U']for x in rows)
out=dict(status='FIXED4_COMPLETE',rows=rows,raw_pct=stats('raw_pct'),bill_pct=stats('bill_pct'),adjusted_pct=stats('adjusted_pct'),pooled_raw_pct=100*(B/A-1),pooled_adjusted_pct=100*(sum(x['candidate_U']*x['candidate_flops']for x in rows)/sum(x['parent_U']*x['parent_flops']for x in rows)-1),mean_operation_calls_delta=statistics.mean(x['operation_calls_delta']for x in rows),models=8,guards='All8 completed exit0 and whole group retired before grading.',limitations='Four fresh rows establish direction only. Exact native cold bills and operation counts are scientific evidence; no production runtime qualification claimed. No tuning or coefficient/layer selection.')
(D/'NATIVE_FIXED4_RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
