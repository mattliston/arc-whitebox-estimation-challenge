"""Combine native293 with seven fresh boundary sessions on identical science48c."""
from pathlib import Path
import json,math,statistics
D=Path(__file__).resolve().parent;prior=D.parent/'peel_fullstack8_v1'
first=json.loads((prior/'RESULT_v1.json').read_text());native=next(r for r in first['rows']if r['rank']==293)
boundary=json.loads((D/'SCIENCE7_RESULT_v1.json').read_text());assert boundary['all_complete']
assert first['prereg']['sha256']==boundary['source_sha256']=='48c126c61f8d4f571768529d9d6610d8e732de24107fce75d9422e3d417a9700'
rows=[dict(rank=293,execution='native',U=native['U'],control_U=native['control_U'],raw_pct=native['raw_pct'],bill_pct=native['bill_pct'],product_pct=native['product_pct'],peel=native['peel'],w3=native['w3'],cap=native['peel_cap'])]
for r in boundary['rows']:rows.append(dict(rank=r['rank'],execution='production_boundary',U=r['U'],control_U=r['control_U'],raw_pct=r['raw_pct'],bill_pct=r['bill_vs_cold_pct'],product_pct=r['product_vs_cold_pct'],peel=r['peel'],w3_calls=r['w3_calls'],cap=r['peel_budget']))
out=dict(source_sha256=boundary['source_sha256'],rows=rows,mean_raw_pct=statistics.mean(r['raw_pct']for r in rows),raw_SE_pp=statistics.stdev(r['raw_pct']for r in rows)/math.sqrt(len(rows)),mean_first_bill_pct=statistics.mean(r['bill_pct']for r in rows),mean_first_product_pct=statistics.mean(r['product_pct']for r in rows),pooled_U_pct=100*(sum(r['U']for r in rows)/sum(r['control_U']for r in rows)-1),lower_raw_count=sum(r['raw_pct']<0 for r in rows),scope='Eight fixed ranks, identical48c numerical source; one native and seven production-boundary executions, exact same offline terminal U definition. Descriptive first-bill summary only, not final runtime qualification. Native300 RSS failure retained separately; no failed predictions enter averages.')
(D/'SCIENCE8_COMBINED_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
