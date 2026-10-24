"""Offline terminal U from saved production predictions and the same A/B32M truths."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[1];O=R.parents[1]
session=Path(sys.argv[1]).resolve();manifest=json.loads((session/'manifest.json').read_text())
rows=[json.loads(line)for line in (session/'rows.jsonl').read_text().splitlines()];out=[]
for row in rows:
 rank=row['rank'];p=session/f'pred_{rank}.bin';dtype=np.dtype('float32'if'32'in row['dtype']else'float64')
 values=np.fromfile(p,dtype=dtype);assert values.size==16384 and np.all(np.isfinite(values))
 y=values.reshape(16,1024)[-1].astype('f8')
 A=O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_A32M.npz';B=O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_B32M.npz'
 if not(A.exists()and B.exists()):out.append(dict(rank=rank,status='NO_AB32M_TRUTH',prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));continue
 a=np.load(A)['mc_post'][15].astype('f8');b=np.load(B)['mc_post'][15].astype('f8');U=float(np.mean((y-a)*(y-b)))
 control=R/f'opus_kinkoracle_v1/res_ctl_{rank}.json';c=json.loads(control.read_text())
 record=dict(rank=rank,status='SCORED',U=U,control_U=c['U'],raw_pct=100*(U/c['U']-1),flops=row['flops'],control_cold_flops=c['flops'],bill_vs_cold_pct=100*(row['flops']/c['flops']-1),product_vs_cold_pct=100*(U/c['U']*row['flops']/c['flops']-1),prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),dtype=str(dtype),peel=row.get('peel_delta'),peel_budget=row.get('peel_budget'),w3_calls=row.get('w3_delta'),control=str(control))
 out.append(record)
report=dict(source=manifest['candidate'],source_sha256=manifest['sha256'],rows=out,scope='Offline U=mean((final_prediction-A32M)*(final_prediction-B32M)); same statistic and truths as native judge. Bill comparison is to cached cold native control, so later warm rows require separate matched runtime-control comparison.')
(session/'ACCURACY_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
