"""Seven remaining frozen science ranks through production boundary, one at a time."""
from pathlib import Path
import hashlib,json,os,statistics,subprocess,sys,time
D=Path(__file__).resolve().parent;R=D.parents[1]
source=R/'astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py'
sha='48c126c61f8d4f571768529d9d6610d8e732de24107fce75d9422e3d417a9700'
assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
identity=D/'FROZEN_SOURCE_IDENTITY_v1.json';identity.write_text(json.dumps(dict(status='EXACT_FROZEN_NUMERICAL_SOURCE',source=str(source),sha256=sha,transformation=None,authorization='Root instructed remaining fixed7 boundary accuracy on unchanged48c; final runtime gate remains separate.'),indent=2)+'\n')
ranks=[300,310,312,316,320,328,333];rows=[];start=time.monotonic();light=Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
for rank in ranks:
 tag=f'science_{rank}_v1';out=D/tag;assert not out.exists()
 print(json.dumps(dict(event='START',rank=rank)),flush=True)
 with(D/f'science_{rank}_launch_v1.log').open('x')as log:
  code=subprocess.call([str(light),'-B',str(D/'run_session_v1.py'),str(source),sha,str(rank),tag,'--parity-receipt',str(identity),'--clock','short'],stdout=log,stderr=subprocess.STDOUT)
 result=json.loads((out/'SESSION_RESULT_v1.json').read_text())if(out/'SESSION_RESULT_v1.json').exists()else{}
 g=result.get('guard',{});scored=result.get('accuracy',{}).get('rows',[])
 row=dict(rank=rank,guard_pass=result.get('guard_pass',False),exit_code=code,grade_exit_code=result.get('grade_exit_code'),max_rss=g.get('max_rss_bytes'),max_footprint=g.get('max_footprint_bytes'),seconds=g.get('elapsed_s'),error=g.get('error'),session=str(out))
 if scored:row.update(scored[0])
 rows.append(row);(D/'SCIENCE7_PARTIAL_v1.json').write_text(json.dumps(dict(source_sha256=sha,ranks=ranks,rows=rows),indent=2)+'\n');print(json.dumps(dict(event='DONE',**row)),flush=True)
 if code!=0 or not row['guard_pass']or row['grade_exit_code']!=0:print(json.dumps(dict(event='STOP',rank=rank)),flush=True);break
scored=[r for r in rows if r.get('status')=='SCORED'];out=dict(source_sha256=sha,ranks=ranks,rows=rows,all_complete=len(rows)==7 and all(r['guard_pass']and r.get('status')=='SCORED'for r in rows),seconds=time.monotonic()-start,scope='Frozen science48c; fresh standard-short-setup production boundary sessions. Preserves failed native300 receipt. This is not final-runtime-source qualification.')
if scored:out.update(mean_raw_pct=statistics.mean(r['raw_pct']for r in scored),mean_first_bill_pct=statistics.mean(r['bill_vs_cold_pct']for r in scored),mean_product_pct=statistics.mean(r['product_vs_cold_pct']for r in scored))
(D/'SCIENCE7_RESULT_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(event='BATCH_COMPLETE',**{k:v for k,v in out.items()if k!='rows'})),flush=True)
