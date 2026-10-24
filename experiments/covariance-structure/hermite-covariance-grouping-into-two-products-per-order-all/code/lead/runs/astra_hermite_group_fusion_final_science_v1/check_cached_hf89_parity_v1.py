"""Root-requested cached full16 parity report; no scoring gate or model calls."""
from pathlib import Path
import hashlib,json
import numpy as np
D=Path(__file__).resolve().parent;OLD=D.parent/'astra_hermite_group_fusion_science_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((D/'PILOT_RETIREMENT_astra_v1.json').read_text());old=json.loads((OLD/'PILOT_RETIREMENT_astra_v1.json').read_text())
assert r['all_actual_exits_observed']and r['all_owned_PID_PPID_PGID_absent']and old['all_actual_exits_observed']and old['all_owned_PID_PPID_PGID_absent']
rows=[]
for e in sorted(r['evaluations'],key=lambda x:x['seed']):
 b=next(x for x in old['evaluations']if x['seed']==e['seed']);paths=[];ys=[]
 for x in(e,b):
  p=Path(x['certificate']);p=p.with_name(p.stem+'_pred.npz');assert sha(p)==x['prediction_npz_sha256'];paths.append(dict(path=str(p),sha256=sha(p)))
  with np.load(p,allow_pickle=False)as z:ys.append(z[str(e['seed'])].copy())
 assert all(y.shape==(16,1024)and y.dtype==np.float32 and np.isfinite(y).all()for y in ys)
 rows.append(dict(seed=e['seed'],all16_byte_exact=ys[0].tobytes()==ys[1].tobytes(),max_abs_diff=float(np.max(np.abs(ys[0].astype(np.float64)-ys[1].astype(np.float64)))),predictions=paths))
out=dict(status='REPORT_ONLY_NO_GATE_CHANGE',all8_all16_byte_exact=all(x['all16_byte_exact']for x in rows),rows=rows,final_source_sha256=r['candidate_sha256'],previous_source_sha256=old['candidate_sha256'],retirement_sha256=sha(D/'PILOT_RETIREMENT_astra_v1.json'),previous_retirement_sha256=sha(OLD/'PILOT_RETIREMENT_astra_v1.json'),source_sha256=sha(__file__),model_calls=0)
p=D/'HF89_CACHED_PARITY_v1.json';p.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}));print(sha(p))
