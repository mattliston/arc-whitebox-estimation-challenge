from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent;H=D.parent
old=H/'hill_climb_pair_rr_inverse_v1_train2_v1';new=H/'rrpi_baseline_recheck_20260930T1605Z_v1_native'
def pin(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
g=json.loads((D/'guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
def rows(p):return {x['rank']:x for line in(p/'rows.jsonl').read_text().splitlines() if (x:=json.loads(line))}
a,b=rows(old),rows(new);out=[]
for r in (35,100):
 x=np.load(old/f'pred_{r}.npy');y=np.load(new/f'pred_{r}.npy')
 out.append(dict(rank=r,all16_bytes_exact=x.tobytes()==y.tobytes(),max_abs_delta=float(np.max(abs(x-y))),mse_old=a[r]['mse'],mse_new=b[r]['mse'],bill_old=a[r]['flops'],bill_new=b[r]['flops'],old=pin(old/f'pred_{r}.npy'),new=pin(new/f'pred_{r}.npy')))
pass_=all(r['all16_bytes_exact'] and r['bill_old']==r['bill_new'] and r['mse_old']==r['mse_new'] for r in out)
with(D/'result.json').open('x')as f:json.dump(dict(passed=pass_,rows=out,guard=pin(D/'guard.json'),purpose='Freshsame-sourcecontrol checkafterseveralsmallregressions; no candidate selection'),f,indent=2);f.write('\n')
print(json.dumps(dict(passed=pass_,rows=[{k:v for k,v in r.items() if k not in ('old','new')} for r in out])))
