import math
import numpy as np
from common import *
verify();assert read(HERE/'unit_native_result.json')['passed']
disabled=H2/'terminal_pair_disabled35_v1'
with (disabled/'rows.jsonl').open() as f:dr=[json.loads(x) for x in f]
base=[json.loads(x) for x in (H2/'both_tail_train2_v6/rows.jsonl').read_text().splitlines()]
assert np.array_equal(np.load(disabled/'pred_35.npy'),np.load(H2/'both_tail_train2_v6/pred_35.npy')) and dr[0]['flops']==base[0]['flops']
rows=[];cand=[json.loads(x) for x in (H2/'terminal_pair_train2_v1/rows.jsonl').read_text().splitlines()]
with np.load(HERE/'predictions.npz') as offline:
    for r,b in zip(cand,base):
        rank=r['rank'];assert rank==b['rank']
        p=np.load(H2/'terminal_pair_train2_v1'/f'pred_{rank}.npy');parent=np.load(H2/'both_tail_train2_v6'/f'pred_{rank}.npy')
        with np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as z:truth=z['mc_post'][15].astype(np.float64)
        mse=math.fsum((float(x)-float(y))**2 for x,y in zip(p[-1],truth))/len(truth)
        adj=mse*max(.1,r['flops']/2**41);assert abs(mse-r['mse'])<1e-18
        correction=np.sqrt(np.mean((offline[str(rank)]-parent[-1])**2));gap=np.sqrt(np.mean((p[-1]-offline[str(rank)])**2))
        rows.append(dict(rank=rank,mse=mse,flops=r['flops'],adjusted=adj,raw_ratio=mse/b['mse'],adjusted_ratio=adj/b['adjusted'],first15_bytes_exact=p[:15].tobytes()==parent[:15].tobytes(),offline_transfer_relative=float(gap/max(correction,1e-20))))
pooled=sum(r['adjusted'] for r in rows)/sum(r['adjusted'] for r in base)
gates=dict(first15=all(r['first15_bytes_exact'] for r in rows),fidelity=all(r['offline_transfer_relative']<=.02 for r in rows),both_wins=all(r['adjusted_ratio']<1 for r in rows),pooled=pooled<=.98)
once(HERE/'native_result.json',dict(passed=all(gates.values()),gates=gates,rows=rows,pooled_adjusted_ratio=pooled,disabled35_all16_bill_exact=True,model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate.py')))
print(json.dumps(dict(passed=all(gates.values()),pooled_adjusted_ratio=pooled,rows=rows)),flush=True)
