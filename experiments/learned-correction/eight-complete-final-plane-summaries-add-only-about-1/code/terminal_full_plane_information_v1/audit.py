import math,resource
import numpy as np
from common import *
seal_check();done('oracle');seal_check('features_sealed.json')
result=read(HERE/'result.json');check(result['payload']);features=read(HERE/'features_result.json')['rows'][:2]
inputs=read(HERE/'inputs.json')['rows'];checks=[];all_pass=[]
with np.load(result['payload']['path'],allow_pickle=False) as payload:
    for meta,record,row in zip(inputs,features,result['rows']):
        rank=row['rank'];assert rank==meta['rank']==record['rank']
        check(meta['truth']);check(record['file'])
        with np.load(record['file']['path'],allow_pickle=False) as z:X=z['X'];Z=z['Z'];base=z['prediction'][-1].astype(np.float64)
        with np.load(meta['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        y=truth-base
        def mean_square(x):return math.fsum(float(v)*float(v) for v in x)/len(x)
        parent=mean_square(y);assert abs(parent-row['parent_raw'])<=1e-14*parent
        audited={}
        for name,M in [('affine',X),('extended',np.column_stack([X,Z]))]:
            c=row['arms'][name];assert c['rank']==M.shape[1],('audit needs recorded full rank',rank,name,c['rank'])
            scale=np.array([math.sqrt(math.fsum(float(v)*float(v) for v in M[:,j])/len(M)) for j in range(M.shape[1])])
            Q,R=np.linalg.qr(M/scale,mode='reduced');coef=np.linalg.solve(R,Q.T@y)/scale
            reference=payload[f'{rank}_{name}_coefficients'];coef_err=float(np.linalg.norm(coef-reference)/np.linalg.norm(reference));assert coef_err<=1e-6
            ideal=base+M@coef;served=np.maximum(ideal,0).astype(np.float32)
            prediction_err=float(np.linalg.norm(ideal-payload[f'{rank}_{name}_ideal'])/np.linalg.norm(ideal));assert prediction_err<=1e-10
            assert served.tobytes()==payload[f'{rank}_{name}_served'].tobytes()
            li=mean_square(ideal-truth);ls=mean_square(served.astype(float)-truth)
            assert abs(li-c['ideal_raw'])<=1e-10*li and abs(ls-c['served_raw'])<=1e-14*ls
            audited[name]=dict(ideal_raw=li,served_raw=ls)
            checks.append(dict(rank=rank,arm=name,coefficient_relative=coef_err,prediction_relative=prediction_err,served_bytes_exact=True))
        passed=all(audited['extended'][k+'_raw']/parent<=.85 and audited['extended'][k+'_raw']/audited['affine'][k+'_raw']<=.90 for k in ['ideal','served'])
        assert passed==row['passed'];all_pass.append(passed)
assert all(all_pass)==result['passed']
write(HERE/'audit_result.json',dict(passed=True,gate_decisions_reproduced=True,full_rank_QR_replay=True,checks=checks,
    all_served_bytes_exact=True,score=pin(HERE/'result.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,max_coefficient_relative=max(r['coefficient_relative'] for r in checks)),flush=True)
