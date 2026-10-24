import resource
import numpy as np
from common import *
from reference import covariance
seal_check();done('build');b=read(HERE/'build.json');verify(b['parameters'])
with np.load(b['parameters']['path'],allow_pickle=False) as z:q={k:z[k] for k in z.files}
rows=[]
for rec in read(HERE/'inputs.json')['legacy']:
    verify(rec['history']);verify(rec['data']);h=np.load(rec['history']['path'],allow_pickle=False)
    with np.load(rec['data']['path'],allow_pickle=False) as z:
        rho=z['rho'];sigma=z['sigma'];target=z['target'];w=z['weight'];response=z['response']
    assert h.shape==(7,1024,64) and all(x.dtype==np.float32 for x in (h,rho,sigma,target,w,response))
    pred=np.stack([covariance(h[l],rho[l],sigma[l],q) for l in range(7)])
    # Same archived FP32 source and next-response objective; no parameter change.
    value=np.stack([np.sum(w[l]*(pred[l]@w[l]),axis=0,dtype=np.float32)*response[l] for l in range(7)])
    target_value=np.stack([np.sum(w[l]*(target[l]@w[l]),axis=0,dtype=np.float32)*response[l] for l in range(7)])
    stats=dict(cov_mse=float(np.mean((pred.astype(np.float64)-target)**2)),cov_base=float(np.mean(target.astype(np.float64)**2)),
        value_mse=float(np.mean((value.astype(np.float64)-target_value)**2)),value_base=float(np.mean(target_value.astype(np.float64)**2)))
    stats['cov_ratio']=stats['cov_mse']/stats['cov_base'];stats['value_ratio']=stats['value_mse']/stats['value_base']
    errors={k:abs(v-rec['expected'][k])/rec['expected'][k] for k,v in stats.items()}
    assert max(errors.values())<=1e-4,(rec['rank'],stats,errors)
    rows.append(dict(rank=rec['rank'],stats=stats,old=rec['expected'],relative_errors=errors))
    del h,rho,sigma,target,w,response,pred,value,target_value
write(HERE/'legacy_replay.json',dict(passed=True,rows=rows,original_proxy_gate_still_failed=True,
    existing_8M_targets_used_for_archived_loss_replay=True,ordinary_1B_targets_read=False,new_fit_steps=0,
    loss_match_is_not_unique_parameter_identity_proof=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,max_relative=max(v for r in rows for v in r['relative_errors'].values())),flush=True)
