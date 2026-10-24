from common import *
verify()
clean_watch('qualify')
sealed=read(HERE/'predictions_sealed.json')
assert sealed['completed'] and sealed['disabled_adapter_exact'] and sealed['before_target_decoding']
for p in sealed['pins']:
    check(p)
import numpy as np
assert np.__version__=='2.4.6'
meta=read(HERE/'inputs.json')
roles={k:read(p['path'])['records'][:2] for k,p in sealed['roles'].items()}
rows=[]
for index,rank in enumerate((35,100)):
    target_pin=meta['targets'][str(rank)]
    check(target_pin)
    with np.load(target_pin['path'],allow_pickle=False) as z:
        target=z['mc_post'].astype(np.float64)
    assert target.shape==(16,1024) and np.isfinite(target).all()
    row=dict(rank=rank,target=target_pin,roles={})
    for role in ('parent','trained'):
        r=roles[role][index]
        prediction=np.load(r['prediction']['path'],allow_pickle=False)
        assert prediction.shape==(16,1024) and prediction.dtype==np.float32
        mse=np.mean((prediction.astype(np.float64)-target)**2,axis=1)
        row['roles'][role]=dict(final_raw=float(mse[-1]),final_adjusted=float(mse[-1]*r['multiplier']),
            per_layer_raw=mse.tolist(),flops=r['flops'],multiplier=r['multiplier'],prediction=r['prediction'])
    row['adjusted_ratio']=row['roles']['trained']['final_adjusted']/row['roles']['parent']['final_adjusted']
    row['raw_ratio']=row['roles']['trained']['final_raw']/row['roles']['parent']['final_raw']
    rows.append(row)
means={role:{key:sum(r['roles'][role][key] for r in rows)/2
             for key in ('final_raw','final_adjusted','flops')} for role in ('parent','trained')}
ratio=means['trained']['final_adjusted']/means['parent']['final_adjusted']
passed=ratio<=.98 and all(r['adjusted_ratio']<1 for r in rows)
write(HERE/'score.json',dict(completed=True,passed=passed,adjusted_ratio=ratio,
      raw_ratio=means['trained']['final_raw']/means['parent']['final_raw'],rows=rows,means=means,
      reused_development=True,original_proxy_gate_passed=False,official_changed=False,goal_achieved=False,
      prediction_seal=pin(HERE/'predictions_sealed.json')))
print(dict(completed=True,passed=passed,adjusted_ratio=ratio,means=means),flush=True)
