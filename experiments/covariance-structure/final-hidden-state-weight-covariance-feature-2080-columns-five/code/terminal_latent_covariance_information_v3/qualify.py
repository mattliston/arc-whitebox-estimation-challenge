"""Meter new features on already qualified, byte-identical native captures."""
import json,resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,D,sha,read,seal_check,write
from feature import latent_covariance
seal_check();assert all(x['num_threads']==1 for x in threadpool_info())
for name in ['unit_watch.json']:
    g=read(HERE/name);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert read(HERE/'unit.json')['passed']
TCI=D/'terminal_covariance_information_v1';g=read(TCI/'capture_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
r=read(TCI/'capture/result.json');assert r['completed'] and r['repeat_all_captured_arrays_exact']
out=HERE/'qualification';out.mkdir(exist_ok=False);records=[];first=None;first_bill=None
for row in r['records']:
    assert sha(row['capture'])==row['capture_sha256'] and row['parent_all16_and_bill_exact'] and row['numeric_cache_retired']
    with np.load(row['capture'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['C','H14','alive','W','mu','v','H15','sigma','prediction']}
    with flops.BudgetContext(flop_budget=5_000_000_000,quiet=True) as ctx:
        features=latent_covariance(fnp,*[fnp.asarray(a[k]) for k in ['C','H14','alive','W','mu','v']])
    bill=ctx.summary_dict()['flops_used'];assert bill<5_000_000_000
    F=np.asarray(features).copy();assert F.shape==(1024,2080) and F.dtype==np.float32 and np.isfinite(F).all()
    if row['call']==0:first=F.copy();first_bill=bill
    if row['call']==2:assert np.array_equal(first,F) and first_bill==bill
    path=out/f"call_{row['call']:02d}.npz";np.savez(path,features=F,hidden=a['H15'],sigma=a['sigma'],prediction=a['prediction'])
    rec=dict(call=row['call'],rank=row['rank'],capture=str(path),capture_sha256=sha(path),parent_flops=row['parent_flops'],parent_multiplier=max(.1,row['parent_flops']/2**41),feature_flops=bill,weights_sha256=row['weights_sha256'],includes_initialization=row['call']==0,source_capture_sha256=row['capture_sha256'],truth_accessed=False)
    records.append(rec);print(json.dumps(rec),flush=True)
write(out/'result.json',dict(completed=True,records=records,feature_repeat_exact=True,full_model_calls=0,truth_accessed=False,source_seal_sha256=sha(HERE/'sources_sealed.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
