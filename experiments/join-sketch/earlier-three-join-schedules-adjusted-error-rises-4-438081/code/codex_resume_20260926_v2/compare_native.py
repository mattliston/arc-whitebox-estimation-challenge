"""Paired pooled-score comparison and bootstrap CI for native_eval receipts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def load(spec):
    rows={};sources=[];hashes=set()
    for name in spec.split(','):
        path=HERE/name
        manifest=json.loads((path/'manifest.json').read_text())
        assert hashlib.sha256(Path(manifest['candidate']).read_bytes()).hexdigest()==manifest['sha256']
        hashes.add(manifest['sha256']);sources.append(manifest)
        for line in (path/'rows.jsonl').read_text().splitlines():
            row=json.loads(line);rank=row['rank']
            assert rank not in rows,rank
            assert np.isfinite([row['mse'],row['adjusted'],row['util']]).all()
            rows[rank]=row
    assert len(hashes)==1,hashes
    return rows,sources

base,bs=load(sys.argv[1]);cand,cs=load(sys.argv[2])
assert set(base)==set(cand),(sorted(base),sorted(cand))
ranks=sorted(base)
b=np.array([[base[r][k] for k in ['mse','adjusted','util','cpu_s']] for r in ranks])
c=np.array([[cand[r][k] for k in ['mse','adjusted','util','cpu_s']] for r in ranks])
rng=np.random.default_rng(20260926)
idx=rng.integers(0,len(ranks),size=(50000,len(ranks)))
ratios=c[idx,:2].mean(1)/b[idx,:2].mean(1)-1
ci=np.quantile(ratios,[.025,.975],axis=0)
result=dict(n=len(ranks),ranks=ranks,raw_pct=100*(c[:,0].mean()/b[:,0].mean()-1),
    raw_ci95_pct=(ci[:,0]*100).tolist(),adjusted_pct=100*(c[:,1].mean()/b[:,1].mean()-1),
    adjusted_ci95_pct=(ci[:,1]*100).tolist(),raw_wins=int((c[:,0]<b[:,0]).sum()),adjusted_wins=int((c[:,1]<b[:,1]).sum()),
    base_mean_adjusted=float(b[:,1].mean()),candidate_mean_adjusted=float(c[:,1].mean()),
    mean_util_pct=100*(c[:,2].mean()/b[:,2].mean()-1),mean_cpu_pct=100*(c[:,3].mean()/b[:,3].mean()-1),
    base_sources=bs,candidate_sources=cs,
    per_network=[dict(rank=r,raw_pct=100*(cand[r]['mse']/base[r]['mse']-1),adjusted_pct=100*(cand[r]['adjusted']/base[r]['adjusted']-1),util_pct=100*(cand[r]['util']/base[r]['util']-1)) for r in ranks])
out=HERE/sys.argv[3]
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['base_sources','candidate_sources','per_network']}))
