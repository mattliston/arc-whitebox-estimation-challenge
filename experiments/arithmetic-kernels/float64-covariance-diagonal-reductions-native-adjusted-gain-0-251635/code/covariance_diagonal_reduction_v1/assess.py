"""Fixed native TRAIN2 endpoint and independent layer0 Gaussian check."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parent;P2=H.parent
OUT=HERE/'assessment.json';assert not OUT.exists()
seal=json.loads((HERE/'seal.json').read_text())
for name,digest in seal['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
g=json.loads((HERE/'train2_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
path=H/'covariance_diagonal_train2_v1'
rows=list(map(json.loads,(path/'rows.jsonl').read_text().splitlines()))
base={r['rank']:r for r in map(json.loads,(H/'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
assert [r['rank'] for r in rows]==[35,100]
assert json.loads((path/'manifest.json').read_text())['sha256']==seal['candidate_sha256']
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
checks=[]
for row in rows:
 rank=row['rank'];y=np.load(path/f'pred_{rank}.npy')
 assert y.shape==(16,1024) and np.isfinite(y).all()
 W=np.load(store/Path(manifest[rank]['path']).parent/'weights.npy',mmap_mode='r')[0].astype(np.float64)
 variance=np.array([math.fsum(float(t)*float(t) for t in W[:,j]) for j in range(1024)])
 gaussian=(np.sqrt(variance)/math.sqrt(2*math.pi)).astype(np.float32).astype(np.float64)
 layer0exact=bool(np.array_equal(y[0],gaussian));assert layer0exact
 truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
 raw=math.fsum((float(a)-float(b))**2 for a,b in zip(y[15],truth))/1024
 adj=raw*max(.1,row['flops']/2**41)
 assert abs(raw-row['mse'])<=1e-20 and abs(adj-row['adjusted'])<=1e-21
 checks.append(dict(rank=rank,layer0_exact_gaussian_bytes=True,raw=raw,adjusted=adj,flops=row['flops'],
                    raw_ratio=raw/base[rank]['mse'],bill_ratio=row['flops']/base[rank]['flops'],
                    adjusted_ratio=adj/base[rank]['adjusted'],prediction_sha256=hashlib.sha256((path/f'pred_{rank}.npy').read_bytes()).hexdigest()))
ratio=sum(r['adjusted'] for r in checks)/sum(base[r['rank']]['adjusted'] for r in checks)
rawratio=sum(r['raw'] for r in checks)/sum(base[r['rank']]['mse'] for r in checks)
billratio=sum(r['flops'] for r in checks)/sum(base[r['rank']]['flops'] for r in checks)
wins=sum(r['adjusted_ratio']<1 for r in checks)
record=dict(passed=wins==2 and ratio<=.99,rows=checks,pooled_adjusted_ratio=ratio,
 adjusted_change_pct=100*(ratio-1),pooled_raw_ratio=rawratio,raw_change_pct=100*(rawratio-1),
 pooled_bill_ratio=billratio,bill_change_pct=100*(billratio-1),adjusted_wins=wins,
 check32_accessed=False,valid40_accessed=False,new_samples=False,utc_finished=datetime.now(timezone.utc).isoformat())
with OUT.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2),flush=True)
