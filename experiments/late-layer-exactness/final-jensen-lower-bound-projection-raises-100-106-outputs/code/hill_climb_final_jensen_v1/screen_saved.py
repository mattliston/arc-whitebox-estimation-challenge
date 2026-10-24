"""Root-only fixed saved-output screen. Does not execute an estimator."""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2")
P=H.parent
D=Path(__file__).resolve().parent
def pin(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
ranks=[35,100]
parent=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
assert pin(parent)['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
store=P/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
mp=store/'manifest.json'
manifest=json.loads(mp.read_text())['roles']['observer_fit']['row_manifests']
base=H/'hill_climb_pair_rr_inverse_v1_train2_v1'
base_rows={x['rank']:x for x in map(json.loads,(base/'rows.jsonl').read_text().splitlines())}
protocol=dict(parent=pin(parent),saved_manifest=pin(base/'manifest.json'),saved_rows=pin(base/'rows.jsonl'),
 weight_manifest=pin(mp),script=pin(Path(__file__)),ranks=ranks,
 formula='Final output = FP32(max(saved FP32 final output, FP64 W15.T @ saved FP32 output14)); earlier outputs unchanged.',
 caveat='Jensen is exact for true means, but the served previous mean is an estimate. This is a fixed hypothesis, not guaranteed improvement.',
 screen='Saved-prediction two-network screen; no model run or native bill claim. No coefficients/windows or variants.',
 gates=dict(estimated_adjusted_pct_at_most=-.1,wins_min=1,max_regression_pct=1),
 conservative_extra_flops=4000000,
 next_stage='Only PASS licenses one real native N2 candidate; own native/boundary/N8 and >=1% S35 bank validation still required.')
once(D/'protocol.json',protocol)
rows=[]
for rank in ranks:
 pred=base/f'pred_{rank}.npy'
 wpath=store/Path(manifest[rank]['path']).parent/'weights.npy'
 truthpath=P/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
 y=np.load(pred);w=np.load(wpath,mmap_mode='r')
 mu=w[15].astype(np.float64).T@y[14]
 z=np.maximum(y[15],mu).astype(np.float32).astype(np.float64)
 truth=np.load(truthpath)['mc_post'][15].astype(np.float64)
 b=base_rows[rank];old=float(np.mean((y[15]-truth)**2));new=float(np.mean((z-truth)**2))
 assert old==b['mse']
 delta=z-y[15]
 adjusted=new*max(.1,(b['flops']+4000000)/2**41)
 rows.append(dict(rank=rank,base_raw=old,candidate_raw=new,raw_pct=100*(new/old-1),
  base_adjusted=b['adjusted'],estimated_candidate_adjusted=adjusted,
  estimated_adjusted_pct=100*(adjusted/b['adjusted']-1),
  changed=int(np.count_nonzero(delta)),max_raise=float(np.max(delta)),
  rms_raise=float(np.sqrt(np.mean(delta**2))),prediction=pin(pred),weights=pin(wpath),truth=pin(truthpath)))
 del y,w,truth,mu,z,delta
adj=100*(sum(r['estimated_candidate_adjusted'] for r in rows)/sum(r['base_adjusted'] for r in rows)-1)
wins=sum(r['estimated_adjusted_pct']<0 for r in rows)
maxreg=max(r['estimated_adjusted_pct'] for r in rows)
result=dict(protocol=pin(D/'protocol.json'),rows=rows,
 raw_pct=100*(sum(r['candidate_raw'] for r in rows)/sum(r['base_raw'] for r in rows)-1),
 estimated_adjusted_pct=adj,wins=wins,max_regression_pct=maxreg,
 admitted=adj<=-.1 and wins>=1 and maxreg<=1,model_executed=False,native_bill_measured=False)
once(D/'assessment.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

