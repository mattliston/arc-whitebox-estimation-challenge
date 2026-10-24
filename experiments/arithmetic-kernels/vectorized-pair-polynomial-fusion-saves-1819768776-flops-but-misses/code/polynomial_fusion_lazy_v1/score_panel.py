"""Recompute paired errors for the fixed fourteen-network panel."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=read(HERE/'build.json')
panel=read(HERE/'panel/completed.json')
parent=read(D/'lazy_rotation_objective_v1/panel/completed.json')
prior={r['rank']:r for r in read(D/'lazy_rotation_objective_v1/panel_score.json')['rows']}
assert panel['completed'] and len(panel['records'])==14
assert sha(HERE/'candidate.py')==panel['candidate_sha256']==build['candidate_sha256']
assert parent['candidate_sha256']==build['parent_sha256']
parents={r['rank']:r for r in parent['records']}
joined=[]
for entry in panel['records']:
    rank=entry['rank']
    arms={'candidate':entry['call'],'lazy':parents[rank]['call']}
    assert arms['candidate']['weights_sha256']==arms['lazy']['weights_sha256']
    predictions={}
    for name,call in arms.items():
        assert call['includes_initialization'] and sha(call['output'])==call['output_sha256']
        pred=np.load(call['output'])
        assert pred.shape==(16,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and (pred>=0).all()
        predictions[name]=pred
    assert np.array_equal(predictions['candidate'][:1],predictions['lazy'][:1])
    delta=predictions['candidate'].astype(np.float64)-predictions['lazy'].astype(np.float64)
    joined.append(dict(rank=rank,arms=arms,first_layer_byte_exact=True,
                       all_layer_parent_maxabs=float(np.max(np.abs(delta))),
                       final_parent_delta_mse=float(np.mean(delta[-1]**2))))
assert [r['rank'] for r in joined]==[35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505]
with (HERE/'panel_comparison_sealed.json').open('x') as f:
    json.dump(dict(candidate_sha256=build['candidate_sha256'],joined=joined,targets_opened=False),f,indent=2)
rows=[]
for entry in joined:
    rank=entry['rank']
    path=D.parent/'public_g19_attribution_v1'/(f'fresh_{rank}_truth4M.npz' if rank>=3016500 else f'mps_truth_rank{rank:04d}_indep4M.npz')
    assert sha(path)==prior[rank]['truth_sha256']
    with np.load(path) as z:
        truth=z['mc_post'][15].astype(np.float64)
        noise=float(np.mean(z['mc_se'][15].astype(np.float64)**2))
    row=dict(rank=rank,truth_sha256=sha(path),noise_mse=noise,
             all_layer_parent_maxabs=entry['all_layer_parent_maxabs'],final_parent_delta_mse=entry['final_parent_delta_mse'])
    for name,call in entry['arms'].items():
        pred=np.load(call['output'])[-1].astype(np.float64)
        raw=float(np.mean((pred-truth)**2))
        row[name]=dict(raw=raw,raw_nc=raw-noise,flops=call['flops'],multiplier=call['multiplier'],
                       adjusted=raw*call['multiplier'],adjusted_nc=(raw-noise)*call['multiplier'])
    assert abs(row['lazy']['adjusted_nc']-prior[rank]['candidate']['adjusted_nc'])<1e-20
    row['adjusted_nc_ratio']=row['candidate']['adjusted_nc']/row['lazy']['adjusted_nc']
    rows.append(row)
groups={}
for name,selected in [('public8',rows[:8]),('fresh6',rows[8:]),('all14',rows)]:
    block={arm:{key:float(np.mean([r[arm][key] for r in selected]))
                for key in ('raw','raw_nc','flops','multiplier','adjusted','adjusted_nc')}
           for arm in ('candidate','lazy')}
    block['adjusted_nc_ratio']=block['candidate']['adjusted_nc']/block['lazy']['adjusted_nc']
    block['wins']=sum(r['adjusted_nc_ratio']<1 for r in selected)
    groups[name]=block
passed=(all(g['adjusted_nc_ratio']<1 for g in groups.values()) and
        groups['all14']['adjusted_nc_ratio']<=.998 and groups['all14']['wins']>=12 and
        max(r['adjusted_nc_ratio'] for r in rows)<=1.01)
result=dict(candidate_sha256=build['candidate_sha256'],rows=rows,groups=groups,
            official_bank_ratio_projection=3.8164492657795035e-9*groups['all14']['adjusted_nc_ratio'],
            panel_gate_passed=passed,raw_targets_recomputed=True,old_component_gate_remains_failed=True,
            caveat='Reused paired noise-corrected panel; no independent or official score claim.')
with (HERE/'panel_score.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
