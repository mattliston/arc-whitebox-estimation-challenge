"""Privileged local spans only; coefficients must never be deployed."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert all(t['num_threads']==1 for t in threadpool_info())
seal=read(HERE/'sources_sealed.json')
for x in seal['sources']+seal['external_pins']:assert sha(x['path'])==x['sha256'],x['path']
guard=read(HERE/'qualification_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
qual=read(HERE/'qualification/result.json');assert qual['completed'] and qual['passed']
out=HERE/'information';out.mkdir(exist_ok=False)
target_hash={35:'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',100:'637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
parent_raw={35:1.8706028490077942e-8,100:1.8424534175274928e-8}
for r in qual['records']:assert sha(r['path'])==r['sha256']
with (out/'before_truth_seal.json').open('x') as f:json.dump({'qualification_sha256':sha(HERE/'qualification/result.json'),'source_seal_sha256':sha(HERE/'sources_sealed.json'),'features':qual['records'],'target_hashes':target_hash},f,indent=2)

def oracle(A,error):
    rms=np.sqrt(np.mean(A*A,axis=0));nonzero=rms>0
    Z=A[:,nonzero]/rms[nonzero]
    U,s,Vh=np.linalg.svd(Z,full_matrices=False);keep=s>s[0]*1e-7
    projection=U[:,keep]@(U[:,keep].T@error);residual=error-projection
    beta=np.zeros(A.shape[1]);beta[nonzero]=(Vh[keep].T@((U[:,keep].T@error)/s[keep]))/rms[nonzero]
    assert np.linalg.norm(A@beta-projection)<=1e-8*np.linalg.norm(error)
    return dict(raw=float(np.mean(residual*residual)),rank=int(keep.sum()),singular_values=s.tolist(),rms=rms.tolist(),ideal_coefficients=beta.tolist(),normal_residual_norm=float(np.linalg.norm(U[:,keep].T@residual)),privileged_target_conditioned=True)

started=time.monotonic();records=[]
for rec in qual['records']:
    rank=rec['rank'];tp=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    assert sha(tp)==target_hash[rank]
    with np.load(tp,allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    with np.load(rec['path'],allow_pickle=False) as z:B=z['B'];J=z['J'];p=z['prediction'][-1].astype(np.float64)
    e=truth-p;raw=float(np.mean(e*e));assert np.isclose(raw,parent_raw[rank],rtol=2e-13,atol=0)
    control=oracle(B,e);cov=oracle(J,e);joint=oracle(np.concatenate([B,J],axis=1),e)
    ratios=dict(covariance_to_parent=cov['raw']/raw,control_to_parent=control['raw']/raw,joint_to_parent=joint['raw']/raw,joint_to_control=joint['raw']/control['raw'])
    passed=ratios['covariance_to_parent']<=.80 and ratios['joint_to_control']<=.80
    row=dict(rank=rank,parent_raw=raw,ratios=ratios,control=control,covariance=cov,joint=joint,gate_passed=bool(passed),truth_sha256=target_hash[rank],feature_sha256=rec['sha256'])
    records.append(row);print(json.dumps({'rank':rank,'ratios':ratios,'gate_passed':bool(passed)}),flush=True)
result=dict(completed=True,information_gate_passed=all(r['gate_passed'] for r in records),records=records,privileged_in_sample=True,linearized_unconstrained_outputs=True,incremental_inference_cost_zero=True,finite_candidate_score=False,new_model_fit=False,new_candidate=False,new_archive=False,new_upload=False,elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
