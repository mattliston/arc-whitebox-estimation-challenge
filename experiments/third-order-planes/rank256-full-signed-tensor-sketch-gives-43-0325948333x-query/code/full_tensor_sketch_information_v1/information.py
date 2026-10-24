"""Fixed optimistic rank256 tensor-sketch comparison; never reads truth."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
from algebra import contract,plane,project,basis

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(HERE/'source_sealed.json')
for name,digest in seal['files_sha256'].items():assert sha(HERE/name)==digest,name
for name in ['unit_watch.json','capture_watch.json']:
    w=read(HERE/name)
    assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes']==0
meta=read(HERE/'capture/result.json')
assert meta['all16_parent_bytes_exact'] and meta['parent_bill_exact'] and meta['cache_retired']
assert sha(HERE/'capture/join.npz')==meta['packet_sha256']
assert all(x['num_threads']==1 for x in threadpool_info())
begin=time.monotonic()
with np.load(HERE/'capture/join.npz') as z:
    Q,F,J,lam,W,parent=[z[k].astype(np.float64) for k in ['Q','F','J','lambda','W','parent_pushed']]
p=np.concatenate([Q@F,J],axis=1)
assert p.shape==(meta['previous_active'],meta['total_columns'])
rng=np.random.Generator(np.random.PCG64(2026092003))
a=rng.standard_normal((p.shape[0],256))
b=rng.standard_normal((p.shape[0],256))
q,r=basis(p,lam,a,b)
orthogonality=float(np.linalg.norm(q.T@q-np.eye(256))/np.sqrt(256))
assert orthogonality<=1e-11
trial=W.T@project(p,q)
reference=W.T@p
out=HERE/'information'
out.mkdir(exist_ok=False)
np.savez(out/'fixed_basis.npz',Q=q,R=r)
with (out/'basis_sealed_before_verification.json').open('x') as stream:
    json.dump(dict(basis_sha256=sha(out/'fixed_basis.npz'),fit_seed=2026092003,
                   fit_probes=256,packet_sha256=meta['packet_sha256'],truth_accessed=False),stream,indent=2)
del a,b,Q,F,J,r
verify=np.random.Generator(np.random.PCG64(2026092004))
aa=verify.standard_normal((1024,512))
bb=verify.standard_normal((1024,512))
target=contract(reference,lam,aa,bb)
old_queries=contract(parent,lam,aa,bb)
new_queries=contract(trial,lam,aa,bb)
old_error=np.sum((old_queries-target)**2,axis=0)
new_error=np.sum((new_queries-target)**2,axis=0)
probe_ratio=float(new_error.sum()/old_error.sum())
probe_wins=int(np.sum(new_error<old_error))
target_plane=plane(reference,lam)
old_plane=plane(parent,lam)
new_plane=plane(trial,lam)
old_plane_error=float(np.sum((old_plane-target_plane)**2))
new_plane_error=float(np.sum((new_plane-target_plane)**2))
plane_ratio=new_plane_error/old_plane_error
old_diag_error=float(np.sum((np.diag(old_plane)-np.diag(target_plane))**2))
new_diag_error=float(np.sum((np.diag(new_plane)-np.diag(target_plane))**2))
np.savez(out/'residual_statistics.npz',parent_probe_energy=old_error,
         candidate_probe_energy=new_error,parent_plane_error=old_plane-target_plane,
         candidate_plane_error=new_plane-target_plane)
record=dict(information_gate_passed=probe_ratio<=.75 and plane_ratio<=.75,
            probe_residual_energy_ratio=probe_ratio,plane_residual_energy_ratio=plane_ratio,
            parent_probe_energy=float(old_error.sum()),candidate_probe_energy=float(new_error.sum()),
            parent_relative_probe_residual_squared=float(old_error.sum()/np.sum(target*target)),
            candidate_relative_probe_residual_squared=float(new_error.sum()/np.sum(target*target)),
            probe_wins=probe_wins,verification_probe_count=512,
            paired_probe_energy_difference_mean=float(np.mean(new_error-old_error)),
            paired_probe_energy_difference_se=float(np.std(new_error-old_error,ddof=1)/np.sqrt(512)),
            parent_plane_squared_error=old_plane_error,candidate_plane_squared_error=new_plane_error,
            parent_relative_plane_residual_squared=old_plane_error/float(np.sum(target_plane*target_plane)),
            candidate_relative_plane_residual_squared=new_plane_error/float(np.sum(target_plane*target_plane)),
            marginal_diagonal_residual_energy_ratio=new_diag_error/old_diag_error,
            qr_orthogonality_relative=orthogonality,fit_probes=256,fit_seed=2026092003,
            verification_seed=2026092004,rank=256,parent_rank=384,layer=8,network_rank=35,
            truth_accessed=False,actual_accuracy_score_measured=False,
            production_cost_measured=False,precision='optimistic FP64 trial versus actual rounded parent factors',
            seconds=time.monotonic()-begin,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            packet_sha256=meta['packet_sha256'],basis_sha256=sha(out/'fixed_basis.npz'),
            source_sha256=sha(__file__),algebra_sha256=sha(HERE/'algebra.py'))
assert all(np.isfinite(x).all() for x in [q,trial,reference,target,old_queries,new_queries,target_plane,old_plane,new_plane])
with (out/'result.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
