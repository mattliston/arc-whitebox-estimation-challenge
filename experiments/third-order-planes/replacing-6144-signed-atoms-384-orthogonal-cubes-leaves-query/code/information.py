"""One fixed modeled-tensor information screen; no truth or estimator calls."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_info
from algebra import Products,merge,contract,plane,norm_squared

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=read(HERE/'sources_sealed.json')
for item in seal['sources']+seal['external_pins']:
    assert sha(item['path'])==item['sha256'],item['path']
watch=read(HERE/'unit_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
assert read(HERE/'unit.json')['passed']
assert all(x['num_threads']==1 for x in threadpool_info())
packet=HERE.parent/'full_tensor_sketch_information_v1/capture/join.npz'
assert sha(packet)=='adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6'
meta=read(packet.parent/'result.json')
assert meta['all16_parent_bytes_exact'] and meta['parent_bill_exact'] and meta['cache_retired']
started=time.monotonic()
with np.load(packet) as z:
    q=z['Q'].astype(np.float64)
    f=z['F'].astype(np.float64)
    lam=z['lambda'][:f.shape[1]].astype(np.float64)
    w=z['W'].astype(np.float64)
assert q.shape==(896,384) and f.shape==(384,6144) and w.shape==(896,1024)
assert all(np.isfinite(z).all() for z in (q,f,lam,w))
products=Products()
basis,coefficient,a,u,eigen,checks=merge(q,f,lam,w[:,0],products)
np.savez(HERE/'fixed_merge.npz',basis=basis,coefficient=coefficient,eigenvalues=eigen)
with (HERE/'merge_sealed_before_verification.json').open('x') as stream:
    json.dump(dict(merge_sha256=sha(HERE/'fixed_merge.npz'),packet_sha256=sha(packet),
                   direction='recorded W column0, projected and normalized',
                   old_atoms=6144,new_atoms=384,physical_span_dimension=384,
                   truth_accessed=False,checks=checks),stream,indent=2)
# This is full signed tensor norm in orthonormal physical coordinates.
norm,absolute=norm_squared(a,lam,products)
retained=float(coefficient@coefficient)
assert norm>0 and np.isfinite(norm) and norm-retained>=-1e-10*norm
p=products.mm(q,f,'diagnostic_form_old_prefix')
reference=products.mm(w.T,p,'diagnostic_push_old')
merged=products.mm(w.T,basis,'diagnostic_push_merged')
verify=np.random.Generator(np.random.PCG64(2026092004))
aa=verify.standard_normal((1024,512));bb=verify.standard_normal((1024,512))
target=contract(reference,lam,aa,bb,products,'diagnostic_reference_query')
trial=contract(merged,coefficient,aa,bb,products,'diagnostic_merged_query')
signals=np.sum(target*target,axis=0)
errors=np.sum((trial-target)**2,axis=0)
assert float(signals.sum())>1e-24
query_ratio=float(errors.sum()/signals.sum())
target_plane=plane(reference,lam,products,'diagnostic_reference_plane')
trial_plane=plane(merged,coefficient,products,'diagnostic_merged_plane')
plane_signal=float(np.sum(target_plane*target_plane))
plane_error=float(np.sum((trial_plane-target_plane)**2))
assert plane_signal>1e-24
plane_ratio=plane_error/plane_signal
diag_signal=float(np.sum(np.diag(target_plane)**2))
diag_error=float(np.sum((np.diag(trial_plane)-np.diag(target_plane))**2))
assert diag_signal>1e-24
assert all(np.isfinite(x).all() for x in (p,reference,merged,target,trial,target_plane,trial_plane))
np.savez(HERE/'query_statistics.npz',reference_energy=signals,residual_energy=errors)
by_category={}
for rec in products.records:
    group=rec['tag'].split('_')[0]
    by_category[group]=by_category.get(group,0)+rec['fp64_leading_billed_flops']
result=dict(completed=True,numerical_checks_passed=True,
            information_gate_passed=query_ratio<=.01 and plane_ratio<=.01,
            query_relative_squared_error=query_ratio,plane_relative_squared_error=plane_ratio,
            diagonal_relative_squared_error=diag_error/diag_signal,
            query_signal_squared=float(signals.sum()),query_residual_squared=float(errors.sum()),
            plane_signal_squared=plane_signal,plane_residual_squared=plane_error,
            old_tensor_frobenius_squared=norm,retained_orthogonal_cube_energy=retained,
            tensor_relative_squared_error=(norm-retained)/norm,
            gram_absolute_to_signed_sum=absolute/norm,
            old_atoms=6144,new_atoms=384,physical_span_dimension=384,
            checks=checks,network_rank=35,entry_layer=8,
            scope='old shared prefix before the entry8 join, from stored rounded Q/F',
            verification_seed=2026092004,verification_probes=512,verification_probes_reused=True,
            precision='optimistic FP64',packet_sha256=sha(packet),
            merge_sha256=sha(HERE/'fixed_merge.npz'),
            source_sha256=sha(__file__),algebra_sha256=sha(HERE/'algebra.py'),
            fp64_gemm_leading_subtotals=by_category,products=products.records,
            complete_fnp_deployment_bill=False,actual_accuracy_score_measured=False,
            truth_accessed=False,model_calls=0,new_samples=0,new_training=False,
            seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            numpy_version=np.__version__,threads=threadpool_info())
with (HERE/'result.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='products'}),flush=True)
