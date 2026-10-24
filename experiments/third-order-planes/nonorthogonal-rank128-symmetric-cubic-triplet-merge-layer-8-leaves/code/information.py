"""One fixed two-sweep triplet endpoint; no mean references or model calls."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
from algebra import Products,fit,cp_query,triplet_query,triplet_plane,norms
HERE=Path(__file__).resolve().parent;D=HERE.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=read(HERE/'sources_sealed.json')
for x in seal['sources']+seal['external_pins']:assert sha(x['path'])==x['sha256'],x['path']
unit=read(HERE/'unit.json');assert unit['passed']
guard=read(HERE/'unit_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert all(t['num_threads']==1 for t in threadpool_info())
packet=D/'full_tensor_sketch_information_v1/capture/join.npz'
assert sha(packet)=='adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6'
meta=read(packet.parent/'result.json');assert meta['all16_parent_bytes_exact'] and meta['parent_bill_exact'] and meta['cache_retired']
started=time.monotonic()
with np.load(packet,allow_pickle=False) as z:q=z['Q'].astype(np.float64);f=z['F'].astype(np.float64);lam=z['lambda'][:f.shape[1]].astype(np.float64);w=z['W'].astype(np.float64)
assert q.shape==(896,384) and f.shape==(384,6144) and w.shape==(896,1024)
assert all(np.isfinite(x).all() for x in [q,f,lam,w])
products=Products();u,r=np.linalg.qr(q,mode='reduced');a=products.mm(r,f,'construction_coordinate_change')
qr=float(np.linalg.norm(products.mm(u,r,'qualification_qr')-q)/np.linalg.norm(q))
orth=float(np.linalg.norm(products.mm(u.T,u,'qualification_orthogonality')-np.eye(384))/np.sqrt(384))
assert qr<=1e-10 and orth<=1e-10
factors,solves=fit(a,lam,128,products)
physical=[products.mm(u,x,'construction_physical_factors') for x in factors]
assert all(np.isfinite(x).all() and x.shape==(896,128) for x in physical)
np.savez(HERE/'fixed_merge.npz',A=physical[0],B=physical[1],C=physical[2])
with (HERE/'merge_sealed_before_verification.json').open('x') as out:
    json.dump(dict(merge_sha256=sha(HERE/'fixed_merge.npz'),packet_sha256=sha(packet),seed=2026092005,sweeps=2,triplets=128,total_factor_columns=384,symmetrized=True,solves=solves,qr_relative_error=qr,orthogonality=orth,truth_accessed=False),out,indent=2)
nr=norms(a,lam,factors,products)
old_norm=read(D/'orthogonal_cubic_merge_information_v1/result.json')['old_tensor_frobenius_squared']
assert np.isclose(nr['target_norm_squared'],old_norm,rtol=1e-10,atol=0)
assert nr['target_norm_squared']>0 and all(np.isfinite(v) for v in nr.values())
assert nr['symmetrized_error_squared']>=-1e-10*nr['target_norm_squared']
assert nr['unsymmetrized_error_squared']+1e-10*nr['target_norm_squared']>=nr['symmetrized_error_squared']
old=products.mm(w.T,products.mm(q,f,'diagnostic_old_physical'),'diagnostic_old_push')
new=[products.mm(w.T,x,'diagnostic_merged_push') for x in physical]
verify=np.random.Generator(np.random.PCG64(2026092004));aa=verify.standard_normal((1024,512));bb=verify.standard_normal((1024,512))
target=cp_query(old,lam,aa,bb,products,'diagnostic_reference_query');trial=triplet_query(new,aa,bb,products,'diagnostic_triplet_query')
signal=np.sum(target*target,axis=0);error=np.sum((trial-target)**2,axis=0)
assert signal.sum()>1e-24
query_ratio=float(error.sum()/signal.sum())
tp=products.mm(old*old,(old*lam[None,:]).T,'diagnostic_reference_plane');mp=triplet_plane(new,products,'diagnostic_triplet_plane')
plane_signal=float(np.sum(tp*tp));plane_error=float(np.sum((mp-tp)**2));assert plane_signal>1e-24
plane_ratio=plane_error/plane_signal
diag_signal=float(np.sum(np.diag(tp)**2));assert diag_signal>1e-24
diag_ratio=float(np.sum((np.diag(mp)-np.diag(tp))**2))/diag_signal
assert all(np.isfinite(x).all() for x in [old,*new,target,trial,tp,mp])
np.savez(HERE/'query_statistics.npz',reference_energy=signal,residual_energy=error)
categories={}
for row in products.records:
    key=row['tag'].split('_')[0];categories[key]=categories.get(key,0)+row['fp64_leading_billed_flops']
result=dict(completed=True,information_gate_passed=query_ratio<=.01 and plane_ratio<=.01,query_relative_squared_error=query_ratio,plane_relative_squared_error=plane_ratio,diagonal_relative_squared_error=diag_ratio,symmetrized_tensor_relative_squared_error=nr['symmetrized_error_squared']/nr['target_norm_squared'],unsymmetrized_tensor_relative_squared_error=nr['unsymmetrized_error_squared']/nr['target_norm_squared'],norms=nr,checks=dict(qr=qr,orthogonality=orth,maximum_normal_equation_relative_residual=max(s['normal_equation_relative_residual'] for s in solves)),solves=solves,old_atoms=6144,triplets=128,total_factor_columns=384,physical_span_dimension=384,network_rank=35,entry_layer=8,verification_probes_reused=True,verification_seed=2026092004,verification_probes=512,initialization_seed=2026092005,full_sweeps=2,fp64_gemm_leading_subtotals=categories,products=products.records,complete_fnp_deployment_bill=False,actual_accuracy_score_measured=False,truth_accessed=False,model_calls=0,new_physical_samples=0,new_mean_training=False,nonlinear_tensor_fit=True,packet_sha256=sha(packet),merge_sha256=sha(HERE/'fixed_merge.npz'),elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'result.json').open('x') as out:json.dump(result,out,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['products','solves']}),flush=True)
