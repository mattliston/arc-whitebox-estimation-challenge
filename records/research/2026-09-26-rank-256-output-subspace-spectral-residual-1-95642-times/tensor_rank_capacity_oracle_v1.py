"""Optimistic rank-256 capacity diagnostic on an existing sealed tensor packet.

This separates subspace capacity from the failed one-pass tensor sketch.
It reads no endpoint labels and measures no deployable gain. Full-tensor
Frobenius error is distinct from endpoint and repeated-index-plane error.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'codex_resume_20260919_v1/full_tensor_sketch_information_v1'
meta=json.loads((OLD/'capture/result.json').read_text())
packet=OLD/'capture/join.npz'
assert hashlib.sha256(packet.read_bytes()).hexdigest()==meta['packet_sha256']
assert meta['all16_parent_bytes_exact'] and meta['parent_bill_exact'] and not meta['truth_accessed']
assert all(p['num_threads']==1 for p in threadpool_info())
out=HERE/'tensor_rank_capacity_oracle_v1'
out.mkdir()
began=time.monotonic()

def inner_from_gram(G,lam):
    signed=0.;absolute=0.
    for start in range(0,len(lam),64):
        block=G[start:start+64,:]
        weighted=(block*block*block)*lam[start:start+64,None]*lam[None,:]
        signed+=float(np.sum(weighted))
        absolute+=float(np.sum(np.abs(weighted)))
    return signed,absolute

# Explicit tiny third tensors certify the complete cross-pair Gram and norm.
rng=np.random.default_rng(20260926)
unit=[]
for kind in ('signed','close_pair','zero_pair'):
    f=rng.normal(size=(4,6));lam=rng.normal(size=6)
    if kind!='signed':
        f[:,1]=f[:,0]+(1e-4*rng.normal(size=4) if kind=='close_pair' else 0)
        lam[0],lam[1]=1.,-1.
        if kind=='zero_pair':lam[2:]=0
    tensor=np.einsum('ic,jc,kc,c->ijk',f,f,f,lam)
    gram=f.T@f
    H=f@((gram*gram)*lam[:,None]*lam[None,:])@f.T
    direct=tensor.reshape(4,-1)@tensor.reshape(4,-1).T
    err=float(np.max(np.abs(H-direct)))
    assert err<=1e-10*(1+float(np.max(np.abs(direct))))
    norm,_=inner_from_gram(gram,lam)
    assert abs(norm-float(np.sum(tensor*tensor)))<=1e-10*(1+abs(norm))
    unit.append(dict(kind=kind,max_gram_error=err))

with np.load(packet) as z:
    Q,F,J,lam,W,parent=[z[k].astype(np.float64) for k in ('Q','F','J','lambda','W','parent_pushed')]
target=W.T@np.concatenate([Q@F,J],axis=1)
del Q,F,J,W
assert target.shape==parent.shape==(1024,meta['total_columns'])
print(json.dumps(dict(event='packet_ready',shape=list(target.shape))),flush=True)
G=target.T@target
target_norm,target_abs=inner_from_gram(G,lam)
weighted=G*G
weighted*=lam[:,None]
weighted*=lam[None,:]
H=(target@weighted)@target.T
del G,weighted
symmetry=float(np.linalg.norm(H-H.T)/np.linalg.norm(H))
H=(H+H.T)*.5
values,vectors=np.linalg.eigh(H)
assert np.isfinite(values).all() and values[-1]>0
assert values[0]>=-1e-8*values[-1], ('PSD diagnostic failed',values[0],values[-1])
assert abs(float(values.sum())/target_norm-1)<1e-7
tail256=float(np.sum(values[:-256]))
print(json.dumps(dict(event='spectrum',target_norm=target_norm,tail256=tail256,
                      signed_energy_condition=target_abs/target_norm)),flush=True)
G=parent.T@parent
parent_norm,parent_abs=inner_from_gram(G,lam)
del G
G=target.T@parent
cross,cross_abs=inner_from_gram(G,lam)
del G
parent_error=target_norm+parent_norm-2*cross
assert parent_error>0 and np.isfinite(parent_error)
record=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            packet_sha256=meta['packet_sha256'],capture_bank='Lazy #331599, not current S31',
            network_rank=35,layer=8,parent_rank=384,trial_rank=256,truth_accessed=False,
            unit_checks=unit,gram_symmetry_relative=symmetry,min_eigenvalue=float(values[0]),
            signed_energy_condition=target_abs/target_norm,
            target_tensor_norm_squared=target_norm,parent_tensor_error_squared=parent_error,
            parent_relative_tensor_error_squared=parent_error/target_norm,
            best_possible_rank256_one_mode_residual=tail256,
            rank256_lower_bound_over_parent_full_error=tail256/parent_error,
            implication='Any rank-256 output subspace has full-tensor projection error at least this one-mode residual, in exact arithmetic.',
            numerical_limitation='FP64 contraction diagnostic, not an interval-certified bound or an endpoint/plane-error certificate.')
if tail256<1.1*parent_error:
    q=vectors[:,-256:]
    small=q.T@target
    G=small.T@small
    projected_norm,_=inner_from_gram(G,lam)
    del G,small
    error=target_norm-projected_norm
    record['hosvd_rank256_full_error_over_parent']=error/parent_error
    record['hosvd_rank256_relative_full_error']=error/target_norm
    np.save(out/'oracle_basis256.npy',q)
record['seconds']=time.monotonic()-began
np.save(out/'flattening_spectrum.npy',values)
with (out/'result.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
