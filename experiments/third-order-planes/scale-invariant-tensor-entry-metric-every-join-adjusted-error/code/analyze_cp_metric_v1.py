"""Full signed-tensor fidelity screen; no endpoint labels or candidate scores."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
CAP=HERE/'cp_metric_capture_train2_v1'
OUT=HERE/'cp_scale_invariant_metric_information_v1.json'
assert not OUT.exists()
assert all(p['num_threads']==1 for p in threadpool_info())
began=time.monotonic()

def metric(p,lam):
    d=lam*lam*np.sum(p*p,axis=0)**2
    return (p*d[None,:])@p.T

def inner(p,lam):
    signed=[];absolute=[]
    for k in range(0,len(lam),128):
        g=p[:,k:k+128].T@p
        v=(g*g*g)*lam[k:k+128,None]*lam[None,:]
        signed.append(float(np.sum(v)))
        absolute.append(float(np.sum(np.abs(v))))
    return math.fsum(signed),math.fsum(absolute)

def basis(p,lam,rank,invariant):
    weight=lam*lam
    if invariant: weight*=np.sum(p*p,axis=0)**2
    rows=np.sort(np.argsort(-((p*p)@weight),kind='stable')[:rank])
    y=p@(weight[:,None]*p[rows,:].T)
    q,_=np.linalg.qr(y,mode='reduced')
    return q,rows

rng=np.random.default_rng(20260926)
unit=[]
for kind in ('signed','close_pair','zero_pair'):
    p=rng.normal(size=(5,8));lam=rng.normal(size=8)
    if kind!='signed':
        p[:,1]=p[:,0]+(1e-3*rng.normal(size=5) if kind=='close_pair' else 0)
        lam[0],lam[1]=1.,-1.
        if kind=='zero_pair':lam[2:]=0
    scale=np.array([.2,-.3,1.7,-3.,2.,.7,-1.4,.6])
    direct=np.einsum('ic,jc,kc,c->ijk',p,p,p,lam)
    scaled=np.einsum('ic,jc,kc,c->ijk',p*scale,p*scale,p*scale,lam/scale**3)
    terr=float(np.max(np.abs(direct-scaled)))
    merr=float(np.max(np.abs(metric(p,lam)-metric(p*scale,lam/scale**3))))
    assert terr<1e-10*(1+float(np.max(np.abs(direct))))
    assert merr<1e-10*(1+float(np.max(np.abs(metric(p,lam)))))
    q=np.linalg.qr(rng.normal(size=(5,3)))[0]
    projected=np.einsum('ia,jb,kc,abc->ijk',q,q,q,
                         np.einsum('ia,jb,kc,ijk->abc',q,q,q,direct))
    exact=float(np.sum((direct-projected)**2))
    n,_=inner(p,lam);pn,_=inner(q.T@p,lam)
    err=abs(n-pn-exact)
    assert err<1e-10*(1+float(np.sum(direct*direct)))
    unit.append(dict(kind=kind,tensor_gauge_max_error=terr,
                     metric_gauge_max_error=merr,projection_identity_error=err))

manifest=json.loads((CAP/'capture_manifest.json').read_text())
rows=list(map(json.loads,(CAP/'rows.jsonl').read_text().splitlines()))
assert len(rows)==2 and all(r['all16_bytes_exact'] and r['bill_exact'] and not r['truth_accessed'] for r in rows)
results=[]
for rec in manifest['records']:
    path=CAP/rec['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
    with np.load(path) as z:
        p=z['J'].astype(np.float64);lam=z['lam'].astype(np.float64)
        q0=np.linalg.qr(z['Qn'].astype(np.float64),mode='reduced')[0]
    rank=q0.shape[1]
    assert rank==384 and p.shape[1]==len(lam)
    norm,absolute=inner(p,lam)
    assert norm>0
    record=dict(rank=rec['rank'],layer=7,packet_sha256=rec['sha256'],
                shape=list(p.shape),retained_rank=rank,target_norm_squared=norm,
                signed_energy_condition=absolute/norm,
                column_norm_squared_quantiles=np.quantile(np.sum(p*p,axis=0),[0,.1,.5,.9,1]).tolist(),
                methods={})
    for name in ('actual_parent','fp64_parent_metric','scale_invariant_metric'):
        if name=='actual_parent':q=q0;selected=None
        else:q,selected=basis(p,lam,rank,name=='scale_invariant_metric')
        projected,projected_abs=inner(q.T@p,lam)
        error=norm-projected
        assert error>0 and np.isfinite(error)
        record['methods'][name]=dict(error_squared=error,relative_error_squared=error/norm,
            projected_signed_condition=projected_abs/projected,
            orthogonality_max_error=float(np.max(np.abs(q.T@q-np.eye(rank)))),
            selected_rows=None if selected is None else selected.tolist())
    old=record['methods']['actual_parent']['error_squared']
    new=record['methods']['scale_invariant_metric']['error_squared']
    ideal=record['methods']['fp64_parent_metric']['error_squared']
    record.update(proposed_over_actual_error=new/old,proposed_over_fp64_parent_error=new/ideal,
                  advance_gate_pass=bool(new<=.8*old))
    results.append(record)
    print(json.dumps({k:v for k,v in record.items() if k!='methods'}),flush=True)
    del p,lam,q0,q
record=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((HERE/'cp_scale_invariant_metric_protocol_v1.md').read_bytes()).hexdigest(),
            truth_accessed=False,unit_checks=unit,results=results,
            advance_to_native_train2=all(r['advance_gate_pass'] for r in results),
            seconds=time.monotonic()-began,
            limitation='FP64 full-tensor projection diagnostic on current S31 L7 TRAIN captures. No endpoint gain, deployable cost, or lower-rank capacity claim.')
with OUT.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(dict(advance_to_native_train2=record['advance_to_native_train2'],seconds=record['seconds'])),flush=True)
