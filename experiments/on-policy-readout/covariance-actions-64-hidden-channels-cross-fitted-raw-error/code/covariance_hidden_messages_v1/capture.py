from pathlib import Path
import ctypes
import gc
import hashlib
import json
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
from fields import source, terminal

HERE=Path(__file__).resolve().parent; H2=HERE.parent; P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((HERE/'seal.json').read_text())
for name,digest in seal['sources'].items(): assert sha(HERE/name)==digest
assert json.loads((HERE/'unit_result.json').read_text())['passed']
ug=json.loads((HERE/'unit_guard.json').read_text())
assert ug['completed'] and ug['whole_group_retired'] and ug['max_swap_growth_bytes']==0
group=int(sys.argv[1]); assert group in (0,1)
protocol=json.loads((HERE/'protocol.json').read_text())
ranks=protocol['ranks'][:2] if group==0 else protocol['ranks'][2:]
if group:
    g=json.loads((HERE/'capture_00_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert json.loads((HERE/'capture_00/complete.json').read_text())['independent_controls_exact']==2
out=HERE/f'capture_{group:02d}';out.mkdir()
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)==protocol['parent_sha256']
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
controls={}
for dirname in ('both_tail_train2_v6','both_tail_train6_v6_routing_control'):
    for line in (H2/dirname/'rows.jsonl').read_text().splitlines():
        r=json.loads(line);controls[r['rank']]=(r,H2/dirname)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('covariance_hidden_capture');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
exec(compile(parent.read_text(),str(parent),'exec'),mod.__dict__)
oldfeatures,oldfinal,oldstep=mod.features,mod.final_scalar_features,mod.gnn_step
current={}

def features(l,m,C,K21,aux,w):
    x=oldfeatures(l,m,C,K21,aux,w)
    if l==14:
        assert not current
        current.update(C=np.array(C,dtype=np.float64), s14=np.array(aux['sigma'],dtype=np.float64))
    return x

def final(l,m,aux,w):
    x=oldfinal(l,m,aux,w)
    assert l==15 and 'U' in current
    current['messages']=terminal(current.pop('U'),current.pop('V'),current.pop('s14'),
        np.asarray(w),np.asarray(aux['sigma']),np.asarray(aux['gate']),np.asarray(aux['pdf'])).astype(np.float32)
    return x

def step(l,x,sigma,w,state,P):
    value=oldstep(l,x,sigma,w,state,P)
    if l==14:
        H=np.array(value[1],dtype=np.float64)
        C=current.pop('C');U,V=source(C,current['s14'],H)
        # Eight scalar entries are independently recomputed by math.fsum.
        import math
        n=len(H); errs=[]
        hc=H-H.mean(axis=0,keepdims=True)
        for i,a in ((0,0),(17,9),(253,31),(1019,63)):
            r=C[i]/(current['s14'][i]*current['s14']);r[i]=0.
            uu=math.fsum(float(r[j]*hc[j,a]) for j in range(n))/math.sqrt(n)
            vv=math.fsum(float(r[j]*r[j]*hc[j,a]) for j in range(n))
            errs.extend((abs(uu-U[i,a]),abs(vv-V[i,a])))
        current.update(U=U,V=V,witness_error=max(errs))
        assert current['witness_error']<1e-11
    if l==15:
        current.update(control=np.concatenate((np.asarray(x),np.asarray(value[1])),axis=1).astype(np.float32),
                       sigma=np.array(sigma,dtype=np.float64))
        assert current['control'].shape==(1024,94) and current['messages'].shape==(1024,128)
    return value

mod.features,mod.final_scalar_features,mod.gnn_step=features,final,step
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
records=[];start=time.monotonic()
with (out/'rows.jsonl').open('x') as stream:
    for rank in ranks:
        assert not current
        wp=store/Path(manifest[rank]['path']).parent/'weights.npy';w=np.load(wp)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
        t0=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
        bill=int(ctx.summary_dict()['flops_used'])
        assert y.shape==(16,1024) and np.isfinite(y).all()
        exact=None
        if rank in controls:
            r,d=controls[rank];base=np.load(d/f'pred_{rank}.npy')
            exact=bool(np.array_equal(y,base) and bill==r['flops']);assert exact,(rank,bill,r['flops'])
            del base
        packet=out/f'row_{rank:04d}.npz'
        with packet.open('xb') as f:
            np.savez(f,control=current['control'],messages=current['messages'],sigma=current['sigma'],prediction=y)
        rec=dict(rank=rank,flops=bill,wall_s=time.monotonic()-t0,path=str(packet.relative_to(HERE)),
                 bytes=packet.stat().st_size,sha256=sha(packet),weights_sha256=sha(wp),
                 independent_all16_and_bill_exact=exact,source_witness_error=current['witness_error'],targets_accessed=False)
        records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
        print(json.dumps(dict(event='capture',group=group,rank=rank,wall_s=rec['wall_s'],elapsed_s=time.monotonic()-start)),flush=True)
        current.clear();del w,mlp,y;gc.collect();relief(None,0)
result=dict(rows=records,targets_accessed=False,group=group,seal_sha256=sha(HERE/'seal.json'),
            parent_sha256=sha(parent),independent_controls_exact=sum(r['independent_all16_and_bill_exact'] is True for r in records),
            bytes=sum(r['bytes'] for r in records),elapsed_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'complete.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(event='complete',group=group,bytes=result['bytes'])),flush=True)
