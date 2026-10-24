"""Read-only native capture. NumPy diagnostics are not charged inference work."""
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
from graph import graph_fields, transport

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
qual=json.loads((HERE/'qualification.json').read_text())
qg=json.loads((HERE/'qualification_guard.json').read_text())
assert qual['passed'] and qual['graph_sha256']==sha(HERE/'graph.py')
assert qual['protocol_sha256']==sha(HERE/'protocol.json')
assert qg['completed'] and qg['whole_group_retired'] and qg['max_swap_growth_bytes']==0
group=int(sys.argv[1]);ranks=protocol['capture_groups'][group]
out=HERE/f'capture_{group:02d}';out.mkdir()
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)==protocol['parent_sha256']
if group:
    smoke=json.loads((HERE/'capture_00'/'complete.json').read_text())
    sg=json.loads((HERE/'capture_00_guard.json').read_text())
    assert smoke['independent_controls_exact']==2 and sg['completed'] and sg['max_swap_growth_bytes']==0
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
controls={r['rank']:r for r in map(json.loads,(H2/'both_tail_train2_v6'/'rows.jsonl').read_text().splitlines())}
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('native_graph_capture');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
exec(compile(parent.read_text(),str(parent),'exec'),mod.__dict__)
oldfeatures,oldfinal,oldstep=mod.features,mod.final_scalar_features,mod.gnn_step
current={}

def features(l,m,C,K21,aux,w):
    x=oldfeatures(l,m,C,K21,aux,w)
    if 12<=l<=14:
        if l==12:
            assert not current
            current['history']=np.empty((1024,0));current['layer']=11
        assert current['layer']==l-1
        history=current['history']
        if history.shape[1]:history=transport(history,aux['gate'],w)
        fields=graph_fields(m,C,aux)*np.asarray(aux['sigma'])[:,None]
        current['history']=np.concatenate((history,fields),axis=1)
        current['layer']=l
    return x

def final(l,m,aux,w):
    x=oldfinal(l,m,aux,w)
    assert l==15 and current['layer']==14
    current['history']=transport(current['history'],aux['gate'],w)
    current['layer']=15
    return x

def step(l,x,sigma,w,state,P):
    value=oldstep(l,x,sigma,w,state,P)
    if l==15:
        assert current['layer']==15
        s=np.array(sigma,dtype=np.float64)
        control=np.concatenate((np.asarray(x),np.asarray(value[1])),axis=1).astype(np.float32)
        graph=(current.pop('history')/s[:,None]).astype(np.float32)
        assert control.shape==(1024,94) and graph.shape==(1024,57)
        assert np.isfinite(control).all() and np.isfinite(graph).all() and (s>0).all()
        current.update(control=control,graph=graph,sigma=s)
    return value

mod.features,mod.final_scalar_features,mod.gnn_step=features,final,step
est=mod.Estimator()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
records=[];start=time.monotonic()
with (out/'rows.jsonl').open('x') as stream:
    for rank in ranks:
        assert not current
        wp=store/Path(manifest[rank]['path']).parent/'weights.npy'
        w=np.load(wp)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
        t0=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
        bill=int(ctx.summary_dict()['flops_used'])
        assert y.shape==(16,1024) and np.isfinite(y).all()
        exact=None
        if group==0:
            base=np.load(H2/'both_tail_train2_v6'/f'pred_{rank}.npy')
            exact=bool(np.array_equal(y,base) and bill==controls[rank]['flops'])
            assert exact,(rank,bill,controls[rank]['flops'])
            del base
        packet=out/f'row_{rank:04d}.npz'
        with packet.open('xb') as f:
            np.savez(f,control=current['control'],graph=current['graph'],sigma=current['sigma'],prediction=y)
        rec=dict(rank=rank,flops=bill,wall_s=time.monotonic()-t0,path=str(packet.relative_to(HERE)),
                 bytes=packet.stat().st_size,sha256=sha(packet),weights_sha256=sha(wp),
                 independent_all16_and_bill_exact=exact,targets_accessed=False)
        records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
        print(json.dumps(dict(event='capture',group=group,rank=rank,wall_s=rec['wall_s'],elapsed_s=time.monotonic()-start)),flush=True)
        current.clear();del w,mlp,y;gc.collect();relief(None,0)
result=dict(records=records,targets_accessed=False,group=group,protocol_sha256=sha(HERE/'protocol.json'),
            capture_sha256=sha(HERE/'capture.py'),graph_sha256=sha(HERE/'graph.py'),parent_sha256=sha(parent),
            independent_controls_exact=sum(r['independent_all16_and_bill_exact'] is True for r in records),
            bytes=sum(r['bytes'] for r in records),elapsed_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'complete.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(event='complete',group=group,bytes=result['bytes'])),flush=True)
