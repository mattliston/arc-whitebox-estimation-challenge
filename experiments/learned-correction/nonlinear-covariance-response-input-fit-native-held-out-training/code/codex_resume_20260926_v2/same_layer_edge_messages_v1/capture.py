"""Passive, target-free native calibration with exact predictor/bill controls."""
from pathlib import Path
import ast
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

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
qualification=json.loads((HERE/'qualification.json').read_text())
guard=json.loads((HERE/'qualification_guard.json').read_text())
assert qualification['passed'] and guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert qualification['binding']['protocol_sha256']==sha(HERE/'protocol.json')
assert qualification['binding']['model_sha256']==sha(HERE/'model.py')
assert not (HERE/'calibration.json').exists()
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)==protocol['parent_sha256']
reference=H2/'response_message_capture_train12_v1'
refmanifest=json.loads((reference/'manifest.json').read_text())
assert refmanifest['sha256']==protocol['parent_sha256'] and refmanifest['ranks']==protocol['fit_ranks']
refrows={r['rank']:r for r in map(json.loads,(reference/'rows.jsonl').read_text().splitlines())}
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
parts=json.loads((P2/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert set(protocol['fit_ranks'])<=set(parts['train']) and set(protocol['fit_ranks']).isdisjoint(parts['valid'])
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('rnm_native_capture');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
source=parent.read_text();exec(compile(source,str(parent),'exec'),mod.__dict__)
fn=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_message_edge_source'][0]
body='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])
marker="    F=F/t['feature_scale'];"
assert body.count(marker)==1
body=body.replace(marker,"    _RNM_CAPTURE(layer,F,R,S,a,p,t)\n"+marker)
records=[];current={};totals={mode:dict(count=0,sum=np.zeros(12),sumsq=np.zeros(12)) for mode in ('message','control')}
def capture(layer,F,R,T,a,p,edge):
    # Copies are diagnostic only: no captured value returns to the estimator.
    F=np.asarray(F,dtype=np.float32);R=np.array(R,dtype=np.float32);T=np.array(T,dtype=np.float32)
    a=np.asarray(a);p=np.asarray(p)
    phi=np.stack((p,a*p,(a*a-1)*p),axis=1).astype(np.float32)
    np.fill_diagonal(R,0);np.fill_diagonal(T,0)
    fields=dict(message=np.concatenate((R@phi/32,(R*R)@phi,T@phi/32,T.T@phi/32),axis=1),
                control=(F/np.asarray(edge['feature_scale'],dtype=np.float32))[:,:12])
    assert layer not in current
    current[layer]={}
    for mode,x in fields.items():
        assert x.shape==(1024,12) and np.isfinite(x).all()
        x=x.astype(np.float64);s=x.sum(0);ss=(x*x).sum(0)
        current[layer][mode]=dict(count=len(x),sum=s.tolist(),sumsq=ss.tolist())
        totals[mode]['count']+=len(x);totals[mode]['sum']+=s;totals[mode]['sumsq']+=ss
mod._RNM_CAPTURE=capture
body=body.replace('def _message_edge_source(', 'def _gn_message_edge_source_base(',1)
exec(compile(body,str(HERE/'capture.py')+':instrumented_edge','exec'),mod.__dict__)
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
started=time.monotonic()
with (HERE/'capture_rows.jsonl').open('x') as fp:
    for rank in protocol['fit_ranks']:
        current.clear()
        w=np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
        t0=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
        used=int(ctx.summary_dict()['flops_used'])
        baseline=np.load(reference/f'pred_{rank}.npy')
        assert np.array_equal(y,baseline) and used==refrows[rank]['flops'],rank
        assert sorted(current)==list(range(8,15)),current.keys()
        row=dict(rank=rank,all16_bytes_exact=True,bill_exact=True,flops=used,wall_s=time.monotonic()-t0,
                 layer_statistics=current.copy(),reference_prediction_sha256=sha(reference/f'pred_{rank}.npy'),targets_accessed=False)
        fp.write(json.dumps(row)+'\n');fp.flush();records.append(dict(rank=rank,flops=used,all16_bytes_exact=True,bill_exact=True))
        print(json.dumps(dict(event='capture',rank=rank,wall_s=row['wall_s'],elapsed_s=time.monotonic()-started)),flush=True)
        del w,mlp,y,baseline;gc.collect();relief(None,0)
stats={}
for mode,s in totals.items():
    assert s['count']==12*7*1024
    mean=s['sum']/s['count'];var=np.maximum(s['sumsq']/s['count']-mean*mean,0);scale=np.maximum(np.sqrt(var),1e-6)
    stats[mode]=dict(mean=mean.astype(np.float32).tolist(),scale=scale.astype(np.float32).tolist(),count=s['count'])
result=dict(statistics=stats,records=records,targets_accessed=False,
    protocol_sha256=sha(HERE/'protocol.json'),capture_sha256=sha(HERE/'capture.py'),
    rows_sha256=sha(HERE/'capture_rows.jsonl'),parent_sha256=sha(parent),
    elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'calibration.json').open('x') as fp:json.dump(result,fp,indent=2,allow_nan=False)
print(json.dumps(dict(event='calibration_complete',elapsed_s=result['elapsed_s'],targets_accessed=False)),flush=True)
