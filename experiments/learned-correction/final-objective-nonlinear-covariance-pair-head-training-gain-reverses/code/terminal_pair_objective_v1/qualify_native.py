import sys,types,time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *
verify();q=read(HERE/'torch_qualification_v3.json');assert q['passed']
guard=read(HERE/'torch_qualification_v3_guard.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
for x in read(HERE/'qualification_seal.json')['sources']:assert pin(x['path'])==x
assert pin(HERE/'qualification_directions_v3.npz')==q['directions']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('terminal_pair_native');mod.__file__=str(PARENT);sys.modules[mod.__name__]=mod
with flops.BudgetContext(flop_budget=2**41,quiet=True) as setup:
    exec(compile(PARENT.read_text(),str(PARENT),'exec'),mod.__dict__)
    arena=fnp.empty((mod._PACKED_ELEMENTS,),dtype=fnp.float32)
    mod._SCRATCH=[arena,mod._cp3_bind_arena_views(arena,mod._PACKED_PUSH),mod._cp3_bind_arena_views(arena,mod._PACKED_READ),None,None,None,None]
    mod._bind_push6(arena)
    with np.load(HERE/'parent_parameters.npz') as z:P={k:fnp.asarray(z[k].copy()) for k in z.files}
calls=[];rows=[];saved={};start=time.monotonic()
def native(a,C,label):
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        W=fnp.asarray(a['W'].copy());CC=fnp.asarray(C.copy());H=fnp.asarray(a['F'][:,30:].copy())
        v,av=mod._flms_covariance_diagonal(W,CC)
        k3=fnp.asarray(a['cut'])+(3*av)*fnp.asarray(a['bpre'])
        k4=(fnp.asarray(a['radial'])*v)*v
        m,aux=mod.nonlinear_scalar(fnp.asarray(a['mu']),v,k3,k4)
        x=mod.final_scalar_features(15,m,aux,W)
        delta,_=mod.gnn_step(15,x,aux['sigma'],W,H,P)
        y=fnp.asarray(fnp.maximum(m+delta.astype(fnp.float64),0.),dtype=fnp.float32)
    ans=np.array(y,dtype=np.float64);vv=np.array(v)
    assert np.isfinite(ans).all() and np.all(vv>0)
    calls.append(dict(label=label,flops=int(ctx.summary_dict()['flops_used'])))
    saved[label]=ans
    return ans,vv
def roster(mu,v):
    z=np.abs(mu/np.sqrt(np.maximum(v,1e-10)))
    count=min(len(z),((np.count_nonzero(z<=3.5)+31)//32)*32)
    return set(np.argsort(z,kind='stable')[:count].tolist())
with np.load(HERE/'qualification_directions_v3.npz') as directions:
    for rank in TRAIN[:2]:
        with np.load(HERE/'capture_00'/f'row_{rank:04d}.npz') as z:a={k:z[k] for k in z.files}
        zero,v0=native(a,a['C'],f'{rank}_zero')
        gap=np.abs(zero-a['prediction'][-1]);bound=2*np.abs(np.spacing(a['prediction'][-1].astype(np.float32))).astype(np.float64)+1e-9
        zero_ok=bool(np.all(gap<=bound));tests=[]
        for j in range(2):
            D=directions[f'{rank}_{j}_D']
            for amp in (1.,.5):
                yp,vp=native(a,a['C']+amp*D,f'{rank}_{j}_{amp}_plus')
                ym,vm=native(a,a['C']-amp*D,f'{rank}_{j}_{amp}_minus')
                native_slope=(yp-ym)/(2*amp)
                shadow_slope=(directions[f'{rank}_{j}_{amp}_plus']-directions[f'{rank}_{j}_{amp}_minus'])/(2*amp)
                rel=float(np.linalg.norm(native_slope-shadow_slope)/max(np.linalg.norm(shadow_slope),1e-20))
                stable=roster(a['mu'],v0)==roster(a['mu'],vp)==roster(a['mu'],vm)
                tests.append(dict(direction=j,amplitude=amp,relative_rms=rel,native_response_rms=float(np.sqrt(np.mean(native_slope**2))),rowcut_membership_stable=stable,passed=rel<=.05 and stable))
        row=dict(rank=rank,zero_max_abs=float(gap.max()),zero_exact=bool(np.array_equal(zero,a['prediction'][-1])),zero_within_bound=zero_ok,tests=tests)
        rows.append(row);print(json.dumps(row),flush=True)
with (HERE/'qualification_native_outputs.npz').open('xb') as f:np.savez(f,**saved)
passed=all(r['zero_within_bound'] and all(t['passed'] for t in r['tests']) for r in rows)
once(HERE/'qualification.json',dict(passed=passed,rows=rows,native_final_replays=calls,setup_flops=int(setup.summary_dict()['flops_used']),torch=pin(HERE/'torch_qualification_v3.json'),targets_accessed=False,wall_s=time.monotonic()-start,
    scope='Current native final block with fixed CP/star values. Padded rowcut membership is checked. Full trained native estimator transfer remains mandatory.'))
print(json.dumps(dict(passed=passed,native_replays=len(calls))),flush=True)
