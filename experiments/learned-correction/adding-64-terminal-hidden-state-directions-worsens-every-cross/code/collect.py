"""Root-only passive native capture. This process never opens target labels."""
import ctypes, gc, sys, time, types
from prepare_capture import *


def main():
    assert len(sys.argv)==4
    pp,c=plan(sys.argv[1],sys.argv[2]);block=int(sys.argv[3]);assert 0<=block<4
    environment(c);prior=controls(c,block)
    if block:
        job=stage(c,pp,block-1);guard(job['guard'],job['command'],240)
        previous=read(D/job['name']/'complete.json');assert previous['passed'] and previous['plan']==pp
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info
    assert np.__version__=='2.4.6' and all(x['num_threads']==1 for x in threadpool_info())
    out=D/f'capture_{block}';out.mkdir()
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    source=checked(c['parent']);mod=types.ModuleType('terminal128_parent');mod.__file__=str(source);sys.modules[mod.__name__]=mod
    exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
    held={};calls=[];terminal=[]
    old_final,old_step,old_layer=mod.final_scalar_features,mod.gnn_step,mod._e2_gnn_layer
    def final(l,m,aux,w):
        value=old_final(l,m,aux,w);assert l==15 and 'host' not in held
        held['host']=m;terminal.append(l);return value
    def layer(k,l):
        value=old_layer(k,l)
        if l==15:
            assert 'effective_gain' not in held
            held.update(effective_gain=value['gain'],rms=value['rms'],head=value['head'],head_b=value['head_b'])
        return value
    def step(l,x,sigma,w,state,p):
        value=old_step(l,x,sigma,w,state,p);calls.append(l)
        if l==15:
            assert 'hidden' not in held
            held.update(hidden=value[1],delta=value[0],sigma=sigma)
        return value
    est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    mod.final_scalar_features,mod.gnn_step,mod._e2_gnn_layer=final,step,layer
    relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
    relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
    fixtures={x['rank']:x for x in c['data']};rows=[]
    for control in prior:
        rank=control['rank'];assert not held and not calls and not terminal
        fixture=fixtures[rank];assert fixture['weights']==control['weights']
        W=np.load(checked(fixture['weights']),allow_pickle=False);assert W.shape==(16,1024,1024)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[W[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
        begin=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
            result=est.predict(mlp,2**41)
        bill=int(budget.summary_dict()['flops_used']);native_wall=time.monotonic()-begin
        # All copying, reconstruction, and saved-control comparison is outside the native bill.
        prediction=np.asarray(result,dtype=np.float64).copy();del result
        packet={k:np.asarray(v).copy() for k,v in held.items()};held.clear()
        k=mod._E2_GNN_STATE[1024]
        packet['h126']=np.asarray(k['H']).copy()
        for key in ('enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b'):
            packet[key]=np.asarray(k['P'][key]).copy()
        assert calls==list(range(16)) and terminal==[15]
        assert packet['h126'].shape==(1024,126) and packet['h126'].dtype==np.float32
        assert packet['hidden'].shape==(1024,64)
        h=packet['h126']@packet['enc0_w'].T+packet['enc0_b'];h=h/(1+np.exp(-h))
        h=h@packet['enc2_w'].T+packet['enc2_b'];h=h/(1+np.exp(-h));packet['h128']=h
        state=h@packet['enc4_w'].T+packet['enc4_b']
        delta=((packet['sigma'].astype(np.float32)*packet['rms'])*(state@packet['head']+packet['head_b']))*packet['effective_gain']
        expected=np.maximum(packet['host'].astype(np.float64)+delta.astype(np.float64),0).astype(np.float32)
        assert state.tobytes()==packet['hidden'].tobytes() and delta.tobytes()==packet['delta'].tobytes()
        assert expected.astype(np.float64).tobytes()==prediction[-1].tobytes()
        assert prediction.shape==(16,1024) and np.isfinite(prediction).all() and np.min(prediction)>=0
        assert all(np.isfinite(v).all() for v in packet.values())
        assert getattr(mod,'_W5_PORT_SCOPE',None) is None
        last=getattr(mod,'_W5_PORT_LAST',None)
        if last is not None:assert last['completed'] and last['numeric_cache_retired']
        with np.load(checked(control['packet']),allow_pickle=False) as old:
            assert prediction.tobytes()==old['prediction'].tobytes() and bill==control['flops']
            for key in ('host','effective_gain','rms','head','head_b','hidden','delta','sigma'):
                assert packet[key].dtype==old[key].dtype and packet[key].shape==old[key].shape and packet[key].tobytes()==old[key].tobytes(),key
        packet['prediction']=prediction;p=out/f'packet_{rank}.npz'
        with p.open('xb') as f:np.savez(f,**packet)
        row=dict(rank=rank,weights=fixture['weights'],source=c['parent'],flops=bill,multiplier=max(.1,bill/2**41),
            packet=pin(p),control=control['packet'],all16_finite=True,targets_accessed=False,wall_s=native_wall,
            independent_all16_and_bill_exact=True,terminal_state_and_delta_replay_exact=True,readout_replay_exact=True)
        rows.append(row);once(out/f'row_{rank}.json',row);print(json.dumps(row),flush=True)
        calls.clear();terminal.clear();del W,mlp,packet,prediction,h,state,delta,expected,k;gc.collect();relief(None,0)
    once(out/'complete.json',dict(passed=True,ranks=c['blocks'][block],rows=rows,source=c['parent'],plan=pp,
        targets_accessed=False,capture_references_retired=True))


if __name__=='__main__':main()
