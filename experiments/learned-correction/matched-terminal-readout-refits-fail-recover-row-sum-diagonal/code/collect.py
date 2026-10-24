"""Root-only cold-block native capture; labels never enter this process."""
import ctypes, gc, sys, time, types
from common import *


def main():
    assert len(sys.argv)==6
    pp,c=plan(sys.argv[1],sys.argv[2]);mode,arm,block=sys.argv[3],sys.argv[4],int(sys.argv[5])
    assert mode in ('control','capture') and arm in ARMS and 0<=block<4
    environment(c);prerequisites(c)
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info
    assert np.__version__=='2.4.6' and all(x['num_threads']==1 for x in threadpool_info())
    replay=load(c['sources']['readout']['path'],'fixed_readout').replay
    name=f'control_{arm}' if mode=='control' else f'capture_{arm}_{block}'
    out=D/name;out.mkdir();ranks=TRAIN[:1] if mode=='control' else TRAIN[block*5:block*5+5]
    prior=None
    if mode=='capture':
        job=stage(c,pp,f'control_{arm}');guard(job['guard'],job['command'],job['seconds'])
        prior=read(D/f'control_{arm}'/'complete.json');assert prior['passed'] and prior['source']==c['arms'][arm] and prior['plan']==pp
        checked(prior['rows'][0]['prediction'])
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    source=checked(c['arms'][arm]);mod=types.ModuleType('fresh_head_'+arm);mod.__file__=str(source);sys.modules[mod.__name__]=mod
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
    if mode=='capture':mod.final_scalar_features,mod.gnn_step,mod._e2_gnn_layer=final,step,layer
    relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
    relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
    fixtures={x['rank']:x for x in c['data']};rows=[]
    for rank in ranks:
        assert not held and not calls and not terminal
        fixture=fixtures[rank];W=np.load(checked(fixture['weights']),allow_pickle=False);assert W.shape==(16,1024,1024)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[W[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
        begin=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
            result=est.predict(mlp,2**41);prediction=np.asarray(result,dtype=np.float64).copy()
            packet={k:np.asarray(v).copy() for k,v in held.items()};held.clear();del result
        bill=int(budget.summary_dict()['flops_used'])
        assert prediction.shape==(16,1024) and np.isfinite(prediction).all() and np.min(prediction)>=0
        assert getattr(mod,'_W5_PORT_SCOPE',None) is None
        last=getattr(mod,'_W5_PORT_LAST',None)
        if last is not None:assert last['completed'] and last['numeric_cache_retired']
        row=dict(rank=rank,weights=fixture['weights'],source=c['arms'][arm],flops=bill,multiplier=max(.1,bill/2**41),
                 all16_finite=True,targets_accessed=False,wall_s=time.monotonic()-begin,independent_all16_and_bill_exact=None)
        if mode=='control':
            assert not packet
            p=out/f'pred_{rank}.npy'
            with p.open('xb') as f:np.save(f,prediction)
            row['prediction']=pin(p)
        else:
            assert calls==list(range(16)) and terminal==[15]
            assert set(packet)=={'host','effective_gain','rms','head','head_b','hidden','delta','sigma'}
            assert packet['hidden'].shape==(1024,64) and all(np.isfinite(v).all() for v in packet.values())
            head=np.r_[packet['head'],packet['head_b']].astype(np.float32);expected,delta=replay(packet,head)
            assert delta.tobytes()==packet['delta'].tobytes() and expected.astype(np.float64).tobytes()==prediction[-1].tobytes()
            if rank==TRAIN[0]:
                control=prior['rows'][0];assert control['rank']==rank and control['weights']==fixture['weights']
                unwrapped=np.load(control['prediction']['path'],allow_pickle=False)
                assert prediction.tobytes()==unwrapped.tobytes() and bill==control['flops'];del unwrapped
                row['independent_all16_and_bill_exact']=True
            packet['prediction']=prediction;p=out/f'packet_{rank}.npz'
            with p.open('xb') as f:np.savez(f,**packet)
            row.update(packet=pin(p),readout_replay_exact=True)
            del head,expected,delta
        rows.append(row);once(out/f'row_{rank}.json',row);print(__import__('json').dumps(row),flush=True)
        calls.clear();terminal.clear();del W,mlp,packet,prediction;gc.collect();relief(None,0)
    once(out/'complete.json',dict(passed=True,mode=mode,arm=arm,ranks=ranks,rows=rows,source=c['arms'][arm],plan=pp,
         targets_accessed=False,capture_references_retired=True))


if __name__=='__main__':main()
