"""Root-only current native passive/teacher-forced angular acquisition."""
import ctypes,gc,sys,time,types
from common import *

def main():
    assert len(sys.argv)==5
    pp,c=plan(sys.argv[1],sys.argv[2]);mode,block=sys.argv[3],int(sys.argv[4]);assert mode in ('passive','capture')
    environment(c);angular=native_prerequisite(c)
    prerequisites=replay_prerequisite(c,pp) if mode=='capture' else None
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info
    assert np.__version__=='2.4.6' and all(v['num_threads']==1 for v in threadpool_info())
    name=f'capture_{block}' if mode=='capture' else mode
    ranks=c['ranks'][5*block:5*block+5] if mode=='capture' else [c['qualification_rank']]
    assert (mode=='capture' and 0<=block<8) or (mode!='capture' and block==0)
    out=D/name;out.mkdir()
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    source=checked(c['candidate']);mod=types.ModuleType('angular_capture');mod.__file__=str(source);sys.modules[mod.__name__]=mod
    exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
    scale=c['mean_scale'];assert scale==mod._ANGULAR_MEAN_SCALE
    pending={};held=[];truth=None
    old_feature,old_final,old_step=mod.features,mod.final_scalar_features,mod.gnn_step
    def feature(l,m,C,K21,aux,w):
        x=old_feature(l,m,C,K21,aux,w);assert not pending
        pending.update(layer=l,host=m,x=x);return x
    def final(l,m,aux,w):
        x=old_final(l,m,aux,w);assert l==15 and not pending
        pending.update(layer=l,host=m,x=x);return x
    def step(l,x,sigma,w,state,P):
        delta,new_state=old_step(l,x,sigma,w,state,P)
        if l==0:
            assert not pending and x is None;host=None
        else:
            assert pending['layer']==l and pending['x'] is x;host=pending['host'];pending.clear()
        # Retain existing immutable results; copy to NumPy only after native prediction returns.
        held.append(dict(layer=l,x=x,host=host,sigma=sigma,delta=delta,hidden=new_state))
        if mode=='capture' and l>=1:
            return mod.fnp.asarray(truth[l]/scale,dtype=mod.fnp.float64)-host,new_state
        return delta,new_state
    est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    mod.features,mod.final_scalar_features,mod.gnn_step=feature,final,step
    relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief;relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
    fixtures={v['rank']:v for v in c['data']};fixtures[c['qualification_rank']]=c['qualification_fixture'];rows=[];parameters=None
    for rank in ranks:
        assert not held and not pending
        fixture=fixtures[rank];W=np.load(checked(fixture['weights']),allow_pickle=False)
        assert W.shape==(16,1024,1024) and np.isfinite(W).all()
        if mode=='capture':
            with np.load(checked(fixture['truth']),allow_pickle=False) as z:truth=z['mc_post'].astype(np.float64)
            assert truth.shape==(16,1024) and np.isfinite(truth).all()
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[W[l] for l in range(16)],seed=0,name=f'phase2-{rank}');start=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
            answer=est.predict(mlp,2**41);prediction=np.asarray(answer,dtype=np.float64).copy()
            packet=[]
            for v in held:
                packet.append({k:(np.asarray(a).copy() if a is not None and k!='layer' else a) for k,a in v.items()})
            held.clear();del answer
            if mode=='passive':P={k:np.asarray(v).copy() for k,v in mod._GNN_PARAMS.items()}
        bill=int(budget.summary_dict()['flops_used']);assert prediction.shape==(16,1024) and np.isfinite(prediction).all() and prediction.min()>=0
        assert getattr(mod,'_W5_PORT_SCOPE',None) is None
        last=getattr(mod,'_W5_PORT_LAST',None)
        if last is not None:assert last['completed'] and last['numeric_cache_retired']
        row=dict(rank=rank,source=c['candidate'],weights=fixture['weights'],flops=bill,wall_s=time.monotonic()-start,all16_finite=True,independent_all16_and_bill_exact=None)
        assert [v['layer'] for v in packet]==list(range(16))
        arrays={k:np.stack([v[k] if v[k] is not None else np.zeros((1024,30) if k=='x' else (1024,),dtype=np.float32) for v in packet]) for k in ('x','host','sigma','delta','hidden')}
        assert arrays['x'].shape==(16,1024,30) and arrays['hidden'].shape==(16,1024,64)
        assert all(np.isfinite(a).all() for a in arrays.values())
        arrays['host_gaussian']=scale*arrays['host'];arrays['sigma_gaussian']=scale*arrays['sigma'];arrays['prediction']=prediction
        if mode=='passive':
            records=[json.loads(v) for v in checked(angular['artifacts']['candidate']['rows']).read_text().splitlines()]
            b=next(v for v in records if v['rank']==rank)
            saved=np.load(checked(angular['artifacts']['candidate']['predictions'][str(rank)]),allow_pickle=False)
            assert saved.tobytes()==prediction.tobytes() and b['flops']==bill
            row['independent_all16_and_bill_exact']=True
            p=out/'parameters.npz'
            with p.open('xb') as f:np.savez(f,**P)
            parameters=pin(p);del saved,P
        else:
            row['truth']=fixture['truth'];row['teacher_forced_max_abs_mean_error']=float(np.max(np.abs(prediction[1:]-truth[1:])))
            assert row['teacher_forced_max_abs_mean_error']<=2e-7*max(1.,float(np.max(np.abs(truth[1:]))))
        p=out/f'packet_{rank}.npz'
        with p.open('xb') as f:np.savez(f,**arrays)
        row['packet']=pin(p);del arrays
        rows.append(row);once(out/f'row_{rank}.json',row);print(json.dumps(row),flush=True)
        del W,mlp,packet,prediction;truth=None;gc.collect();relief(None,0)
    once(out/'complete.json',dict(passed=True,mode=mode,ranks=ranks,rows=rows,source=c['candidate'],plan=pp,parameters=parameters,prerequisites=prerequisites,targets_accessed=mode=='capture',features_before_current_truth_feedback=True,layer0_host_features_unused_zero=True))

if __name__=='__main__':main()
