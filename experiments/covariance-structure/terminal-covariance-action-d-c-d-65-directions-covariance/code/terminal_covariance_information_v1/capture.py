"""Passive native terminal capture. No truth inputs or numerical additions."""
import argparse,ast,hashlib,importlib.util,json,resource,sys,time,types
from pathlib import Path

HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def observed_source():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    tree=ast.parse(s);patches=[];proof=[]
    for name,old,new in [
        ('_predict_binary','                k3 = k3 + 3 * active_v * bpre',
         '                _TCI_TAP.update(k3base=k3,bpre=bpre,radial=radial,active_v=active_v)\n                k3 = k3 + 3 * active_v * bpre'),
        ('gnn_step','    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)',
         '    if l == 15:\n        _TCI_TAP.update(m1=m1,m2=m2,m3=m3,e1=e1,e2=e2,e3=e3,normalized_x=x)\n    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)')]:
        node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        original=ast.get_source_segment(s,node);assert original.count(old)==1
        changed=original.replace(old,new,1);assert changed.replace(new,old,1)==original
        ast.parse(changed);patches.append(changed)
        proof.append(dict(function=name,original_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_sha256=hashlib.sha256(changed.encode()).hexdigest(),inverse_byte_exact=True))
    return '\n\n'.join(patches),proof

def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(t['num_threads']==1 for t in threadpool_info())
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args()
    args.out.mkdir(exist_ok=False)
    seal=json.loads((HERE/'sources_sealed.json').read_text())
    for x in seal['sources']+seal['external_pins']:assert sha(x['path'])==x['sha256'],x['path']
    unit=json.loads((HERE/'unit.json').read_text());assert unit['passed']
    source,proof=observed_source()
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('tci_lazy_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    rows=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    baselines=json.loads((D/'lazy_rotation_v1/dev/result.json').read_text())['records']
    held={};layer=[None];records=[];previous=None
    for call,rank in enumerate([35,100,35]):
        wp=(STORE/rows[rank]['path']).parent/'weights.npy';weights=np.load(wp)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        held.clear();started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._TCI_TAP=held
                exec(compile(source,str(HERE/'observed.py'),'exec'),mod.__dict__)
                original_cov,original_scalar,original_step,original_silu=mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step,mod.silu
                def cov(w,c,indices,alive):
                    held.update(W=w,C=c,indices=indices,alive=alive)
                    return original_cov(w,c,indices,alive)
                def scalar(mu,v,k3,k4):
                    out=original_scalar(mu,v,k3,k4);held.update(mu=mu,v=v,k3=k3,k4=k4,host=out[0])
                    return out
                def step(l,x,sigma,w,state,p):
                    layer[0]=l
                    if l==15:held.update(H14=state,sigma=sigma,P=p)
                    out=original_step(l,x,sigma,w,state,p)
                    if l==15:held.update(H15=out[1],delta=out[0])
                    layer[0]=None
                    return out
                def silu(x):
                    if layer[0]==15:held['pre1' if 'pre1' not in held else 'pre2']=x
                    return original_silu(x)
                mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step,mod.silu=cov,scalar,step,silu
                estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        elapsed=time.monotonic()-started;flop=ctx.summary_dict()['flops_used']
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        arrays={k:np.asarray(v).copy() for k,v in held.items() if k!='P'}
        arrays.update({'P_'+k:np.asarray(v).copy() for k,v in held['P'].items()})
        arrays['prediction']=np.asarray(prediction).copy()
        for k,v in arrays.items():assert np.isfinite(v).all(),k
        assert arrays['prediction'].shape==(16,1024) and arrays['prediction'].dtype==np.float32
        assert (arrays['v']>=1e-10).all() and (arrays['prediction']>=0).all()
        baseline=baselines[call];assert baseline['rank']==rank and baseline['weights_sha256']==sha(wp)
        assert flop==baseline['flops'] and np.array_equal(arrays['prediction'],np.load(baseline['output']))
        if rank==35:
            if previous is None:previous=arrays
            else:assert all(np.array_equal(v,previous[k]) for k,v in arrays.items())
        out=args.out/f'call_{call:02d}.npz';np.savez(out,**arrays)
        records.append(dict(call=call,rank=rank,capture=str(out),capture_sha256=sha(out),capture_bytes=out.stat().st_size,parent_flops=flop,parent_wall_s=elapsed,input_flops=ingress.summary_dict()['flops_used'],weights_sha256=sha(wp),parent_all16_and_bill_exact=True,numeric_cache_retired=True))
        print(json.dumps(records[-1]),flush=True);held.clear();del prediction,weights,ww,mlp
    assert sum(r['capture_bytes'] for r in records)<100_000_000
    result=dict(completed=True,records=records,source_proof=proof,parent_sha256=PARENT_SHA,repeat_all_captured_arrays_exact=True,truth_accessed=False,extra_parent_operations=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (args.out/'result.json').open('x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
