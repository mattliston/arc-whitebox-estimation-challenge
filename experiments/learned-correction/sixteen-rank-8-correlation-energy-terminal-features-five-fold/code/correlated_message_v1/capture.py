"""Observe unchanged parent, then meter the new information separately."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SHA='5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('rank',type=int)
    parser.add_argument('out',type=Path)
    parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=False)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature import correlated_energy
    assert all(t['num_threads']==1 for t in threadpool_info())
    candidate=D/'cost_rule75_scalar.py'
    assert sha(candidate)==SHA
    stub=types.ModuleType('whestbench')
    stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('cm_parent',candidate)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    manifest=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    wp=(STORE/manifest[args.rank]['path']).parent/'weights.npy'
    weight_sha=sha(wp)
    records=[]
    held={}
    previous_arrays=None
    for call in range(2 if args.repeat else 1):
        weights=np.load(wp)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
            ww=[fnp.asarray(weights[l]) for l in range(16)]
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{args.rank}')
        held.clear()
        started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod)
                original_cov=mod._cov_compact_diagonal
                original_step=mod.gnn_step
                original_scalar=mod.nonlinear_scalar

                def cov(w,c,indices,alive):
                    assert 'covariance' not in held
                    held['covariance']=c
                    return original_cov(w,c,indices,alive)

                def scalar(*a,**kw):
                    answer=original_scalar(*a,**kw)
                    held['pdf']=answer[1]['pdf']
                    return answer

                def step(layer,x,sigma,w,state,p):
                    answer=original_step(layer,x,sigma,w,state,p)
                    if layer==15:
                        held.update(state14=state,state15=answer[1],sigma=sigma,weights=w,parameters=p)
                    return answer

                mod._cov_compact_diagonal=cov
                mod.nonlinear_scalar=scalar
                mod.gnn_step=step
                estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        parent_wall=time.monotonic()-started
        parent_flops=ctx.summary_dict()['flops_used']
        assert set(held)=={'covariance','pdf','state14','state15','sigma','weights','parameters'}
        with flops.BudgetContext(flop_budget=1_000_000_000,quiet=True) as feature_ctx:
            extra=correlated_energy(fnp,held['covariance'],held['state14'],held['weights'],held['parameters'],held['pdf'])
        feature_flops=feature_ctx.summary_dict()['flops_used']
        assert feature_flops<1_000_000_000
        arrays=dict(prediction=np.asarray(prediction,dtype=np.float32).copy(),
                    hidden=np.asarray(held['state15'],dtype=np.float32).copy(),
                    sigma=np.asarray(held['sigma'],dtype=np.float32).copy(),
                    correlated=np.asarray(extra,dtype=np.float32).copy())
        assert arrays['prediction'].shape==(16,1024) and arrays['hidden'].shape==(1024,64)
        assert arrays['correlated'].shape==(1024,16)
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert (arrays['prediction']>=0).all() and (arrays['sigma']>0).all()
        if args.rank==35:
            parent=json.loads((D/'cost75_dev_v1/result.json').read_text())['records'][0 if call==0 else 2]
            assert np.array_equal(arrays['prediction'],np.load(parent['output']))
            if call==0:
                assert parent_flops==parent['flops']
        if previous_arrays is not None:
            assert all(np.array_equal(a,previous_arrays[k]) for k,a in arrays.items())
        npz=args.out/f'call_{call:02d}.npz'
        np.savez(npz,**arrays)
        rec=dict(call=call,rank=args.rank,weights_sha256=weight_sha,candidate_sha256=SHA,
            capture=str(npz.resolve()),capture_sha256=sha(npz),parent_flops=parent_flops,
            parent_multiplier=max(.1,parent_flops/2**41),includes_initialization=call==0,
            feature_flops=feature_flops,parent_wall_s=parent_wall,
            feature_wall_s=feature_ctx.summary_dict()['wall_time_s'],
            input_flops=ingress.summary_dict()['flops_used'],truth_accessed=False,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(rec)
        print(json.dumps(rec),flush=True)
        previous_arrays=arrays
        held.clear()
        del prediction,extra,weights,ww,mlp
    result=dict(completed=True,source_sha256=sha(__file__),feature_source_sha256=sha(HERE/'feature.py'),
                protocol_sha256=sha(HERE/'PROTOCOL.md'),records=records,
                repeat_byte_exact=bool(args.repeat),truth_accessed=False)
    with (args.out/'completed.json').open('x') as f:
        json.dump(result,f,indent=2)


if __name__=='__main__':
    main()
