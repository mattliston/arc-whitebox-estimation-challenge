"""Observe the unchanged Lazy parent and separately meter the fixed features."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
CANDIDATE = D / 'lazy_rotation_objective_v1/candidate.py'
SHA = 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def observed_source():
    source = CANDIDATE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SHA
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary']
    original = ast.get_source_segment(source, nodes[-1])
    before = '            fnp.fill_diagonal(target, td)'
    after = before + '\n            if l == 14:\n                _TWOHOP_TAP["third_plane"] = target'
    assert original.count(before) == 1
    changed = original.replace(before, after, 1)
    assert changed.replace(after, before, 1) == original
    ast.parse(changed)
    return changed, {'original_predict_sha256':hashlib.sha256(original.encode()).hexdigest(),
                     'observed_predict_sha256':hashlib.sha256(changed.encode()).hexdigest(),
                     'inverse_byte_exact':True,'extra_numeric_operations_inside_parent':0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--ranks', required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature import twohop
    assert all(t['num_threads'] == 1 for t in threadpool_info())
    certificate = json.loads((HERE / 'certificate.json').read_text())
    assert certificate['passed'] and certificate['feature_sha256'] == sha(HERE / 'feature.py')
    observed, proof = observed_source()
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('twohop_lazy_parent', CANDIDATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    manifest = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    ranks = list(map(int,args.ranks.split(',')))
    baseline = (json.loads((D / 'lazy_rotation_v1/dev/result.json').read_text())['records']
                if ranks == [35,100,35] else None)
    records, previous, held = [], {}, {}
    for call, rank in enumerate(ranks):
        wp = (STORE / manifest[rank]['path']).parent / 'weights.npy'
        weights = np.load(wp)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
            ww = [fnp.asarray(weights[l]) for l in range(16)]
        mlp = types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        held.clear()
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call == 0:
                spec.loader.exec_module(mod)
                mod._TWOHOP_TAP = held
                exec(compile(observed,str(HERE / 'observed_predict.py'),'exec'),mod.__dict__)
                original_cov, original_scalar, original_step = mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step

                def cov(w,c,indices,alive):
                    assert 'covariance' not in held
                    held['covariance'] = c
                    return original_cov(w,c,indices,alive)

                def scalar(*a,**kw):
                    answer = original_scalar(*a,**kw)
                    held['pdf'],held['alpha'] = answer[1]['pdf'],answer[1]['alpha']
                    return answer

                def step(layer,x,sigma,w,state,p):
                    answer = original_step(layer,x,sigma,w,state,p)
                    if layer == 15:
                        held.update(hidden=answer[1],sigma=sigma,weights=w)
                    return answer

                mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step = cov,scalar,step
                estimator = mod.Estimator()
            prediction = estimator.predict(mlp,2**41)
        parent_wall = time.monotonic()-started
        parent_flops = ctx.summary_dict()['flops_used']
        assert set(held) == {'third_plane','covariance','pdf','alpha','hidden','sigma','weights'}
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        with flops.BudgetContext(flop_budget=9_000_000_000,quiet=True) as feature_ctx:
            extra = twohop(fnp,held['covariance'],held['third_plane'],held['weights'],held['sigma'],held['alpha'],held['pdf'])
        feature_flops = feature_ctx.summary_dict()['flops_used']
        assert feature_flops < 9_000_000_000
        arrays = {k:np.asarray(held[k],dtype=np.float32).copy() for k in ['hidden','sigma']}
        arrays['prediction'] = np.asarray(prediction,dtype=np.float32).copy()
        arrays['twohop'] = np.asarray(extra,dtype=np.float32).copy()
        for key,shape in [('hidden',(1024,64)),('sigma',(1024,)),('prediction',(16,1024)),('twohop',(1024,10))]:
            assert arrays[key].shape == shape and np.isfinite(arrays[key]).all(),key
        assert (arrays['prediction'] >= 0).all() and (arrays['sigma'] > 0).all()
        if baseline is not None:
            parent = baseline[call]
            assert rank == parent['rank'] and sha(wp) == parent['weights_sha256']
            assert np.array_equal(arrays['prediction'],np.load(parent['output']))
            assert parent_flops == parent['flops'],(call,parent_flops,parent['flops'])
        if rank in previous:
            assert all(np.array_equal(a,previous[rank][k]) for k,a in arrays.items())
        previous[rank] = arrays
        target = args.out / f'call_{call:02d}.npz'
        np.savez(target,**arrays)
        rec = {'call':call,'rank':rank,'weights_sha256':sha(wp),'candidate_sha256':SHA,
               'capture':str(target.resolve()),'capture_sha256':sha(target),
               'parent_flops':parent_flops,'parent_multiplier':max(.1,parent_flops/2**41),
               'includes_initialization':call==0,'parent_wall_s':parent_wall,
               'feature_flops':feature_flops,'feature_wall_s':feature_ctx.summary_dict()['wall_time_s'],
               'input_flops':ingress.summary_dict()['flops_used'],'truth_accessed':False,
               'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        records.append(rec)
        print(json.dumps(rec),flush=True)
        held.clear()
        del prediction,extra,weights,ww,mlp
    result = {'completed':True,'source_sha256':sha(__file__),'feature_source_sha256':sha(HERE/'feature.py'),
              'protocol_sha256':sha(HERE/'PROTOCOL.md'),'source_proof':proof,'records':records,
              'repeat_byte_exact':ranks==[35,100,35],
              'parent_all16_and_flops_exact':baseline is not None,'truth_accessed':False}
    with (args.out/'completed.json').open('x') as f:json.dump(result,f,indent=2)


if __name__ == '__main__':
    main()
