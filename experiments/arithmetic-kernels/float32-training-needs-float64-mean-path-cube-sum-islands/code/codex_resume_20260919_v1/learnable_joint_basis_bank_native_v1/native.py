"""Fixed paid estimator calls; all diagnostic checks are after their scopes."""
import argparse
import gc
import importlib.metadata
import importlib.util
import os
import resource
import time
import types
import weakref
from common import *
from retirement import Allocator, rss

allocator = Allocator()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jr_diagnostics(frames, mod, np):
    # Reference-only reductions after the paid prediction. Nothing here feeds
    # back into the estimator or its bill, and no target data is available.
    out=[]
    for layer,H,variance,projected,z,covariance in frames:
        h=np.asarray(H).astype(np.float64)
        v=np.asarray(variance).astype(np.float64)
        projection=np.asarray(projected).astype(np.float64)
        x=np.asarray(z).astype(np.float64).reshape(1024,16,3)
        cv=np.asarray(covariance)
        assert h.shape==(1024,64) and projection.shape==(1024,64) and v.shape==(1024,)
        assert cv.shape==(1024,1024) and np.isfinite(x).all() and np.isfinite(cv).all()
        assert np.isfinite(v).all() and v.min()>=1e-10
        sigma=np.sqrt(v)
        a,b,c=x[:,:,0],x[:,:,1],x[:,:,2]
        q=np.stack((a*a,2*a*b,2*a*c,b*b,2*b*c,c*c),axis=2)
        third=np.sum((q@np.asarray(mod._jr_K3,dtype=np.float64))*x,axis=(1,2))/(v*sigma)
        fourth=np.sum((q@np.asarray(mod._jr_K4,dtype=np.float64))*q,axis=(1,2))/(v*v)
        relative_variance=np.diag(cv).astype(np.float64)/v
        assert float(relative_variance.min())>=0 and float(relative_variance.max())<=.00875*(1+3e-5)
        normalized=x/sigma[:,None,None]
        out.append(dict(layer=layer,previous_hidden_identity_and_bytes_verified=True,
            source_columns=48,hidden_rms=float(np.sqrt(np.mean(h*h))),
            projected_channel_rms=np.sqrt(np.mean(projection*projection,axis=0)).tolist(),
            normalized_scaled_factor_rms=np.sqrt(np.mean(normalized*normalized,axis=(0,1))).tolist(),
            source_relative_variance_max=float(relative_variance.max()),
            source_relative_variance_mean=float(relative_variance.mean()),
            source_normalized_third_rms=float(np.sqrt(np.mean(third*third))),
            source_normalized_third_absmax=float(np.max(np.abs(third))),
            source_normalized_fourth_rms=float(np.sqrt(np.mean(fourth*fourth))),
            source_normalized_fourth_absmax=float(np.max(np.abs(fourth))),
            reductions_after_paid_context=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['control'])
    ap.add_argument('role', choices=['disabled','zero','sentinel','identity_sentinel','basis_zero','physical_only','head_only'])
    ap.add_argument('--rank', type=int)
    args = ap.parse_args()
    stage_ready(args.stage)
    admitted = admission()
    meta = read(HERE/'inputs.json')
    assert args.stage == 'control'
    if args.rank is None:
        ranks, name = [35,100,35], f'native_{args.role}'
    else:
        assert (args.role,args.rank) in [('zero',100),('physical_only',35),('head_only',35),('basis_zero',35)]
        ranks, name = [args.rank], f'native_{args.role}_cold{args.rank}'
    source = admitted[args.role]
    check(source)
    out = HERE/name
    out.mkdir(exist_ok=False)
    assert os.getpriority(os.PRIO_PROCESS,0) >= 10
    assert all(os.environ.get(k) == '1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
        'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'])
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert np.__version__ == '2.4.6' and importlib.metadata.version('flopscope') == '0.12.0'
    assert all(t['num_threads'] == 1 for t in threadpool_info())
    old = load_module('downstream_qualified_lifecycle', A/'birth_direction_fit_v1/native_v2.py')
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('downstream_current_p', source['path'])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    variances, selectors, records = [], [], []
    hidden_owner, gnn_states, gnn_layers = {}, [], []
    jr_source_frames = []
    array_hash = lambda x: hashlib.sha256(x.tobytes()).hexdigest()

    for index, rank in enumerate(ranks):
        wp = meta['weights'][str(rank)]
        check(wp)
        weights = np.load(wp['path'], allow_pickle=False)
        assert weights.shape == (16,1024,1024) and weights.dtype == np.float32 and np.isfinite(weights).all()
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ingress:
            ww = [fnp.array(w, dtype=fnp.float32) for w in weights]
        input_record = dict(index=index, rank=rank, **old.metrics(ingress, started),
            charged_to_participant=False, exact_bytes=all(w.tobytes() == weights[l].tobytes() for l,w in enumerate(ww)),
            owned_fnp_float32=all(isinstance(w,fnp.ndarray) and w.dtype == np.float32 and w.shape == (1024,1024) for w in ww),
            independent_buffers=all(not np.may_share_memory(np.asarray(w),weights) for w in ww),
            weight=wp, context_closed=True)
        input_bill = out/f'call_{index:02d}_input_bill.json'
        write(input_bill,input_record)
        ref = weakref.ref(ingress)
        input_reset = old.reset_saved_history(flops,ingress,input_bill)
        del ingress
        gc.collect()
        input_reset.update(context_released=ref() is None, complete=True, index=index, rank=rank)
        input_reset_path = out/f'call_{index:02d}_input_reset.json'
        write(input_reset_path,input_reset)
        assert input_record['flops'] == 16*1024*1024 and input_record['wall_s'] < 120
        assert input_record['exact_bytes'] and input_record['owned_fnp_float32'] and input_record['independent_buffers']
        assert input_reset['closed_context_bill_unchanged_by_reset'] and input_reset['context_released']
        # Same public protocol-3.0 derivation as the pinned official seed source.
        seed = 0
        mlp = types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=seed,name=f'rank{rank}')
        variances.clear()
        selectors.clear()
        hidden_owner.clear(); gnn_states.clear(); gnn_layers.clear(); jr_source_frames.clear()
        start = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
            if index == 0:
                spec.loader.exec_module(mod)
                original_gaussian, original_select = mod.gaussian_raw, mod.select
                def gaussian(mu,v):
                    variances.append(v)
                    return original_gaussian(mu,v)
                def select(score,floor):
                    answer = original_select(score,floor)
                    selectors.append((score,floor,answer))
                    return answer
                mod.gaussian_raw, mod.select = gaussian, select
                original_gnn = mod.gnn_step
                def observed_gnn(layer,x,sigma,w,state,p):
                    incoming_sha=array_hash(np.asarray(state))
                    if layer:
                        assert hidden_owner['layer']==layer-1 and hidden_owner['state_id']==id(state)
                        assert hidden_owner['state_sha']==incoming_sha
                    else:
                        assert not hidden_owner
                    answer=original_gnn(layer,x,sigma,w,state,p)
                    outgoing_sha=array_hash(np.asarray(answer[1]))
                    hidden_owner.update(layer=layer,state_id=id(answer[1]),state_sha=outgoing_sha)
                    gnn_layers.append(layer)
                    gnn_states.append(dict(layer=layer,incoming_state_sha=incoming_sha,
                        outgoing_state_sha=outgoing_sha,delta_sha=array_hash(np.asarray(answer[0]))))
                    return answer
                mod.gnn_step=observed_gnn
                original_jr_source = mod._jr_causal_source
                def observed_jr_source(backend, mm, wt, H, variance, theta, phi, basis):
                    layer = 8 + len(jr_source_frames)
                    assert hidden_owner['layer'] == layer-1 and hidden_owner['state_id'] == id(H)
                    assert hidden_owner['state_sha'] == array_hash(np.asarray(H))
                    projected = []
                    def observed_mm(left, right):
                        answer = mm(left, right)
                        if right is H:
                            projected.append(answer)
                        return answer
                    answer = original_jr_source(backend, observed_mm, wt, H, variance, theta, phi, basis)
                    assert len(projected) == 1
                    jr_source_frames.append((layer, H, variance, projected[0], answer[0][0], answer[1]))
                    return answer
                mod._jr_causal_source = observed_jr_source
                estimator = mod.Estimator()
            prediction = estimator.predict(mlp,2**41)
        metrics = old.metrics(ctx,start)
        source_diagnostics = jr_diagnostics(jr_source_frames,mod,np)
        assert [x['layer'] for x in source_diagnostics] == (list(range(8,15)) if mod._JR_ENABLED else [])
        bill_path = out/f'call_{index:02d}_bill.json'
        write(bill_path,dict(**metrics, includes_initialization=index==0,
            multiplier=max(.1,metrics['flops']/2**41), complete_counted_predict=True,
            initialization_subtracted=False, extra_participant_array_operations=0,
            input_flops_separate=input_record['flops']))
        context_ref = weakref.ref(ctx)
        lifecycle = old.reset_saved_history(flops,ctx,bill_path)
        del ctx
        gc.collect()
        lifecycle.update(context_released=context_ref() is None, complete=True, index=index, rank=rank)
        lifecycle_path = out/f'call_{index:02d}_lifecycle.json'
        write(lifecycle_path,lifecycle)
        old.guard(metrics)
        assert lifecycle['closed_context_bill_unchanged_by_reset'] and lifecycle['context_released']
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert len(variances) == 16 and len(selectors) == 45
        domains, cells = {}, {}
        for layer,v in enumerate(variances):
            x = np.asarray(v)
            assert x.shape == (1024,) and np.isfinite(x).all() and x.min() >= 1e-10
            domains[str(layer)] = dict(minimum=float(x.min()),finite=True,sha256=array_hash(x))
        for k,(score,floor,(ids,mask)) in enumerate(selectors):
            s,ix,live = np.asarray(score),np.asarray(ids),np.asarray(mask)
            assert s.shape == live.shape == (1024,) and np.isfinite(s).all() and floor == .01
            count = min(1024,((int(np.count_nonzero(s >= floor))+31)//32)*32)
            expected_ix = np.argsort(-s,kind='stable')[:count]
            expected_live = np.zeros(1024,dtype=bool)
            expected_live[expected_ix] = True
            assert ix.shape == (count,) and np.array_equal(ix,expected_ix) and np.array_equal(live,expected_live)
            cells[f'{k//3}:{k%3}'] = dict(count=count,floor=floor,indices_sha256=array_hash(ix),
                mask_sha256=array_hash(live),score_sha256=array_hash(s))
        value = np.asarray(prediction).copy()
        assert value.shape == (16,1024) and value.dtype == np.float32 and np.isfinite(value).all() and (value >= 0).all()
        output_path = out/f'call_{index:02d}_rank{rank:04d}.npy'
        with output_path.open('xb') as f:
            np.save(f,value,allow_pickle=False)
        record = dict(index=index,rank=rank,prediction=pin(output_path),bill=pin(bill_path),weights=wp,
            input_bill=pin(input_bill),input_reset=pin(input_reset_path),lifecycle=pin(lifecycle_path),
            flops=metrics['flops'],multiplier=max(.1,metrics['flops']/2**41),includes_initialization=index==0,
            wall_s=metrics['wall_s'],domain=domains,selectors=cells,numeric_cache_retired=True,
            passive_extra_fnp_ops=0,truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gnn_layers=list(gnn_layers),gnn_states=[dict(r) for r in gnn_states],
            recurrent_state_ownership_verified=True,
            source_causal_ownership_verified=True,source_diagnostics=source_diagnostics)
        if len(ranks) > 1 and index == len(ranks)-1:
            assert rank == ranks[0] and record['prediction']['sha256'] == records[0]['prediction']['sha256']
            assert domains == records[0]['domain'] and cells == records[0]['selectors']
        write(out/f'call_{index:02d}.json',record)
        records.append(record)
        print(dict(stage=args.stage,role=args.role,index=index,rank=rank,flops=metrics['flops'],wall_s=metrics['wall_s']),flush=True)
        tracked = [weakref.ref(weights), weakref.ref(prediction)] + [weakref.ref(w) for w in ww]
        variances.clear()
        selectors.clear()
        hidden_owner.clear(); gnn_states.clear(); gnn_layers.clear(); jr_source_frames.clear()
        del prediction,value,weights,ww,mlp,v,x,score,ids,mask,s,ix,live,expected_ix,expected_live
        gc.collect()
        alive = [i for i,ref in enumerate(tracked) if ref() is not None]
        before_rss = rss()
        assert not alive, ('call objects retained', index, alive)
        saved_bill = pin(bill_path)
        released = allocator.relieve()
        after_rss = rss()
        assert pin(bill_path) == saved_bill
        write(out / f'call_{index:02d}_retirement.json', dict(complete=True,index=index,rank=rank,
            all_tracked_call_objects_released=True,tracked_objects=18,live_objects=alive,
            after_closed_saved_bill=True,saved_bill=saved_bill,saved_bill_unchanged=True,
            allocator_called=True,allocator_reported_released_bytes=released,
            before_rss_bytes=before_rss,after_rss_bytes=after_rss,
            allocator_statistics_called=False,candidate_state_reset=False))
    mod.gaussian_raw,mod.select = original_gaussian,original_select
    mod.gnn_step=original_gnn
    mod._jr_causal_source=original_jr_source
    check(source)
    write(out/'result.json',dict(completed=True,records=records,source=source,role=args.role,stage=args.stage,
        full_estimator_calls=len(records),truth_accessed=False,passive_extra_fnp_ops=0,
        all_16_domains_and_45_selectors_qualified=True,passive_observers_restored=True,
        repeat_output_domain_selector_bytes_exact=len(ranks)>1,owned_fnp_input_and_closed_context_lifecycle=True))


if __name__ == '__main__':
    main()
