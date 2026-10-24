import ast,gc,hashlib,importlib.util,resource,sys,time,types
from common import *

def observed():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA;t=ast.parse(s);functions={};proof=[]
    edits={
      '_predict_learned_compact':[(
        '        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)',
        '        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)\n        _ITP_HISTORY[l]=state')],
      '_predict_binary':[(
        '            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)',
        '            if 8 <= l <= 14:\n                _ITP_INPUT[l]=dict(mu=mu,C=Cpre,K=K21,radial=radial,H=_ITP_HISTORY[l-1])\n            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)'),(
        '            host = m.copy()',
        '            if 8 <= l <= 14:\n                _ITP_RESPONSE[l]=dict(sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"])\n            host = m.copy()'),(
        '                m = m + final_hook(l, m, aux, W[l])',
        '                _ITP_RESPONSE[l]=dict(sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"])\n                m = m + final_hook(l, m, aux, W[l])')]
    }
    for name,pairs in edits.items():
        n=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        old=ast.get_source_segment(s,n);new=old
        for a,b in pairs:assert new.count(a)==1;new=new.replace(a,b,1)
        inverse=new
        for a,b in reversed(pairs):assert inverse.count(b)==1;inverse=inverse.replace(b,a,1)
        assert inverse==old;ast.parse(new);functions[name]=new
        proof.append(dict(function=name,source_sha256=hashlib.sha256(old.encode()).hexdigest(),instrumented_sha256=hashlib.sha256(new.encode()).hexdigest(),exact_inverse=True))
    return functions,proof

def get_weights(ordinal):
    import numpy as np
    import pyarrow.parquet as pq
    sys.path.insert(0,str(P2/'data_gate'));import target_blind_mini_v2 as mini
    r=row(ordinal);e=mini.AUTHORITY['splits']['mini']['shards'][r['shard_index']]
    p=mini._attest_shard_pointer(mini.pinned_snapshot_path()/e['path'],e)
    columns=['mlp_id','mlp_name','mlp_seed','weights']
    for i,batch in enumerate(pq.ParquetFile(p).iter_batches(batch_size=1,columns=columns,use_threads=False)):
        assert batch.schema.names==columns
        if i==r['row_in_shard']:
            for k in columns[:3]:assert batch.column(batch.schema.get_field_index(k))[0].as_py()==r[k]
            W=mini._weights_view(batch).copy();break
    assert W.shape==(16,1024,1024) and W.dtype==np.float32
    wp=HERE/f'weights_mini{ordinal:04d}.npy';assert not wp.exists();np.save(wp,W)
    return W,wp,dict(identity=r,columns=columns,source_blob=str(p))

def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();resource_floor()
    assert read(HERE/'unit_result.json')['passed']
    ordinal=int(sys.argv[1]);assert ordinal in TRAIN+HELDOUT
    W,wp,provenance=get_weights(ordinal);funcs,proof=observed();records=[];reference=None;digests0=None
    modes=('plain','observed','repeat') if ordinal==57 else ('observed',)
    for mode in modes:
        history={};inputs={};response={}
        stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
        spec=importlib.util.spec_from_file_location('itp_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ing:ww=[fnp.asarray(W[l]) for l in range(16)]
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'mini{ordinal}')
        start=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            spec.loader.exec_module(mod)
            if mode!='plain':
                mod._ITP_HISTORY=history;mod._ITP_INPUT=inputs;mod._ITP_RESPONSE=response
                for code in funcs.values():exec(compile(code,str(HERE/'passive_source.py'),'exec'),mod.__dict__)
            prediction=mod.Estimator().predict(mlp,2**41)
        wall=time.monotonic()-start;bill=ctx.summary_dict()['flops_used'];prediction=np.asarray(prediction)
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert prediction.shape==(16,1024) and prediction.dtype==np.float32 and np.isfinite(prediction).all()
        if mode=='plain':reference=(prediction.copy(),bill)
        exact=None
        if reference is not None:assert np.array_equal(prediction,reference[0]) and bill==reference[1];exact=True
        arrays={f'{k}_{l:02d}':np.asarray(v) for l,d in inputs.items() for k,v in d.items()}
        arrays.update({f'{k}_{l:02d}':np.asarray(v) for l,d in response.items() for k,v in d.items()});arrays['prediction']=prediction
        if mode!='plain':
            assert set(inputs)==set(LAYERS) and set(response)==set(range(8,16))
            assert all(np.min(np.diag(arrays[f'C_{l:02d}']))>=1e-10 for l in LAYERS)
            assert all(np.isfinite(v).all() for v in arrays.values())
            digests={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
            if mode=='repeat':assert digests==digests0
            else:
                digests0=digests;op=HERE/f'capture_mini{ordinal:04d}.npz';assert not op.exists();np.savez(op,**arrays)
        pp=HERE/f'prediction_mini{ordinal:04d}_{mode}.npy';np.save(pp,prediction)
        record=dict(mode=mode,ordinal=ordinal,flops=bill,wall_s=wall,prediction=pin(pp),all16_and_bill_exact=exact,
                    ingress_flops=ing.summary_dict()['flops_used'],cache_retired=True)
        records.append(record);print(record,flush=True)
        history.clear();inputs.clear();response.clear();sys.modules.pop('itp_parent');del mod,arrays,prediction,ww,mlp;gc.collect()
    write(HERE/f'capture_mini{ordinal:04d}.json',dict(passed=True,ordinal=ordinal,weights=pin(wp),capture=pin(op),records=records,
          input_array_digests=digests0,provenance=provenance,source_proof=proof,truth_values_decoded=0,
          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
