import ast,gc,hashlib,importlib.util,json,resource,sys,time,types
from common import *

def observed():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    tree=ast.parse(s);functions={};proof=[]
    edits_by_name={
      '_predict_binary':[
       ('                m = m + final_hook(l, m, aux, W[l])',
        '                m = m + final_hook(l, m, aux, W[l])\n                _CIS_TAP[l] = dict(m=m,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"])'),
       ('            previous, previous_alive = (live_indices, alive)',
        '            if l >= 7:\n                _CIS_TAP[l] = dict(m=m,C_post=C,C_pre=Cpre,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"])\n            previous, previous_alive = (live_indices, alive)')],
      '_edge_source':[
       ("    raw=(fnp.tanh(h)@t['output'])*pf[:,None]*gf[None,:]",
        "    _CIS_EDGE[layer] = dict(F=F,R=R,S=S,pf=pf,gf=gf,sf=sf)\n    raw=(fnp.tanh(h)@t['output'])*pf[:,None]*gf[None,:]"),
       ("    return (0.5*(raw+raw.T)*(sf[:,None]*sf[None,:])*t['cov_factor']).astype(fnp.float64)",
        "    result = (0.5*(raw+raw.T)*(sf[:,None]*sf[None,:])*t['cov_factor']).astype(fnp.float64)\n    _CIS_EDGE[layer]['correction'] = result\n    return result")]
    }
    for name,edits in edits_by_name.items():
        n=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name][-1]
        old=ast.get_source_segment(s,n);new=old
        for a,b in edits:
            assert new.count(a)==1
            new=new.replace(a,b,1)
        back=new
        for a,b in reversed(edits):assert back.count(b)==1;back=back.replace(b,a,1)
        assert back==old;ast.parse(new);functions[name]=new
        proof.append(dict(function=name,original_sha256=hashlib.sha256(old.encode()).hexdigest(),observed_sha256=hashlib.sha256(new.encode()).hexdigest(),exact_inverse=True))
    return functions,proof

def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify()
    assert read(HERE/'unit.json')['passed']
    name=sys.argv[1];ranks=list(map(int,sys.argv[2].split(',')));out=HERE/name;out.mkdir(exist_ok=False)
    funcs,proof=observed();tap={};edge={};recs=[];first=None
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('cis_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    baselines=read(D/'full_lower_multilayer_mean_observer_v1/parent_triple/result.json')['records']
    for call,rank in enumerate(ranks):
        wp=weights(rank);W=np.load(wp,allow_pickle=False)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ing:ww=[fnp.asarray(W[l]) for l in range(16)]
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        start=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._CIS_TAP=tap;mod._CIS_EDGE=edge
                for f in funcs.values():exec(compile(f,str(HERE/'passive_source.py'),'exec'),mod.__dict__)
                estimator=mod.Estimator()
            pred=estimator.predict(mlp,2**41)
        elapsed=time.monotonic()-start;flop=ctx.summary_dict()['flops_used']
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert set(tap)==set(range(7,16)) and set(edge)==set(range(8,15))
        arrays={f'{k}_{l:02d}':np.asarray(v) for l,d in tap.items() for k,v in d.items() if not(l==7 and k=='C_pre')}
        arrays.update({f'edge_{k}_{l:02d}':np.asarray(v) for l,d in edge.items() for k,v in d.items()})
        arrays['prediction']=np.asarray(pred)
        arrays.update({f'param_{k}':np.asarray(v) for k,v in mod._EDGE_PARAMETERS.items()})
        assert all(np.isfinite(v).all() for v in arrays.values())
        digest={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        exact=None
        if name=='qualification':
            b=baselines[call];assert b['rank']==rank and b['weights_sha256']==sha(wp)
            assert flop==b['flops'] and np.array_equal(arrays['prediction'],np.load(b['output'],allow_pickle=False));exact=True
            if call==0:first=digest
            if call==2:assert first==digest
        if name=='qualification' and call==2:
            file=Path(recs[0]['capture']['path'])
        else:
            file=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(file,**arrays)
        rec=dict(rank=rank,call=call,capture=pin(file),array_digests=digest,weights=pin(wp),flops=flop,wall_s=elapsed,ingress_flops=ing.summary_dict()['flops_used'],all16_and_bill_exact=exact,cache_retired=True)
        recs.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='array_digests'}),flush=True)
        tap.clear();edge.clear();del arrays,pred,W,ww,mlp;gc.collect()
    write(out/'result.json',dict(completed=True,records=recs,source_proof=proof,parent_sha256=PARENT_SHA,extra_parent_operations=0,truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
