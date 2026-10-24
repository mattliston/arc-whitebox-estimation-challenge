"""Passive qualification or natural-strength native terminal-only candidate."""
import argparse,ast,importlib.util,json,sys,types,time,resource,gc,hashlib
from common import HERE,D,PARENT,PARENT_SHA,read,write,pin,sha,verify

def instrument(enabled):
    source=PARENT.read_text();node=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'][-1]
    original=ast.get_source_segment(source,node);s=original
    edits=[('        W = mlp.weights','        W = mlp.weights\n        _ttcs_packet = None'),
      ('            host = m.copy()',
       '            if l == 14 and depth == 16 and n == 1024:\n                _ttcs_packet = _TTCS_PREPARE(pp,lam,w32,star,previous_alive,Bpre,bpre,Cpre,aux["gate"],aux["pdf"],aux["sigma"],alive,C,K21)\n            host = m.copy()'),
      ('                m = m + final_hook(l, m, aux, W[l])',
       '                m = m + final_hook(l, m, aux, W[l])\n                if _ttcs_packet is not None:\n                    _ttcs_delta = _TTCS_FINISH(_ttcs_packet,w32,pp,aux,m)\n'+('                    m = m + _ttcs_delta\n' if enabled else '')+'                    del _ttcs_packet,_ttcs_delta')]
    for a,b in edits:assert s.count(a)==1;s=s.replace(a,b,1)
    restored=s
    for a,b in reversed(edits):restored=restored.replace(b,a,1)
    assert restored==original;ast.parse(s)
    return s,dict(inverse_byte_exact=True,enabled=enabled,original_sha256=hashlib.sha256(original.encode()).hexdigest(),instrumented_sha256=hashlib.sha256(s.encode()).hexdigest())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enabled',action='store_true');parser.add_argument('--cold100',action='store_true');args=parser.parse_args()
    assert not args.cold100 or args.enabled;verify();assert read(HERE/'unit.json')['passed'] and sha(PARENT)==PARENT_SHA
    if args.enabled:
        r=read(HERE/'qualification/result.json');g=read(HERE/'qualification_watch.json')
        assert r['qualified'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature_fnp import prepare,probes,feature
    from reference import reference
    assert all(x['num_threads']==1 for x in threadpool_info())
    inputs=read(HERE/'inputs.json');rows=inputs['qualification'] if not args.cold100 else [inputs['cold100']]
    out=HERE/('cold100' if args.cold100 else 'candidate' if args.enabled else 'qualification');out.mkdir(exist_ok=False)
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('ttcs_parent',PARENT);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    source,proof=instrument(args.enabled);bill=0;detail={};records=[];first=None
    def prep(*a):
        nonlocal bill
        t=ctx.summary_dict()['flops_used'];ans=prepare(*a);bill+=ctx.summary_dict()['flops_used']-t;return ans
    def finish(packet,w,pushed,aux,m):
        nonlocal bill,detail
        t=ctx.summary_dict()['flops_used'];started=time.monotonic();om=probes()
        dk4,info=feature(packet,w,pushed,om,aux['pre_v'])
        dm=dk4*((aux['alpha']*aux['alpha']-1)*aux['pdf']/(24*aux['pre_v']*aux['sigma']))
        charged=ctx.summary_dict()['flops_used']-t;bill+=charged;producer_wall=time.monotonic()-started
        detail=dict(delta_mean=np.asarray(dm).copy(),base_mean=np.asarray(m).copy(),delta_k4=np.asarray(dk4).copy(),
                    rays=np.asarray(info['rays']).copy(),radial=float(info['radial']),trace3=float(info['trace3']),producer_wall_s=producer_wall)
        # The independent reference is deliberately omitted in candidate runs.
        if not args.enabled:
            start=time.monotonic();ref,rd=reference({k:np.asarray(v) for k,v in packet.items()},np.asarray(w),np.asarray(om),np.asarray(aux['pre_v']))
            er=float(np.linalg.norm(np.asarray(dk4)-ref)/max(np.linalg.norm(ref),1e-30))
            rayrel=float(np.linalg.norm(np.asarray(info['rays'])-rd['rays'])/max(np.linalg.norm(rd['rays']),1e-30))
            trrel=abs(float(info['trace3'])-rd['trace3'])/max(abs(rd['trace3']),1e-30)
            detail.update(reference_delta_relative=er,reference_rays_relative=rayrel,reference_trace_relative=trrel,
                          tdiag_relative=rd['tdiag_relative'],reference_wall_s=time.monotonic()-start)
            assert max(er,rayrel,trrel,rd['tdiag_relative'])<=.002,detail
            assert ctx.summary_dict()['flops_used']==t+charged
        packet.clear();return dm
    for call,row in enumerate(rows):
        weights=np.load(row['weights']['path'],allow_pickle=False);assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ic:ww=[fnp.asarray(x) for x in weights]
        assert ic.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{row["rank"]}')
        bill=0;detail={};started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._TTCS_PREPARE=prep;mod._TTCS_FINISH=finish
                exec(compile(source,str(HERE/'instrumented.py'),'exec'),mod.__dict__);estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        F=ctx.summary_dict()['flops_used'];pred=np.asarray(prediction);wall=time.monotonic()-started
        parent=np.load(row['prediction']['path'],allow_pickle=False)
        assert pred[:15].tobytes()==parent[:15].tobytes() and np.isfinite(pred).all() and (pred>=0).all()
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        # Enabled adds only the final FP64 vector addition; all paid feature operations are measured above.
        remainder=F-bill-row['flops'];assert remainder==(2048 if args.enabled else 0),remainder
        if not args.enabled:assert pred.tobytes()==parent.tobytes()
        path=out/f'call_{call:02d}_rank{row["rank"]:04d}.npy';np.save(path,pred,allow_pickle=False)
        payload={k:detail.pop(k) for k in ['delta_mean','base_mean','delta_k4','rays']}
        feature_path=out/f'feature_{call:02d}_rank{row["rank"]:04d}.npz';np.savez(feature_path,**payload)
        digest={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in payload.items()}
        if call==0:first=(pred.tobytes(),digest)
        if call==2:assert (pred.tobytes(),digest)==first
        rec=dict(call=call,rank=row['rank'],prediction=pin(path),feature=pin(feature_path),feature_digests=digest,
                 whole_flops=F,parent_flops=row['flops'],extra_feature_flops=bill,final_addition_flops=remainder,
                 extra_ratio=(F-row['flops'])/row['flops'],wall_s=wall,includes_initialization=call==0,
                 first15_parent_bytes_exact=True,all16_parent_bytes_exact=not args.enabled,parent_bill_decomposition_exact=True,
                 numeric_cache_retired=True,details=detail,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(rec);print(json.dumps(rec),flush=True)
        del payload,prediction,pred,parent,weights,ww,mlp;gc.collect()
    result=dict(completed=True,qualified=all(x['extra_ratio']<=.05 for x in records),enabled=args.enabled,records=records,source_proof=proof,
                truth_accessed=False,repeat_bytes_exact=len(rows)==3,full_estimator_calls=len(rows),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'result.json',result)
if __name__=='__main__':main()
