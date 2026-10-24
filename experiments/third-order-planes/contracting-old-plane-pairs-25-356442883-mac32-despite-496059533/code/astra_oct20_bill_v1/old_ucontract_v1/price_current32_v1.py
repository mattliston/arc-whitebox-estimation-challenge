"""Exact current32 roster tariff for one fixed L10-pair cache-preserving cut.

Official tariff plus bounded metadata backend; no setup/predict/numeric GEMM.
Both original and proposed B-transform caches execute their real source logic.
V2 interleaves the unchanged u product after each v product, enforcing the real600MB combined cache cap.
"""
import argparse,collections,hashlib,importlib.util,json,resource,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
    parent=HERE.parent/'fusion_shape_frontier_v1/price_v2.py';sp=importlib.util.spec_from_file_location('vpair_metadata_parent',parent);api=importlib.util.module_from_spec(sp);sp.loader.exec_module(api)
    threading.Thread(target=api.rss_guard,daemon=True).start();mod,fn,fs,measure,provenance=api.load()
    import numpy as np
    from flopscope import _budget
    original=_budget._call_numpy
    def backend(function,*args,**kwargs):
        name=getattr(function,'__name__','')
        if name in('sum','mean'):
            x=args[0];axis=kwargs.get('axis',args[1]if len(args)>1 else None);axes=tuple(range(x.ndim))if axis is None else(axis,)if isinstance(axis,int)else tuple(axis);axes=tuple(k%x.ndim for k in axes);shape=tuple(1 if k in axes else z for k,z in enumerate(x.shape))if kwargs.get('keepdims',False)else tuple(z for k,z in enumerate(x.shape)if k not in axes);dtype=kwargs.get('dtype',args[2]if len(args)>2 else None)or x.dtype
            return np.broadcast_to(np.array(0,dtype=dtype),shape)
        if name in('sort','argsort'):
            assert args[0].ndim==1 and args[0].size<=20000
            return function(*args,**kwargs)
        return original(function,*args,**kwargs)
    for name,module in list(sys.modules.items()):
        if name.startswith('flopscope')and getattr(module,'_call_numpy',None)is original:module._call_numpy=backend
    helper=HERE/'helper_v1.py';exec(compile(helper.read_text(),str(helper),'exec'),mod.__dict__)
    rows=[];pins=[]
    for path in sorted(api.CENSUS.glob('candidate_part*/rows.jsonl')):
        assert json.loads((path.parent/'manifest.json').read_text())['sha256']==api.SOURCE_SHA;pins.append(dict(path=str(path),sha256=sha(path)))
        for line in path.read_text().splitlines():
            row=json.loads(line);shapes=[tuple(x[4:7])for x in row['product_geometries']if x[:4]==['shared_old384',0,'_ubx_product',14054]and x[6]==384]
            assert len(shapes)==5 and len({x[1]for x in shapes})==1 and all(x[2]==384 for x in shapes)
            ushapes=[tuple(x[4:7])for x in row['product_geometries']if x[:4]==['shared_old384',0,'_ubx_product',14054]and x[6]==256]
            assert len(ushapes)==5 and all(v[:2]==u[:2]for v,u in zip(shapes,ushapes))
            rows.append(dict(rank=row['rank'],flops=row['flops'],prefix_shapes=shapes,unchanged_u_shapes=ushapes))
    sequences={};reports=[]
    for row in rows:
        sequence=tuple(row['prefix_shapes']);P=sequence[0][1]
        if sequence not in sequences:
            with fs.BudgetContext(flop_budget=2**60,quiet=True):
                Fv=fn.empty((384,P),dtype=fn.float32);Fu=fn.empty((256,P),dtype=fn.float32);owner=fn.empty((384,2*P),dtype=fn.float32);inputs={m:fn.empty((m,P),dtype=fn.float32)for m,_,_ in sequence}
            uv=(Fu,Fv,[(0,P)],256);parts=[]
            for candidate in (False,True):
                mod._O21_UPAIR_CACHE.clear();mod._UBX_STATE.update(run=None,cache={},bytes=0);mod._UBC_RUN[0]=dict(owner=owner,generation=(P,384),uv=uv)
                costs=[]
                for layer,(m,_,_)in zip(range(10,15),sequence):
                    mod._UVP_CUR[0]=layer
                    call=lambda m=m:mod._ubx_product(inputs[m],Fv.T,uv,'vt')
                    actual=measure(call)
                    ucall=(lambda m=m:mod._o21_upair_inner(inputs[m],Fu,uv))if candidate else(lambda m=m:mod._ubx_product(inputs[m],Fu.T,uv,'ut'))
                    actual_u=measure(ucall)
                    # A second old-only call gives a conservative fully warmed
                    # baseline independent of layout-binding amortization.
                    warmed=actual if candidate else measure(call)
                    warmed_u=actual_u if candidate else measure(ucall)
                    assert actual['result_shape']==[m,384]
                    costs.append(dict(layer=layer,shape=[m,P,384],actual=actual,warmed=warmed,actual_u=actual_u,warmed_u=warmed_u,cache_bytes=mod._UBX_STATE['bytes']))
                if candidate:assert mod._O21_UPAIR_STATS['calls']>=5 and mod._O21_UPAIR_CACHE['pairs']==P
                parts.append(costs)
            old,new=parts;oldf=sum(x['actual']['flops']+x['actual_u']['flops']for x in old);oldw=sum(x['warmed']['flops']+x['warmed_u']['flops']for x in old);newf=sum(x['actual']['flops']+x['actual_u']['flops']for x in new)
            sequences[sequence]=dict(original=old,candidate=new,original_flops=oldf,original_fully_warm_lower_flops=oldw,candidate_flops=newf,net_saving_flops=oldf-newf,conservative_net_saving_flops=oldw-newf,charged_call_delta=sum(x['actual']['charged']+x['actual_u']['charged']for x in new)-sum(x['actual']['charged']+x['actual_u']['charged']for x in old),keep=mod._O21_UPAIR_CACHE['retained'],P=P)
        result=dict(**row,price=sequences[sequence]);result.update(bill_delta_pct=-100*result['price']['net_saving_flops']/row['flops'],conservative_bill_delta_pct=-100*result['price']['conservative_net_saving_flops']/row['flops']);reports.append(result)
    assert len(reports)==32
    receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='CURRENT777A_OFFICIAL_METADATA_FIXED_L10_U_PAIR_CUT_INTERLEAVED_V_U_V1',reference_submission=337422,source_sha256=api.SOURCE_SHA,helper_sha256=sha(helper),script_sha256=sha(__file__),price_parent_sha256=sha(parent),backend=provenance,official_version=fs.__version__,census_pins=pins,records=sorted(reports,key=lambda x:x['rank']),unique_sequences=len(sequences),mean_bill_delta_pct=sum(x['bill_delta_pct']for x in reports)/32,mean_conservative_bill_delta_pct=sum(x['conservative_bill_delta_pct']for x in reports)/32,bill_delta_range=[min(x['bill_delta_pct']for x in reports),max(x['bill_delta_pct']for x in reports)],max_charged_call_delta=max(x['price']['charged_call_delta']for x in reports),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_prediction=True,no_truth=True,numeric_products=0,caveat='Current fixedrosters only. Real right-transform cache code and a fixed firstL10 u-pairset are priced; real unchanged v calls precede every u call, sharing original600MB cache cap; wholeprediction lifetime/changedrosters must be validated natively. No MSE or wall claim.')
    with a.out.open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({k:receipt[k]for k in('status','unique_sequences','mean_bill_delta_pct','mean_conservative_bill_delta_pct','bill_delta_range','max_charged_call_delta','peak_rss_bytes')},indent=2))
if __name__=='__main__':main()
