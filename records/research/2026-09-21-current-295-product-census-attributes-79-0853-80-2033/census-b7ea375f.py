"""Passive exact-meter attribution; research-only, no estimator arithmetic edits."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)
def main():
    rank=int(sys.argv[1]);assert rank in (35,100)
    seal=read(HERE/'sources_sealed.json')
    for row in seal['sources']+seal['external_pins']:assert sha(row['path'])==row['sha256'],row['path']
    assert all(t['num_threads']==1 for t in threadpool_info())
    control_path=D/('lazy_rotation_v1/dev/result.json' if rank==35 else 'lazy_rotation_v1/cold100/result.json')
    control=read(control_path);ref=control['records'][0]
    assert ref['rank']==rank and ref['includes_initialization'] and control['candidate_sha256']==sha(SOURCE)
    assert sha(ref['output'])==ref['output_sha256']
    out=HERE/f'rank{rank:04d}';out.mkdir(exist_ok=False)
    weights_manifest=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    weight_path=STORE/Path(weights_manifest[rank]['path']).parent/'weights.npy'
    assert sha(weight_path)==ref['weights_sha256']
    weights=np.load(weight_path,allow_pickle=False);assert weights.shape==(16,1024,1024)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
        ww=[fnp.asarray(weights[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==ref['input_flops_separate']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('observed_lazy_cost',SOURCE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    plumbing={'_mm','_shared_old384_mm','_square_product','_cp3_bind_arena_views',
              '_exact_view','_a_frame','_b_frame','_o_frame','_packed_frame',
              '_read5_view','_read5_child_frame','_read5_packet_frames','_read5_leaf_packets',
              '_p1_d6_832_product','_p1_d6_832_bind','_bind_push6',
              '_split_views_3','_split_views_4','_split_views_5','_split_views_6',
              '<genexpr>','<listcomp>'}
    def is_plumbing(name):
        return name in plumbing or name.startswith('_packed_') or name.startswith('_w5_')
    categories=collections.Counter();sites=collections.Counter();ops=collections.Counter()
    stacks=collections.Counter();op_counts=collections.Counter();layer_bills=collections.Counter()
    product_calls=[];depth=0
    original_charge=_budget.BudgetContext._charge_op
    original_mm=None
    def charged(context,*args,**kwargs):
        before=context._flops_used
        answer=original_charge(context,*args,**kwargs)
        delta=context._flops_used-before
        op=kwargs.get('op_name',args[0] if args else None)
        shapes=kwargs.get('shapes',args[3] if len(args)>3 else ())
        dtypes=kwargs.get('dtypes')
        stack=[];layer=-1
        frame=sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename==str(SOURCE):
                name=frame.f_code.co_name
                stack.append((name,frame.f_lineno))
                if name=='_predict_binary':
                    value=frame.f_locals.get('l',-1)
                    if isinstance(value,int):layer=value
            frame=frame.f_back
        del frame
        site=stack[0] if stack else ('outside_candidate',0)
        semantic=next((x for x in stack if not is_plumbing(x[0])),('outside_candidate',0))
        categories[semantic[0]]+=delta
        sites[(semantic[0],semantic[1])]+=delta
        key=(str(op),str(shapes),str(tuple(str(d) for d in dtypes)) if dtypes is not None else 'None')
        ops[key]+=delta;op_counts[key]+=1;layer_bills[layer]+=delta
        stacks[tuple(stack)]+=delta
        return answer
    began=time.monotonic()
    try:
        _budget.BudgetContext._charge_op=charged
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as context:
            spec.loader.exec_module(mod)
            original_mm=mod._mm
            def observed_mm(left,right):
                nonlocal depth
                outer=depth==0;before=context._flops_used
                shape=(tuple(int(x) for x in left.shape),tuple(int(x) for x in right.shape))
                caller=sys._getframe(1)
                origin=(caller.f_code.co_name,caller.f_lineno)
                del caller
                depth+=1
                try:return original_mm(left,right)
                finally:
                    depth-=1
                    if outer:
                        product_calls.append(dict(left=list(shape[0]),right=list(shape[1]),
                                                  flops=context._flops_used-before,
                                                  caller=list(origin)))
            mod._mm=observed_mm
            result=mod.Estimator().predict(mlp,2**41)
    finally:
        _budget.BudgetContext._charge_op=original_charge
        if original_mm is not None:mod._mm=original_mm
    seconds=time.monotonic()-began
    prediction=np.asarray(result,dtype=np.float32).copy()
    assert prediction.shape==(16,1024) and np.isfinite(prediction).all() and (prediction>=0).all()
    np.save(out/'prediction.npy',prediction)
    total=context.summary_dict()['flops_used']
    assert total==ref['flops']
    assert sha(out/'prediction.npy')==ref['output_sha256']
    assert sum(categories.values())==sum(sites.values())==sum(ops.values())==sum(layer_bills.values())==sum(stacks.values())==total
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    assert _budget.BudgetContext._charge_op is original_charge
    write(out/'attribution.json',dict(
        total_flops=total,
        by_function=[dict(function=k,flops=v,fraction=v/total) for k,v in categories.most_common()],
        by_site=[dict(function=k[0],line=k[1],flops=v,fraction=v/total) for k,v in sites.most_common()],
        by_op_shape=[dict(op=k[0],shapes=k[1],dtypes=k[2],flops=v,calls=op_counts[k]) for k,v in ops.most_common()],
        by_layer=[dict(layer=k,flops=v) for k,v in sorted(layer_bills.items())],
        by_stack=[dict(stack=[dict(function=f,line=l) for f,l in k],flops=v) for k,v in stacks.most_common()],
        outermost_mm_calls=product_calls,
        outermost_mm_sum=sum(x['flops'] for x in product_calls),
        mm_sum_excludes_direct_products_and_non_mm_work=True,
        each_charged_operation_attributed_exactly_once=True,
        plumbing=sorted(plumbing),plumbing_prefixes=['_packed_','_w5_'],
    ))
    rec=dict(completed=True,rank=rank,parent_sha256=sha(SOURCE),source_seal_sha256=sha(HERE/'sources_sealed.json'),
             output_sha256=sha(out/'prediction.npy'),all16_prediction_bytes_exact=True,
             parent_bill_exact=True,complete_flops=total,includes_initialization=True,
             ingress_flops=0,weights_sha256=sha(weight_path),control_path=str(control_path),
             control_sha256=sha(control_path),attribution_sha256=sha(out/'attribution.json'),
             original_charge_called_without_argument_changes=True,charge_method_restored=True,
             numeric_cache_retired=True,truth_accessed=False,new_model=False,wall_s=seconds,
             observer_wall_is_not_deployment_wall=True,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             threads=threadpool_info())
    write(out/'result.json',rec)
    print(json.dumps(rec),flush=True)
    print(json.dumps(dict(top_functions=list(categories.most_common(18)))),flush=True)

if __name__=='__main__':main()

