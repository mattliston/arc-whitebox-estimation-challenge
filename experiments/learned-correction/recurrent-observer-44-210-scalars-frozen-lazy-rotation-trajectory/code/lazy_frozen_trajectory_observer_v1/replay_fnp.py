"""Actual FNP copied-GNN replay and fixed two-sided perturbation, no labels."""
import importlib.util,json,resource,sys,time,types
from common import HERE,SOURCE,sha,read,seal_check,write,enc4_direction
seal_check()
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
assert all(t['num_threads']==1 for t in threadpool_info())
cap=read(HERE/'capture_qualification/result.json');guard=read(HERE/'capture_qualification_watch.json')
assert cap['completed'] and cap['repeat_all_captured_bytes_exact']
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert sha(cap['parent_P']['path'])==cap['parent_P']['sha256']
with np.load(cap['parent_P']['path'],allow_pickle=False) as z:P0={k:z[k].copy() for k in z.files}
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('lfto_neural_replay',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
with flops.BudgetContext(flop_budget=2**41,quiet=True) as setup:spec.loader.exec_module(mod)
out=HERE/'fnp_replay';out.mkdir(exist_ok=False);records=[];first=None;started=time.monotonic()
direction=enc4_direction(np)
for rec in cap['records']:
    assert sha(rec['path'])==rec['sha256'] and sha(rec['weights'])==rec['weights_sha256']
    with np.load(rec['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    weights=np.load(rec['weights'],allow_pickle=False)
    outputs={};bills={};max_state=0.
    for label,sign in [('zero',0),('plus',1),('minus',-1)]:
        proposal={k:v.copy() for k,v in P0.items()}
        if sign:proposal['enc4_w']=proposal['enc4_w']+np.float32(sign*2**-8)*direction
        with flops.BudgetContext(flop_budget=8_000_000_000,quiet=True) as ctx:
            P={k:fnp.asarray(v.copy(),dtype=fnp.float32) for k,v in proposal.items()}
            state=fnp.zeros((1024,64),dtype=fnp.float32);states=[];deltas=[]
            for l in range(16):
                delta,state=mod.gnn_step(l,fnp.asarray(a['features'][l]),fnp.asarray(a['sigma'][l]),fnp.asarray(weights[l]),state,P)
                states.append(state);deltas.append(delta)
            final=fnp.asarray(fnp.maximum(fnp.asarray(a['final_host'])+fnp.asarray(delta,dtype=fnp.float64),0),dtype=fnp.float32)
        bill=ctx.summary_dict()['flops_used'];assert bill<=8_000_000_000
        H=np.stack([np.asarray(x).copy() for x in states]);dd=np.stack([np.asarray(x).astype(np.float32) for x in deltas]);answer=np.asarray(final).copy()
        assert np.isfinite(H).all() and np.isfinite(dd).all() and np.isfinite(answer).all()
        if label=='zero':
            assert H.tobytes()==a['hidden'].tobytes() and dd.tobytes()==a['delta'].tobytes()
            assert answer.tobytes()==a['prediction'][-1].tobytes()
        outputs[label+'_hidden']=H;outputs[label+'_delta']=dd;outputs[label+'_prediction']=answer;bills[label]=bill
        del states,deltas,state,delta,final,P,proposal,H,dd,answer
    if rec['rank']==35:
        if first is None:first={k:v.copy() for k,v in outputs.items()}
        else:assert all(v.tobytes()==first[k].tobytes() for k,v in outputs.items())
    path=out/f"call_{rec['call']:02d}_rank{rec['rank']:04d}.npz";np.savez(path,**outputs,direction=direction)
    row=dict(call=rec['call'],rank=rec['rank'],path=str(path),sha256=sha(path),capture_sha256=rec['sha256'],bills=bills,zero_all_hidden_delta_output_bytes_exact=True,truth_accessed=False)
    records.append(row);print(json.dumps(row),flush=True)
    del a,weights,outputs
write(out/'result.json',dict(completed=True,passed=True,records=records,repeat_all_outputs_exact=True,parent_P=cap['parent_P'],module_import_flops=setup.summary_dict()['flops_used'],maximum_replay_flops=max(v for r in records for v in r['bills'].values()),replay_calls=9,new_physical_calls=0,truth_accessed=False,source_seal_sha256=sha(HERE/'capture_sources_sealed.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
