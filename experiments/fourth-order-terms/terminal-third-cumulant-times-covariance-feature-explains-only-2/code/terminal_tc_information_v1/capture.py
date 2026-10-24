"""Read-only native state capture, preserving all parent outputs and bills."""
import ast,hashlib,importlib.util,json,resource,sys,time,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

H=Path(__file__).resolve().parent;D=H.parent;P2=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(H/'source_sealed.json')
for name,digest in seal['files_sha256'].items():assert sha(H/name)==digest,name
for path,digest in seal['external_sha256'].items():assert sha(path)==digest,path
assert read(H/'unit.json')['passed']
guard=read(H/'unit_watch.json')
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert all(t['num_threads']==1 for t in threadpool_info())
parent=D/'lazy_rotation_objective_v1/candidate.py'
text=parent.read_text();tree=ast.parse(text);ls=text.splitlines(keepends=True)
node=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'][-1]
start=sum(map(len,ls[:node.lineno-1]));stop=sum(map(len,ls[:node.end_lineno]));body=text[start:stop]
a='            host = m.copy()\n'
b=a+"            if l == 14:\n                _tc_observe14(Cpre, Bpre, bpre, pp, lam, star, previous, previous_alive, w32, W[15], aux, C)\n"
c='                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)\n'
d=c+'                _tc_observe15(aux)\n'
assert body.count(a)==body.count(c)==1
changed=body.replace(a,b).replace(c,d)
assert changed.replace(b,a).replace(d,c)==body
private=text[:start]+changed+text[stop:]
assert private[:start]+private[start:start+len(changed)].replace(b,a).replace(d,c)+private[start+len(changed):]==text
packet={}
def keep(name,value):
    assert name not in packet
    packet[name]=np.asarray(value).copy()
def observe14(Cpre,Bpre,bpre,P,lam,star,previous,alive,w14,w15,aux,Cpost):
    for k,v in dict(Cpre=Cpre,Bpre=Bpre,bpre=bpre,P=P,**{'lambda':lam},star=star,
                    previous=previous,previous_alive=alive,W14=w14,W15=w15,
                    gate14=aux['gate'],pdf14=aux['pdf'],sigma14=aux['sigma'],Cpost_preP=Cpost).items():keep(k,v)
def observe15(aux):
    for k,v in dict(v15=aux['pre_v'],sigma15=aux['sigma'],alpha15=aux['alpha'],pdf15=aux['pdf']).items():keep(k,v)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('observed_tc_parent');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
controls=read(D/'lazy_rotation_v1/dev/result.json')['records']
out=H/'capture';out.mkdir(exist_ok=False);records=[];first_hashes=None
for call,control in enumerate(controls):
    rank=control['rank'];assert [35,100,35][call]==rank
    wp=(store/manifest[rank]['path']).parent/'weights.npy';assert sha(wp)==control['weights_sha256']
    weights=np.load(wp,allow_pickle=False)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
        ww=[fnp.asarray(weights[l]) for l in range(16)]
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    packet={};began=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if call==0:
            exec(compile(private,str(parent), 'exec'),mod.__dict__)
            mod._tc_observe14=observe14;mod._tc_observe15=observe15
            estimator=mod.Estimator()
        pred=estimator.predict(mlp,2**41)
    elapsed=time.monotonic()-began;bill=ctx.summary_dict()['flops_used']
    prediction=np.asarray(pred).copy();baseline=np.load(control['output'],allow_pickle=False)
    assert sha(control['output'])==control['output_sha256']
    assert prediction.dtype==np.float32 and prediction.shape==(16,1024)
    assert np.array_equal(prediction,baseline) and prediction.tobytes()==baseline.tobytes()
    assert bill==control['flops'],(bill,control['flops'])
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    keep('prediction',prediction)
    assert len(packet)==19,len(packet)
    assert all(np.isfinite(v).all() for v in packet.values())
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in packet.items()}
    if call==0:first_hashes=hashes
    if call==2:assert hashes==first_hashes
    target=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(target,**packet)
    assert target.stat().st_size<220_000_000
    rec=dict(call=call,rank=rank,packet=str(target),packet_sha256=sha(target),packet_bytes=target.stat().st_size,
             arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),raw_sha256=hashes[k]) for k,v in packet.items()},
             output_sha256=control['output_sha256'],weights_sha256=control['weights_sha256'],
             flops=bill,input_flops_separate=ingress.summary_dict()['flops_used'],wall_s=elapsed,
             parent_all16_byte_exact=True,parent_bill_exact=True,numeric_cache_retired=True,
             max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='arrays'}),flush=True)
    del weights,ww,mlp,pred,prediction,baseline;packet.clear()
assert sha(parent)==seal['parent_sha256']
result=dict(completed=True,records=records,all16_and_bills_exact=True,repeat_packet_arrays_exact=True,
            private_source_whole_inverse_exact=True,private_source_sha256=hashlib.sha256(private.encode()).hexdigest(),
            parent_source_unchanged=True,source_seal_sha256=sha(H/'source_sealed.json'),truth_accessed=False,
            diagnostic_copies_are_not_deployment_work=True)
with (out/'completed.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':True,'calls':3,'repeat_exact':True}),flush=True)
