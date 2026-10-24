import sys,gc,time
import numpy as np
import torch
from common import *
from basis import actions,NAMES
sys.path.insert(1,str(OLD))
from consumer_v4 import tensors,evaluate

verify();torch.set_num_threads(1)
u=read(HERE/'unit_result.json');guard=read(HERE/'unit_guard.json')
assert u['passed'] and u['seal_sha256']==sha(HERE/'seal.json')
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert read(OLD/'qualification_v4.json')['passed']
def load(rank):
    with np.load(packet(rank),allow_pickle=False) as z:return {k:z[k] for k in z.files}
with np.load(OLD/'parent_parameters.npz') as z:P=tensors({k:z[k] for k in z.files})
start=time.monotonic();responses={};variances={};basepred={};inputpins=[]
for rank in TRAIN+CHECK:
    a=load(rank);t=tensors(a);s=a['s14'];W=a['W'].astype(np.float64)
    response=[];variance=[]
    for name,D in actions(a['C'],a['K'],s,a['F'][:,5].astype(np.float64),a['F'][:,6].astype(np.float64)):
        dv=np.sum(W*(D@W),axis=0)
        _,jvp=torch.autograd.functional.jvp(lambda v:evaluate(t,P,v)[0],t['v'],torch.from_numpy(dv),create_graph=False,strict=False)
        response.append(jvp.detach().numpy().copy());variance.append(dv)
    responses[rank]=np.stack(response,axis=1);variances[rank]=np.stack(variance,axis=1);basepred[rank]=a['prediction'][-1].copy()
    inputpins.append(pin(packet(rank)))
    assert np.isfinite(responses[rank]).all() and np.isfinite(variances[rank]).all()
    print(json.dumps(dict(event='responses',rank=rank,actions=len(response),elapsed_s=time.monotonic()-start)),flush=True)
    del a,t,D,W,response,variance,jvp;gc.collect()
with (HERE/'responses.npz').open('xb') as f:
    np.savez(f,**{f'{k}_{rank}':value for rank in TRAIN+CHECK for k,value in (('response',responses[rank]),('variance',variances[rank]),('parent',basepred[rank]))})
once(HERE/'before_targets.json',dict(packets=inputpins,responses=pin(HERE/'responses.npz'),targets_accessed=False))
truth={};targetpins=[];baseline={}
for rank in TRAIN:
    p=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(p) as z:truth[rank]=z['mc_post'][15].astype(np.float64)
    baseline[rank]=float(np.mean((basepred[rank]-truth[rank])**2));targetpins.append(pin(p))
scale=np.sqrt(np.mean([np.mean(responses[r]**2,axis=0)/baseline[r] for r in TRAIN],axis=0))
assert np.all(scale>1e-20) and np.isfinite(scale).all()
coefficients={};normal={};models={'scale':scale};fitrows=[]
for arm,count in (('local',5),('augmented',15)):
    X=np.concatenate([responses[r][:,:count]/(scale[:count]*np.sqrt(baseline[r])) for r in TRAIN])
    y=np.concatenate([(truth[r]-basepred[r])/np.sqrt(baseline[r]) for r in TRAIN])
    gram=X.T@X/len(y);rhs=X.T@y/len(y);system=gram+.01*np.eye(count)
    beta=np.linalg.solve(system,rhs);coeff=np.zeros(15);coeff[:count]=beta/scale[:count]
    coefficients[arm]=coeff;models[arm+'_coeff']=coeff;models[arm+'_beta']=beta;models[arm+'_gram']=gram;models[arm+'_rhs']=rhs
    residual=system@beta-rhs;assert np.linalg.norm(residual)<1e-10*max(np.linalg.norm(rhs),1e-30)
    fitrows.append(dict(arm=arm,parameters=count,linear_train_ratio=float(np.mean((y-X@beta)**2)),normal_equation_max_abs=float(np.max(np.abs(residual))),coefficients=coeff.tolist()))
with (HERE/'model.npz').open('xb') as f:np.savez(f,**models)
once(HERE/'model_frozen.json',dict(model=pin(HERE/'model.npz'),fits=fitrows,train_targets=targetpins.copy(),check_targets_accessed=False,lambda_ridge=.01))
forecasts={};diagnostics=[]
for rank in TRAIN+CHECK:
    a=load(rank);t=tensors(a)
    with torch.no_grad():
        origin,_=evaluate(t,P,t['v'])
        for arm,coef in coefficients.items():
            dv=variances[rank]@coef;v=t['v']+torch.from_numpy(dv)
            assert bool(torch.all(v>0))
            changed,_=evaluate(t,P,v);delta=(changed-origin).numpy();linear=responses[rank]@coef
            pred=basepred[rank]+delta;assert np.isfinite(pred).all()
            discrepancy=float(np.linalg.norm(delta-linear)/max(np.linalg.norm(linear),1e-30))
            relative=float(np.max(np.abs(dv/a['v'])))
            diagnostics.append(dict(rank=rank,arm=arm,relative_variance_max=relative,minimum_variance=float(v.min()),nonlinear_relative=discrepancy,correction_rms=float(np.sqrt(np.mean(delta*delta)))))
            forecasts[f'{arm}_{rank}']=pred
    del a,t;gc.collect()
with (HERE/'predictions.npz').open('xb') as f:np.savez(f,**forecasts)
once(HERE/'predictions_frozen.json',dict(predictions=pin(HERE/'predictions.npz'),diagnostics=diagnostics,check_targets_accessed=False,model=pin(HERE/'model.npz')))
validity=all(x['minimum_variance']>0 and x['relative_variance_max']<=.02 and x['nonlinear_relative']<=.05 for x in diagnostics)
if not validity:
    once(HERE/'result.json',dict(passed=False,validity=False,check_targets_accessed=False,diagnostics=diagnostics,fits=fitrows,elapsed_s=time.monotonic()-start))
    print(json.dumps(dict(event='validity_failed',max_nonlinear=max(x['nonlinear_relative'] for x in diagnostics),max_relative_variance=max(x['relative_variance_max'] for x in diagnostics))),flush=True)
    raise SystemExit(0)
for rank in CHECK:
    p=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(p) as z:truth[rank]=z['mc_post'][15].astype(np.float64)
    baseline[rank]=float(np.mean((basepred[rank]-truth[rank])**2));targetpins.append(pin(p))
rows=[];summary={}
for arm in coefficients:
    for rank in TRAIN+CHECK:
        mse=float(np.mean((forecasts[f'{arm}_{rank}']-truth[rank])**2))
        rows.append(dict(arm=arm,rank=rank,split='TRAIN' if rank in TRAIN else 'CHECK',mse=mse,parent_mse=baseline[rank],raw_ratio=mse/baseline[rank]))
    held=[r for r in rows if r['arm']==arm and r['split']=='CHECK']
    training=[r for r in rows if r['arm']==arm and r['split']=='TRAIN']
    summary[arm]=dict(check_pooled_raw_ratio=sum(r['mse'] for r in held)/sum(r['parent_mse'] for r in held),check_wins=sum(r['raw_ratio']<1 for r in held),check_worst=max(r['raw_ratio'] for r in held),train_pooled_raw_ratio=sum(r['mse'] for r in training)/sum(r['parent_mse'] for r in training))
aug=summary['augmented'];relative=aug['check_pooled_raw_ratio']/summary['local']['check_pooled_raw_ratio']
gates=dict(pooled=aug['check_pooled_raw_ratio']<=.95,wins=aug['check_wins']>=6,worst=aug['check_worst']<=1.05,feature_value=relative<=.975)
once(HERE/'result.json',dict(passed=all(gates.values()),validity=True,gates=gates,summary=summary,augmented_to_local_ratio=relative,rows=rows,targets=targetpins,diagnostics=diagnostics,fits=fitrows,seal_sha256=sha(HERE/'seal.json'),model=pin(HERE/'model.npz'),elapsed_s=time.monotonic()-start,native_scored=False,scope='Current-S31 offline terminal covariance action; reused development networks; native fidelity and own bill untested.'))
print(json.dumps(dict(event='complete',passed=all(gates.values()),gates=gates,summary=summary,augmented_to_local_ratio=relative)),flush=True)
