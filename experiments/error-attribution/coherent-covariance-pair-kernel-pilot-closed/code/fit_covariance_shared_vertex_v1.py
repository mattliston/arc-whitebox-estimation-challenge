"""Exact fixed36-column source-information fit on existing coherent8M rows.

No new MC, recursion or estimator update. All normal equations and held-out
metrics are archived. Only TRAIN summaries enter normalization or fitting.
"""
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'covariance_shared_vertex_verdict_v1.json'
PROGRESS=HERE/'covariance_shared_vertex_progress_v1'
MODEL=HERE/'covariance_shared_vertex_model_v1.npz'
assert not OUT.exists() and not PROGRESS.exists() and not MODEL.exists()
assert json.loads((HERE/'covariance_shared_vertex_cost_v1.json').read_text())['passed']
assert json.loads((HERE/'covariance_innovation_probe_rows_v2.json').read_text())['n']==8
PROGRESS.mkdir()
TRAIN=(35,100,150,200,250,300);VALID=(400,500);K=36
PAIRS=[(a,b) for a in range(3) for b in range(a,3)]
LABELS=[f'middle{c}_rho{p}_endpoint{a}{b}' for c in range(3) for p in range(2) for a,b in PAIRS]
start=time.time();summaries={};precision=None


def kernel_data(d,l,first=False):
    global precision
    # Inputs all existed before current truth feedback in source-row protocol.
    raw=d['basis'][l].astype(np.float64);sigma=d['sigma'][l].astype(np.float64)
    R=d['rho'][l].astype(np.float64);R=(R+R.T)/2;np.fill_diagonal(R,0)
    F=raw[:,:3]*sigma[:,None]
    g,pdf,ap,ms=raw[:,0],raw[:,1],raw[:,2],raw[:,4]
    middles=[np.ones(len(g)),2*g-2*ms*pdf,2*pdf+2*ms*ap]
    W=d['weight'][l].astype(np.float64);response=d['response'][l].astype(np.float64)
    fields=[];values=[];errors=[]
    wf=[F[:,a,None]*W for a in range(3)]
    for c,middle in enumerate(middles):
        R32=R.astype(np.float32)
        M=(R32@(middle.astype(np.float32)[:,None]*R32)).astype(np.float64)
        M=(M+M.T)/2
        if first:
            M64=R@(middle[:,None]*R)
            errors.append(float(np.linalg.norm(M-M64)/max(np.linalg.norm(M64),1e-30)))
        for power in range(2):
            L=M if power==0 else M*R
            # Offline next-mean Jacobian for six endpoints reuses three
            # L@(F[:,a]*W) products. These are NOT deployed feature costs.
            z=[L@wa for wa in wf]
            for a,b in PAIRS:
                endpoint=np.outer(F[:,a],F[:,b])
                value=(wf[a]*z[b]).sum(0)
                if a!=b:
                    endpoint+=np.outer(F[:,b],F[:,a])
                    value+=(wf[b]*z[a]).sum(0)
                fields.append(L*endpoint)
                values.append(value*response)
    if first:
        precision=dict(rank=35,layer=8,relative_errors=errors,passed=max(errors)<=1e-5)
        (PROGRESS/'feature_precision.json').write_text(json.dumps(precision,indent=2)+'\n')
        assert precision['passed'],precision
    return np.stack(fields),np.stack(values)


def summarize(rank):
    with np.load(HERE/'rows_covariance_innovation_probe_v2'/f'row_{rank:04d}'/'data.npz') as z:
        data={k:z[k] for k in z.files}
    record=json.loads((HERE/'rows_covariance_innovation_probe_v2'/f'row_{rank:04d}'/'result.json').read_text())
    assert record['protocol']=='coherent_8m_v2' and record['mean_and_covariance_samples']==8000000
    out={k:np.zeros((K,K)) for k in ('cov_H','value_H')}
    out.update({k:np.zeros(K) for k in ('cov_b','value_b')});out.update(cov_c=0.,value_c=0.)
    n=1024;layers=[];t0=time.time()
    for l in range(7):
        field,value=kernel_data(data,l,first=rank==35 and l==0)
        # First current target read occurs AFTER feature-only precision gate.
        E=data['target'][l].astype(np.float64);W=data['weight'][l].astype(np.float64)
        response=data['response'][l].astype(np.float64)
        target_value=(W*(E@W)).sum(0)*response
        F=field.reshape(K,-1)
        out['cov_H']+=F@F.T/(7*n*n)
        out['cov_b']+=F@E.ravel()/(7*n*n)
        out['cov_c']+=float(np.sum(E*E))/(7*n*n)
        out['value_H']+=value@value.T/(7*n)
        out['value_b']+=value@target_value/(7*n)
        out['value_c']+=float(np.sum(target_value*target_value))/(7*n)
        layers.append(dict(layer=l+8,covariance_target_mse=float(np.mean(E*E)),value_target_mse=float(np.mean(target_value*target_value))))
        del field,value,F,E,target_value
    path=PROGRESS/f'rank_{rank:04d}.npz';assert not path.exists()
    np.savez(path,**out)
    (PROGRESS/f'rank_{rank:04d}.json').write_text(json.dumps(dict(rank=rank,layers=layers,wall_s=time.time()-t0,protocol=record['protocol']),indent=2)+'\n')
    print(json.dumps(dict(rank=rank,summary_complete=True,wall_s=time.time()-t0)),flush=True)
    return out


def report(theta,ranks):
    answer={}
    for rank in ranks:
        z=summaries[rank];row={}
        for prefix in ('cov','value'):
            base=z[prefix+'_c']
            mse=float(base-2*theta@z[prefix+'_b']+theta@z[prefix+'_H']@theta)
            assert mse>=-1e-10*base
            row[prefix+'_base']=base;row[prefix+'_mse']=mse;row[prefix+'_ratio']=mse/base
        answer[str(rank)]=row
    return answer


with threadpool_limits(limits=3):
    # Entire training dictionary is summarized first. No held-out data is
    # used in the fit, even indirectly through scales or column choices.
    for rank in TRAIN:summaries[rank]=summarize(rank)
    train={k:sum(summaries[r][k] for r in TRAIN)/len(TRAIN) for k in summaries[TRAIN[0]]}
    G=train['cov_H']/train['cov_c']+train['value_H']/train['value_c']
    b=train['cov_b']/train['cov_c']+train['value_b']/train['value_c']
    scale=np.sqrt(np.maximum(np.diag(G),1e-30));normalized=G/np.outer(scale,scale)
    ridge=1e-3
    coefficient=np.linalg.solve(normalized+ridge*np.eye(K),b/scale)
    theta=coefficient/scale
    fit=dict(ridge=ridge,normalized_gram_eigenvalues=np.linalg.eigvalsh(normalized).tolist(),
             normalized_system_condition=float(np.linalg.cond(normalized+ridge*np.eye(K))),
             train=report(theta,TRAIN))
    # Freeze fitted coefficients BEFORE opening held-out summaries.
    np.savez(MODEL,theta=theta,scale=scale,ridge=ridge,labels=np.asarray(LABELS),G=G,b=b)
    (PROGRESS/'fit_frozen_before_validation.json').write_text(json.dumps(fit,indent=2)+'\n')
    for rank in VALID:summaries[rank]=summarize(rank)
    validation=report(theta,VALID)
passed=all(r['cov_ratio']<=.75 and r['value_ratio']<=.75 for r in validation.values())
result=dict(train_nets=TRAIN,valid_nets=VALID,layers=list(range(8,15)),parameters=K,feature_precision=precision,
            fit=fit,valid=validation,information_gate_pass=passed,wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Fixed36-dimensional shared-vertex covariance-source information test on existing coherent8M oracle-state rows. No recursive candidate, actual final-layer MSE, new sampling, independent covariance noise correction or submission.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
