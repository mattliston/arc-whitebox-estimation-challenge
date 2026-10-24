"""Fixed TRAIN32 residual linear fit. Never reads captures before full retirement."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fit(records, ridge):
    X=np.concatenate([r['x'] for r in records]); sigma=np.concatenate([r['sigma'] for r in records])
    target=np.concatenate([(r['target']-r['raw'])/r['sigma'] for r in records])
    center=X.mean(axis=0);scale=np.maximum(X.std(axis=0),1e-6)
    Z=np.column_stack((np.ones(len(X)),(X-center)/scale))
    weights=sigma*sigma;weights/=weights.mean()
    G=(Z.T*weights)@Z/len(Z);rhs=Z.T@(weights*target)/len(Z)
    penalty=np.eye(Z.shape[1])*ridge;penalty[0,0]=0
    beta=np.linalg.solve(G+penalty,rhs)
    coef=beta[1:]/scale; bias=float(beta[0]-center@coef)
    assert np.isfinite(coef).all() and np.isfinite(bias)
    return dict(coef=coef,bias=bias,train_feature_mean=center,train_feature_scale=scale)

def evaluate(model,r):
    # Deployment expression: full-precision residual applied before original clamp/cast.
    delta=(r['x'][:,:30]@model['coef'][:30]+r['x'][:,30:]@model['coef'][30:])+model['bias']
    pred=np.maximum(r['raw']+r['sigma']*delta,0).astype(np.float32)
    mse=float(np.mean((pred.astype(np.float64)-r['target'])**2))
    ratio=mse/r['baseline_mse'];assert np.isfinite(ratio) and ratio>0
    return dict(rank=r['rank'],baseline_mse=r['baseline_mse'],mse=mse,log_mse_ratio=math.log(ratio),delta_rms=float(np.sqrt(np.mean((r['sigma']*delta)**2))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--retirement',type=Path,required=True);p.add_argument('--retirement-sha256',required=True);p.add_argument('--run-authorized',action='store_true');a=p.parse_args();assert a.run_authorized
    assert sha(a.retirement)==a.retirement_sha256
    receipt=json.loads(a.retirement.read_text());assert receipt['approved'] and receipt['reviewed_by']=='root'
    assert receipt['actual_managed_exits_observed'] and receipt['all_owned_PID_PPID_PGID_absent'] and receipt['all_guards_clean'] and receipt['complete_calls']==32
    protocol=json.loads((D/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text());rule=json.loads((D/'FIT_RULE_FROZEN_v1.json').read_text())
    assert rule==protocol['fit_rule'] and rule['ridge_grid']==[.1,1.,10.,100.,1000.,10000.] and rule['folds']==[list(range(4*i,4*i+4))for i in range(8)]
    items=receipt['evaluations'];assert sorted(x['seed'] for x in items)==list(range(32))
    records=[];base_params=None
    for item in sorted(items,key=lambda x:x['seed']):
        assert sha(item['capture_path'])==item['capture_sha256']
        with np.load(item['capture_path'],allow_pickle=False)as z:
            x=np.column_stack((z['features30'],z['state64'])).astype(np.float64)
            sigma=z['sigma'].astype(np.float64);raw=z['raw_final_float64'].astype(np.float64);target=z['target_final_float64'].astype(np.float64)
            base=z['predictions_float32'][15].astype(np.float64)
            assert x.shape==(1024,94) and np.isfinite(x).all() and np.all(sigma>0)
            assert np.maximum(raw,0).astype(np.float32).tobytes()==base.astype(np.float32).tobytes()
            params={k:z[k].copy() for k in ('head','head_b','rms','gain')}
        if base_params is None:base_params=params
        else:assert all(base_params[k].tobytes()==params[k].tobytes() for k in params)
        records.append(dict(rank=item['seed'],x=x,sigma=sigma,raw=raw,target=target,baseline_mse=float(np.mean((base-target)**2))))
    results=[]
    for ridge in rule['ridge_grid']:
        scored=[]
        for fold in rule['folds']:
            model=fit([r for r in records if r['rank'] not in fold],ridge)
            scored.extend(evaluate(model,r)for r in records if r['rank'] in fold)
        mean=float(np.mean([v['log_mse_ratio']for v in scored]))
        results.append(dict(ridge=ridge,mean_log_mse_ratio=mean,geometric_mse_change_pct=100*math.expm1(mean),networks_lower=sum(v['log_mse_ratio']<0 for v in scored),per_network=sorted(scored,key=lambda r:r['rank'])))
    # Stable tie: zero first; among nonzero equal scores, larger ridge first.
    choices=[dict(ridge=None,mean_log_mse_ratio=0.)]+sorted(results,key=lambda r:-r['ridge'])
    winner=min(choices,key=lambda r:r['mean_log_mse_ratio'])
    selected=winner['ridge'];output=None
    if selected is not None:
        final=fit(records,selected)
        output=D/'FROZEN_LINEAR_COEFFICIENTS_v1.npz';assert not output.exists()
        np.savez(output,coefficient94=final['coef'],bias=np.array(final['bias']),**{f'baseline_{k}':v for k,v in base_params.items()})
    result=dict(status='TRAIN_CV_NEGATIVE_REQUIRES_INDEPENDENT_FRESH' if selected is not None else 'BASELINE_ZERO_SELECTED_PARK',train_ranks=list(range(32)),ridge_grid=rule['ridge_grid'],selected_ridge=selected,selected_cv=winner,all_cv=results,coefficients_path=str(output)if output else None,coefficients_sha256=sha(output)if output else None,capture_retirement_path=str(a.retirement),capture_retirement_sha256=a.retirement_sha256,fit_rule_sha256=sha(D/'FIT_RULE_FROZEN_v1.json'),fit_script_sha256=sha(__file__),no_independent_gain_claim=True,no_model_calls=True)
    out=D/'NETWORK_BLOCK_CV_RESULT_v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],selected_ridge=selected,selected_cv_geometric_change_pct=100*math.expm1(winner['mean_log_mse_ratio']),coefficients_sha256=result['coefficients_sha256'])))
if __name__=='__main__':main()
