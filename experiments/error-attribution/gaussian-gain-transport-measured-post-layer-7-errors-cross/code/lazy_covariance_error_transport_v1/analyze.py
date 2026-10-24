"""Privileged, fixed error transport; no fit or estimator state writes."""
import json,time,resource,gc
from common import HERE,D,P2,A,sha,pin,write,verify_seal,weight_path,tangent
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify_seal();verify_seal('capture_sealed.json');started=time.monotonic()
    capture_result=json.loads((HERE/'capture/result.json').read_text());assert capture_result['completed']
    recs=[];forecasts={};out_arrays={}
    def sym(x):return (x+x.T)*.5
    def energy(xs):
        observed=float(sum(np.mean(x*x) for x in xs)/len(xs))
        out=dict(observed_mse=observed)
        if len(xs)==2:
            cross=float(np.mean(xs[0]*xs[1]));noise=float(np.mean((xs[0]-xs[1])**2)/2)
            assert abs(observed-cross-noise)<=1e-12*max(observed,1e-30)
            out.update(cross_mse=cross,noise_mse=noise,noise_fraction=noise/observed if observed else None)
        return out
    def compare(target,pred):
        base=energy(target);res=energy([a-b for a,b in zip(target,pred)])
        out=dict(target=base,residual=res,observed_ratio=res['observed_mse']/base['observed_mse'] if base['observed_mse']>0 else None)
        if len(target)==2:out['cross_ratio']=res['cross_mse']/base['cross_mse'] if base['cross_mse']>0 else None
        return out
    def truth(path):
        with np.load(path,allow_pickle=False) as z:
            assert int(z['P'])==8_000_000
            return dict(m=z['mc_post'].astype(np.float64),C=z['mc_post_cov'].astype(np.float64),pre_m8=z['mc_pre_m'][8].astype(np.float64),pre_C8=z['mc_pre_cov'][8].astype(np.float64))
    for rank,call in ((35,0),(100,1)):
        cap_path=capture_result['records'][call]['capture']['path']
        with np.load(cap_path,allow_pickle=False) as z:cap={k:z[k] for k in z.files}
        W=np.load(weight_path(rank),allow_pickle=False).astype(np.float64)
        paths=[P2/f'public_g19_attribution_v1/mps_cov2_rank{rank:04d}.npz']
        if rank==35:paths.append(A/'covariance_replica_truth_v1/rank_0035_8M_v1.npz')
        truths=[truth(path) for path in paths]
        source_checks=[]
        for path,t in zip(paths,truths):
            derived_m=W[8].T@t['m'][7];derived_C=W[8].T@t['C'][7]@W[8]
            mr=float(np.linalg.norm(derived_m-t['pre_m8'])/np.linalg.norm(t['pre_m8']))
            cr=float(np.linalg.norm(derived_C-t['pre_C8'])/np.linalg.norm(t['pre_C8']))
            source_checks.append(dict(source=pin(path),affine_mean_relative=mr,affine_covariance_relative=cr,passed=max(mr,cr)<=1e-4))
        assert all(x['passed'] for x in source_checks),source_checks
        seed_e=[t['m'][7]-cap['m_07'] for t in truths]
        seed_E=[sym(t['C'][7]-cap['C_post_07']) for t in truths]
        states={name:[(e.copy(),E.copy() if name=='full' else np.zeros_like(E)) for e,E in zip(seed_e,seed_E)] for name in ('full','mean_only')}
        layers=[]
        for l in range(8,16):
            pre={k:cap[f'{k}_{l:02d}'] for k in ('sigma','alpha','pdf','gate','mg')}
            if l<15:pre['C_pre']=cap[f'C_pre_{l:02d}']
            for name in states:states[name]=[tangent(pre,e,E,W[l],final=l==15) for e,E in states[name]]
            actual_e=[t['m'][l]-(cap['prediction'][l].astype(np.float64) if l==15 else cap[f'm_{l:02d}']) for t in truths]
            prev_e=[t['m'][l-1]-cap[f'm_{l-1:02d}'] for t in truths]
            prev_E=[sym(t['C'][l-1]-cap[f'C_post_{l-1:02d}']) for t in truths]
            one_step=[tangent(pre,e,E,W[l],final=l==15) for e,E in zip(prev_e,prev_E)]
            if l==15:
                active=cap['m_15']>0
                states={name:[(e*active,None) for e,E in values] for name,values in states.items()}
                one_step=[(e*active,None) for e,E in one_step]
            row=dict(layer=l,one_step_mean=compare(actual_e,[v[0] for v in one_step]),free_mean={name:compare(actual_e,[v[0] for v in values]) for name,values in states.items()})
            if l<15:
                actual_E=[sym(t['C'][l]-cap[f'C_post_{l:02d}']) for t in truths]
                row['one_step_covariance']=compare(actual_E,[v[1] for v in one_step])
                row['free_covariance']={name:compare(actual_E,[v[1] for v in values]) for name,values in states.items()}
                host=cap[f'C_post_{l:02d}'];row['parent_covariance_antisymmetry_relative']=float(np.linalg.norm(host-host.T)/np.linalg.norm(host))
            layers.append(row)
        final=layers[-1];cross_gate=None
        if rank==35:
            f=final['free_mean']['full'];m=final['free_mean']['mean_only']
            denominator=m['residual']['cross_mse'];increment=f['residual']['cross_mse']/denominator if denominator>0 else None
            cross_gate=dict(full_parent_ratio=f['cross_ratio'],full_mean_only_ratio=increment,positive_denominators=f['target']['cross_mse']>0 and denominator>0)
            cross_gate['passed']=cross_gate['positive_denominators'] and f['cross_ratio']<=.75 and increment<=.75
        forecasts[rank]={name:np.stack([v[0] for v in values]) for name,values in states.items()}
        forecasts[rank]['parent']=cap['prediction'][-1].astype(np.float64)
        for name,x in forecasts[rank].items():out_arrays[f'r{rank}_{name}']=x
        rec=dict(rank=rank,source_checks=source_checks,replicas=len(truths),layers=layers,cross_gate=cross_gate)
        recs.append(rec);print(json.dumps(dict(rank=rank,cross_gate=cross_gate,final_observed_full_ratio=final['free_mean']['full']['observed_ratio'],final_observed_mean_only_ratio=final['free_mean']['mean_only']['observed_ratio'])),flush=True)
        del cap,W,truths,states,seed_E,prev_E,one_step;gc.collect()
    forecast_path=HERE/'fixed_forecasts.npz';np.savez(forecast_path,**out_arrays)
    write(HERE/'before_1B_truth.json',dict(forecasts=pin(forecast_path),all_forecasts_fixed=True,ordinary_1B_means_accessed=False,records=recs))
    for rec in recs:
        rank=rec['rank'];f=forecasts[rank];tp=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
        with np.load(tp,allow_pickle=False) as z:target=z['mc_post'][-1].astype(np.float64)
        e=target-f['parent'];parent=float(np.mean(e*e));mse={name:float(np.mean((e[None,:]-f[name])**2)) for name in ('full','mean_only')}
        ratios=dict(full_parent=mse['full']/parent,mean_only_parent=mse['mean_only']/parent,full_mean_only=mse['full']/mse['mean_only'])
        rec['ordinary_1B']=dict(reference=pin(tp),parent_mse=parent,mse=mse,ratios=ratios,passed=ratios['full_parent']<=.75 and ratios['full_mean_only']<=.75)
        print(json.dumps(dict(rank=rank,ordinary_1B=rec['ordinary_1B'])),flush=True)
    passed=recs[0]['cross_gate']['passed'] and recs[1]['ordinary_1B']['passed']
    result=dict(passed=bool(passed),records=recs,forecast_file=pin(forecast_path),fitted_parameters=0,new_samples=0,native_state_writes=0,qualification='Privileged Gaussian-gain transport point estimates on reused development networks; no complete-estimator derivative, deployable supplier, adjusted score or confidence bound.',wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'analysis.json',result);print(json.dumps(dict(passed=passed,wall_s=result['wall_s'],peak_rss_bytes=result['peak_rss_bytes'])),flush=True)
if __name__=='__main__':main()
