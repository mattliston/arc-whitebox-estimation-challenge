"""No-fit physics diagnostic on captured TRAIN16/259/277 only.
D = (W15²).T @ (true POSTvariance14 - model outgoing POSTvariance14).
E = true PREvariance15 - model PREvariance15; R = E-D.
R includes cross-covariance error and any transport/gate-cut discrepancy.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
D=Path(__file__).resolve().parent;O=D.parents[2];RANKS=(16,259,277)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def formula(mu,v,k3,k4):
    assert np.all(v>0)
    s=np.sqrt(np.maximum(v,1e-10));a=mu/s;p=np.exp(-a*a/2)/np.sqrt(2*np.pi)
    return s*p+mu*ndtr(a)+k3*(-a*p/v)/6+k4*((a*a-1)*p/(v*s))/24

def main():
    rows=[];evidence={}
    for rank in RANKS:
        cp=D/f'run_capture_train{rank}_v1/native_{rank}_capture_capture.npz'
        with np.load(cp,allow_pickle=False)as z:c={k:z[k]for k in z.files if k.startswith(('L14_','L15_'))or k=='predictions_float64'}
        wp=O.parent/f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
        w=np.load(wp,mmap_mode='r',allow_pickle=False);rawhash=hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest();W=w[15].astype(float);W2=W*W
        true=[];paths=[cp,wp]
        for half in ('A','B'):
            path=O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_{half}32M.npz';paths.append(path)
            with np.load(path,allow_pickle=False)as z:
                assert int(z['rank'])==rank and str(z['weights_sha256'])==rawhash
                true.append(dict(pre=z['mc_pre_v'][15],post=z['mc_post_var'][14]))
        tp=O.parent/f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz';paths.append(tp)
        with np.load(tp,allow_pickle=False)as z:target=z['mc_post'][15].astype(float)
        v=c['L15_pre_v'];mu=c['L15_mu'];k3=c['L15_pre_k3'];k4=c['L15_pre_k4'];final=c['L15_final']
        host=formula(mu,v,k3,k4);assert np.max(np.abs(host-c['L15_host']))<1e-10
        errors=[t['pre']-v for t in true]
        diagonal=[W2.T@(t['post']-c['L14_outgoing_variance'])for t in true]
        remaining=[e-d for e,d in zip(errors,diagonal)]
        E=(errors[0]+errors[1])/2;DD=(diagonal[0]+diagonal[1])/2;R=(remaining[0]+remaining[1])/2
        base=float(np.mean((c['predictions_float64'][15]-target)**2));parts={}
        for name,halves in [('total_prevariance',errors),('previous_diagonal',diagonal),('remaining_covariance_transport',remaining)]:
            x=(halves[0]+halves[1])/2
            correction=formula(mu,v+x,k3,k4)-host
            y=(final+correction).astype(np.float32).astype(float)
            mse=float(np.mean((y-target)**2))
            a=final+formula(mu,v+halves[0],k3,k4)-host
            b=final+formula(mu,v+halves[1],k3,k4)-host
            noise=(halves[0]-halves[1])/2
            parts[name]=dict(error_mean=float(x.mean()),error_sd=float(x.std()),error_rms=float(np.sqrt(np.mean(x*x))),
                half_difference_noise_rms=float(np.sqrt(np.mean(noise*noise))),
                projection_share_of_total=float(np.sum(x*E)/np.sum(E*E)),
                centered_projection_share_of_total=float(np.sum((x-x.mean())*(E-E.mean()))/np.sum((E-E.mean())**2)),
                corr_with_total=float(np.corrcoef(x,E)[0,1]),mse=mse,mean_improvement_pct=100*(1-mse/base),
                independent_half_cross_mse=float(np.mean((a-target)*(b-target))),
                oracle_mean_correction_rms=float(np.sqrt(np.mean(correction*correction))),
                oracle_correction_error_corr=float(np.corrcoef(correction,target-c['predictions_float64'][15])[0,1]))
        rows.append(dict(rank=rank,base_mse=base,component_additivity_max_abs=float(np.max(np.abs(E-DD-R))),
            diag_remaining_corr=float(np.corrcoef(DD,R)[0,1]),parts=parts))
        evidence.update({str(p):sha(p)for p in paths})
    summaries={}
    for name in rows[0]['parts']:
        summaries[name]=dict(geometric_mean_improvement_pct=float(100*(1-np.exp(np.mean([np.log(r['parts'][name]['mse']/r['base_mse'])for r in rows])))),
            lower_count=sum(r['parts'][name]['mse']<r['base_mse']for r in rows))
    result=dict(ranks=list(RANKS),rank287_opened=False,no_fit=True,no_models=True,no_gain_accepted=True,
        definition='E=true prevar15-current prevar15; D=W15_squared.T@(true postvar14-captured outgoing postvar14); R=E-D',
        qualification='D is full unmasked diagonal error propagation. R also includes differences from response gate cuts, other transport approximations and cross-covariances; not a pure offdiagonal oracle.',
        held_fixed=['terminal mu','absolute k3','absolute k4','learned mean correction'],rows=rows,summary=summaries,
        source_sha256=sha(__file__),evidence=evidence)
    out=D/'TERMINAL_VARIANCE_DECOMPOSITION_v1.json'
    with out.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2,allow_nan=False))

if __name__=='__main__':main()
