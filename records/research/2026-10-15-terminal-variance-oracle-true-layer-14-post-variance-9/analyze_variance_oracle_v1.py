"""Offline TRAIN16 mean-only oracle; no candidate import or estimator calls/fit.
Current mu, absolute k3/k4 and learned mean increment stay fixed. L12–14 are
local open-loop diagnostics; L15 is the terminal mean-only counterfactual.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=D/'run_capture_train16_v1';C=H/'native_16_capture_capture.npz'

def formula(mu,v,k3,k4):
    s=np.sqrt(np.maximum(v,1e-10));a=mu/s;p=np.exp(-a*a/2)/np.sqrt(2*np.pi)
    return (s*p+mu*ndtr(a))+k3*(-a*p/v)/6+k4*((a*a-1)*p/(v*s))/24

def main():
    owner=json.loads((H/'OWNER_RETIREMENT_v1.json').read_text())
    assert owner['actual_managed_exit_observed'] is True and owner['exit_code']==0 and owner['all_owned_PID_PPID_PGID_absent'] is True
    cfg=json.loads((D/'CAPTURE_PROTOCOL_HELD_v1.json').read_text())
    with np.load(C,allow_pickle=False)as z:c={k:z[k]for k in z.files}
    truths=[]
    for path in cfg['truth_paths']:
        with np.load(path,allow_pickle=False)as z:
            assert int(z['rank'])==16 and str(z['weights_sha256'])==cfg['weights_raw_sha256']
            truths.append({k:z[k]for k in ('mc_pre_v','mc_post')})
    target_path=O.parent/'fable_refit_v1/truth688/truthpost_rank0016.npz'
    with np.load(target_path,allow_pickle=False)as z:target=z['mc_post'].astype(float)
    out=[]
    for l in (12,13,14,15):
        v=c[f'L{l}_pre_v'];mu=c[f'L{l}_mu'];k3=c[f'L{l}_pre_k3'];k4=c[f'L{l}_pre_k4']
        host=c[f'L{l}_host'];final=c[f'L{l}_final'];s=c[f'L{l}_sigma'];a=c[f'L{l}_alpha'];p=c[f'L{l}_pdf']
        modeled=formula(mu,v,k3,k4);identity=float(np.max(np.abs(modeled-host)))
        assert identity<1e-10,('mean formula mismatch',l,identity)
        va,vb=(t['mc_pre_v'][l]for t in truths);tv=(va+vb)/2
        delta=formula(mu,tv,k3,k4)-modeled
        ya=final+formula(mu,va,k3,k4)-modeled;yb=final+formula(mu,vb,k3,k4)-modeled
        base=float(np.mean((final-target[l])**2));arm=float(np.mean((final+delta-target[l])**2))
        # Independent variance-input halves, independent organizer mean target:
        # the cross loss removes the leading variance-input MC noise.
        cross=float(np.mean((ya-target[l])*(yb-target[l])))
        full_dv=p/(2*s)+k3*(3*a-a**3)*p/(12*v*v)+k4*(a**4-6*a*a+3)*p/(48*v*v*s)
        gaussian_delta=p/(2*s)*(tv-v);linear_delta=full_dv*(tv-v)
        rel=np.log(v/tv);noise=np.log(va/vb)/2
        pooled_target=(truths[0]['mc_post'][l]+truths[1]['mc_post'][l])/2
        out.append(dict(layer=l,scope='terminal fixed-state mean-only' if l==15 else 'local open-loop, no downstream propagation',
            formula_max_abs_error=identity,base_mse=base,oracle_mse=arm,oracle_change_pct=100*(arm/base-1),
            independent_half_cross_mse=cross,independent_half_cross_change_pct=100*(cross/base-1),
            half_A_mse=float(np.mean((ya-target[l])**2)),half_B_mse=float(np.mean((yb-target[l])**2)),
            gaussian_linear_change_pct=100*(np.mean((final+gaussian_delta-target[l])**2)/base-1),
            cumulant_linear_change_pct=100*(np.mean((final+linear_delta-target[l])**2)/base-1),
            relative_variance_error_mean=float(rel.mean()),relative_variance_error_sd=float(rel.std()),
            half_difference_noise_sd=float(noise.std()),oracle_delta_rms=float(np.sqrt(np.mean(delta**2))),
            oracle_delta_vs_negative_error_corr=float(np.corrcoef(delta,target[l]-final)[0,1]),
            pooled_MC_target_base_mse=float(np.mean((final-pooled_target)**2)),
            pooled_MC_target_oracle_mse=float(np.mean((final+delta-pooled_target)**2)),
            physical_features_shape=list(c[f'L{l}_features30'].shape),gnn_state_shape=list(c[f'L{l}_state64'].shape)))
    result=dict(rank=16,TRAIN_only=True,no_fit=True,no_gain_accepted=True,held_fixed=['mu','absolute k3','absolute k4','learned dm'],
        capture_sha256=sha(C),target_path=str(target_path),target_sha256=sha(target_path),truth_sha256={p:sha(p)for p in cfg['truth_paths']},layers=out)
    path=D/'CURRENT_VARIANCE_ORACLE_TRAIN16_v1.json'
    with path.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
