"""One fixed shared native innovation regression; no model calls or exported map."""
from pathlib import Path
import hashlib, json, math, os, resource, time
import numpy as np

D=Path(__file__).resolve().parent
R=D.parent
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.monotonic()
    assert all(os.environ.get(k)=='1' for k in THREADS)
    assert os.getpriority(os.PRIO_PROCESS,0)==10
    rule=json.loads((D/'FIXED_RULE_v1.json').read_text())
    assert rule['script_sha256']==sha(__file__)
    assert rule['ridge']==10. and rule['shrink']==.5
    assert not (D/'SAVED_RESULT_v1.json').exists()
    source=R/'astra_native_intermediate_mean_refit_v1'
    meta=json.loads((source/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text())['network_metadata']
    ret=json.loads((source/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text())
    assert ret['approved'] and ret['actual_managed_exits_observed'] and ret['all_owned_PID_PPID_PGID_absent']
    captures={e['seed']:e for e in ret['evaluations']}
    records=[]; pins={}; data=[]
    for rank in range(32):
        m=meta[rank]; assert m['rank']==rank
        p=Path(captures[rank]['capture_path'])
        assert sha(p)==captures[rank]['capture_sha256']; pins[str(p)]=sha(p)
        tp=R/f'astra_native_terminal_linear_refit_v1/run_capture_s{rank%4}_v1/native_{rank}_capture_capture.npz'
        pins[str(tp)]=sha(tp)
        w=np.load(m['weights_path'],mmap_mode='r',allow_pickle=False)
        with np.load(m['truth_path'],allow_pickle=False) as z: truth=z['mc_post'].astype(np.float64)
        with np.load(p,allow_pickle=False) as z:
            pred=z['predictions_float32'].astype(np.float64)
            err=truth-pred
            G=np.zeros((95,95)); rhs=np.zeros(95); sx=np.zeros(94); sxx=np.zeros(94); sw=0.; count=0
            local=[]
            for l in range(7,15):
                X=np.column_stack([z[f'L{l}_features30'],z[f'L{l}_state64']]).astype(np.float64)
                sigma=z[f'L{l}_sigma'].astype(np.float64)
                gate=X[:,5].copy()
                assert np.all((gate>=0)&(gate<=1)) and np.all(sigma>0)
                inherited=gate*(err[l-1]@np.asarray(w[l],np.float64))
                innovation=err[l]-inherited
                imse=float(np.mean(innovation**2)); assert imse>0
                y=innovation/sigma
                weight=sigma**2/imse
                Z=np.column_stack([np.ones(1024),X])
                G+=(Z.T*weight)@Z; rhs+=Z.T@(weight*y)
                sw+=weight.sum(); sx+=X.sum(0); sxx+=(X*X).sum(0); count+=1024
                local.append(dict(layer=l,X=X,sigma=sigma,gate=gate,innovation=innovation,imse=imse))
            with np.load(tp,allow_pickle=False) as t:
                assert np.array_equal(t['predictions_float32'],pred.astype(np.float32))
                terminal_gate=t['features30'][:,5].astype(np.float64)
                assert np.array_equal(t['target_final_float64'],truth[15])
            data.append(dict(rank=rank,local=local,pred=pred,truth=truth,terminal_gate=terminal_gate,weight_path=m['weights_path'],stats=(G,rhs,sw,sx,sxx,count)))
        del w
    print(json.dumps(dict(phase='loaded32',seconds=time.monotonic()-start)),flush=True)
    allstats=[sum(row['stats'][i] for row in data) for i in range(6)]
    foldrows=[]; prediction_rows=[]
    for fi,fold in enumerate(rule['folds']):
        stats=[allstats[i]-sum(data[r]['stats'][i] for r in fold) for i in range(6)]
        G,rhs,sw,sx,sxx,count=stats
        center=sx/count; scale=np.sqrt(np.maximum(sxx/count-center**2,1e-12))
        T=np.diag(np.r_[1.,1./scale]); T[0,1:]=-center/scale
        penalty=np.eye(95)*10.; penalty[0,0]=0.
        beta=np.linalg.solve(T.T@(G/sw)@T+penalty,T.T@(rhs/sw))
        coef=beta[1:]/scale; bias=float(beta[0]-center@coef)
        assert np.isfinite(coef).all() and math.isfinite(bias)
        for rank in fold:
            row=data[rank]; w=np.load(row['weight_path'],mmap_mode='r',allow_pickle=False)
            propagated=np.zeros(1024); local_logs=[]
            for a in row['local']:
                l=a['layer']
                correction=.5*a['sigma']*(a['X']@coef+bias)
                corrected=np.maximum(row['pred'][l]+correction,0.)
                applied=corrected-row['pred'][l]
                mse=float(np.mean((a['innovation']-applied)**2))
                lm=math.log(mse/a['imse']); local_logs.append(lm)
                records.append(dict(rank=rank,layer=l,baseline_innovation_mse=a['imse'],innovation_mse=mse,log_mse_ratio=lm,correction_rms=float(np.sqrt(np.mean(applied**2)))))
                propagated=a['gate']*(propagated@np.asarray(w[l],np.float64))+applied
            propagated=row['terminal_gate']*(propagated@np.asarray(w[15],np.float64))
            final=np.maximum(row['pred'][15]+propagated,0.).astype(np.float32).astype(np.float64)
            baseline=float(np.mean((row['truth'][15]-row['pred'][15])**2))
            mse=float(np.mean((row['truth'][15]-final)**2))
            prediction_rows.append(dict(rank=rank,innovation_mean_log_ratio=float(np.mean(local_logs)),terminal_baseline_mse=baseline,terminal_surrogate_mse=mse,terminal_surrogate_log_ratio=math.log(mse/baseline),terminal_correction_rms=float(np.sqrt(np.mean(propagated**2)))))
            del w
        foldrows.append(dict(fold=fi,heldout=fold,innovation_mean_log_ratio=float(np.mean([r['innovation_mean_log_ratio']for r in prediction_rows if r['rank']in fold])),terminal_surrogate_mean_log_ratio=float(np.mean([r['terminal_surrogate_log_ratio']for r in prediction_rows if r['rank']in fold]))))
        print(json.dumps(dict(phase='fold',fold=fi,seconds=time.monotonic()-start)),flush=True)
        assert time.monotonic()-start<175
    ilogs=np.array([r['innovation_mean_log_ratio']for r in prediction_rows]); tlogs=np.array([r['terminal_surrogate_log_ratio']for r in prediction_rows])
    result=dict(status='FIXED_SAVED_CV_COMPLETE_NO_NATIVE_GAIN_CLAIM',script_sha256=sha(__file__),rule_sha256=sha(D/'FIXED_RULE_v1.json'),ridge=10.,shrink=.5,shared_parameters=95,networks=32,layers=list(range(7,15)),innovation_geometric_change_pct=100*math.expm1(float(ilogs.mean())),innovation_SE_log_pct=100*float(ilogs.std(ddof=1))/math.sqrt(32),innovation_networks_lower=int((ilogs<0).sum()),terminal_surrogate_geometric_change_pct=100*math.expm1(float(tlogs.mean())),terminal_surrogate_SE_log_pct=100*float(tlogs.std(ddof=1))/math.sqrt(32),terminal_surrogate_networks_lower=int((tlogs<0).sum()),terminal_surrogate_clears_point8=100*math.expm1(float(tlogs.mean()))<=-.8,per_network=sorted(prediction_rows,key=lambda x:x['rank']),per_fold=foldrows,per_layer=[dict(layer=l,geometric_innovation_change_pct=100*math.expm1(float(np.mean([r['log_mse_ratio']for r in records if r['layer']==l]))))for l in range(7,15)],rows=records,capture_pins=pins,seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,captures=0,exported_coefficients=False,limitation='Frozen gate-W propagation omits covariance recentering, GNN state, FHM and cumulant responses; terminal result is a diagnostic surrogate only. Captures use native QR parent 4d7739a5, preceding current Hermite regrouping. Runtime proposed is one shared 94-column dot per layer after centered-state construction; no second matrix propagation at deployment.',runtime_added_main_flops=8*(2*1024*94+8*1024))
    (D/'SAVED_RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('per_network','per_fold','per_layer','rows','capture_pins')}),flush=True)

if __name__=='__main__':main()
