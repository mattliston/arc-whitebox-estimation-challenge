"""One fixed saved-TRAIN32 terminal Jensen+Cauchy clipping diagnostic; no model/fit."""
import os, json, hashlib, subprocess, re
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
O=D.parents[2]
capture_receipt=O/'lead/runs/astra_native_terminal_linear_refit_v1/ROOT_CAPTURE_RETIREMENT_v1.json'
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(capture_receipt)=='94dbc42175d4e7349217f6a3c37134ad83f454654fed4a17e1f58e27b6d6fde6'
assert os.getpriority(os.PRIO_PROCESS,0)==10
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    assert os.environ[k]=='1'
pressure=subprocess.check_output(['/usr/bin/memory_pressure'],text=True)
free=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1))
assert free>=50
cap=json.loads(capture_receipt.read_text())
assert cap['approved'] and cap['all_owned_PID_PPID_PGID_absent'] and cap['all_guards_clean'] and cap['complete_calls']==32
rows=[]; evidence={}
for e in sorted(cap['evaluations'],key=lambda e:e['seed']):
    p=Path(e['capture_path'])
    assert sha(p)==e['capture_sha256']; evidence[str(p)]=e['capture_sha256']
    with np.load(p,allow_pickle=False) as z:
        alpha=z['features30'][:,3].copy()
        sigma=z['sigma'].copy()
        raw=z['raw_final_float64'].copy()
        pred=z['predictions_float32'][15].copy()
        truth=z['target_final_float64'].copy()
    assert all(np.isfinite(a).all() for a in (alpha,sigma,raw,pred,truth))
    assert np.all(sigma>0)
    assert np.array_equal(pred,np.maximum(raw,0).astype(np.float32))
    mu=alpha.astype(np.float64)*sigma
    lo=np.maximum(mu,0)
    hi=(mu+np.hypot(mu,sigma))*0.5
    assert np.all(lo<=hi)
    fixed=np.minimum(np.maximum(raw,lo),hi).astype(np.float32)
    # Full F32 ULP, conservative relative to round-to-nearest alpha quantization.
    alpha_ulp=np.maximum(np.abs(np.nextafter(alpha,np.float32(np.inf)).astype(np.float64)-alpha),
                         np.abs(alpha.astype(np.float64)-np.nextafter(alpha,np.float32(-np.inf))))
    mu_uncertainty=alpha_ulp*sigma
    baseline_mse=float(np.mean((pred.astype(np.float64)-truth)**2))
    fixed_mse=float(np.mean((fixed.astype(np.float64)-truth)**2))
    lower=pred.astype(np.float64)<lo
    upper=pred.astype(np.float64)>hi
    row=dict(seed=e['seed'],n=len(pred),lower_violations=int(lower.sum()),upper_violations=int(upper.sum()),
        robust_lower_violations=int((pred.astype(np.float64)<np.maximum(mu-mu_uncertainty,0)).sum()),
        robust_upper_violations=int((pred.astype(np.float64)>(mu+mu_uncertainty+np.hypot(mu+mu_uncertainty,sigma))*0.5).sum()),
        changed_float32=int((fixed!=pred).sum()),baseline_mse=baseline_mse,fixed_mse=fixed_mse,
        log_ratio=float(np.log(fixed_mse/baseline_mse)),max_absolute_change=float(np.max(np.abs(fixed.astype(np.float64)-pred))),
        max_mu_quantization_uncertainty=float(mu_uncertainty.max()))
    rows.append(row)
logs=np.array([r['log_ratio'] for r in rows]); n=sum(r['n'] for r in rows)
summary=dict(networks=len(rows),neurons=n,lower_violations=sum(r['lower_violations'] for r in rows),
    upper_violations=sum(r['upper_violations'] for r in rows),
    robust_lower_violations=sum(r['robust_lower_violations'] for r in rows),
    robust_upper_violations=sum(r['robust_upper_violations'] for r in rows),
    changed_float32=sum(r['changed_float32'] for r in rows),
    networks_changed=sum(r['changed_float32']>0 for r in rows),
    networks_lower_mse=int((logs<0).sum()),meanlog_mse_pct=float(logs.mean()*100),
    geometric_mse_pct=float(np.expm1(logs.mean())*100),se_log_pct=float(logs.std(ddof=1)/np.sqrt(len(logs))*100),
    pooled_mse_pct=float((sum(r['fixed_mse'] for r in rows)/sum(r['baseline_mse'] for r in rows)-1)*100),
    max_absolute_change=max(r['max_absolute_change'] for r in rows),
    max_mu_quantization_uncertainty=max(r['max_mu_quantization_uncertainty'] for r in rows))
result=dict(status='ONE_FIXED_RULE_COMPLETE',rule='clip(raw_final,max(mu,0),(mu+hypot(mu,sigma))/2), then float32',
    no_models_or_training=True,source_family='QR4d captured parent; pre-HF reassociation, not exact latest 1c913',
    mu_source='raw features30[:,3] (alpha float32) times sigma float64',
    variance_source='sigma squared, current floored approximate preactivation variance',
    caveat='Moment bounds are guaranteed only for true moments; approximate mu/variance can make clipping harmful. Alpha reconstruction is quantized.',
    prior_distinct_closure='Hidden L1-14 Jensen lower-only +0.407059% adjusted; not this terminal simultaneous pair.',
    evidence_sha256={str(capture_receipt):sha(capture_receipt),**evidence},
    script_sha256=sha(__file__),entry_free_pct=free,summary=summary,per_network=rows)
out=D/'FIXED_TERMINAL_BOUNDS_RESULT_v1.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,sort_keys=True)); print('result_sha256='+sha(out))

