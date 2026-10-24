"""Judge one sealed source against reused independent fourth truth replicas."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ASTRA=HERE.parents[1]/'astra_phase2_reexamination_v1'
sys.path.insert(0,str(ASTRA))
from current_pair_oracle_math_v1 import exact_change,independent_fourth_map


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()


start=time.monotonic();assert not (HERE/'assessment.json').exists()
seal=json.loads((HERE/'seal.json').read_text())
for path,digest in seal['files'].items():assert sha(Path(path))==digest,path
rec=json.loads((HERE/'capture.json').read_text());assert rec['passed']
assert rec['output_sha256']==sha(HERE/'capture.npz')
cg=json.loads((HERE/'capture_guard.json').read_text())
assert cg['completed'] and cg['whole_group_retired'] and cg['max_swap_growth_bytes']==0
assert json.loads((ASTRA/'current_pair_oracle_certificate_v1.json').read_text())['passed']
cap=dict(np.load(HERE/'capture.npz',allow_pickle=False))
input_path=ASTRA/'fourth_trace_oracle_inputs_r35_v1.npz'
input_hash=sha(input_path);assert input_hash==json.loads(input_path.with_suffix('.json').read_text())['output_sha256']
f=dict(np.load(input_path,allow_pickle=False));s=f['pre13_sigma']
scale4=s**4;scale31=s[:,None]**3*s[None,:];scale22=np.outer(s*s,s*s)
R=f['pre13_Cpre']/np.outer(s,s)
common=(R,s,f['pre13_alpha'],f['pre13_pdf'],f['pre13_gate'],f['pre13_gaussian_mean']/s,f['pre13_m']/s)
fn=independent_fourth_map();base31=f['pre13_k31']/scale31;np.fill_diagonal(base31,0)
basem,baseC,*_=fn(f['pre13_mu'],f['pre13_Cpre'],f['pre13_K21'],f['pre13_k4'],f['pre13_k22'],'covariance',q31_override=base31)
baseline_error=max(float(np.max(np.abs(basem-f['pre13_m']))),float(np.max(np.abs(baseC-f['pre13_C']))))
assert baseline_error<=1e-12
w=f['W14'].astype(np.float64);old=np.load(ASTRA/'fourth_trace_oracle_value_v1.npz',allow_pickle=False)


def change(k4,q31,q22):
    a=k4/scale4;b=q31/scale31;c=q22/scale22
    np.fill_diagonal(b,0);np.fill_diagonal(c,0)
    dm,dc=exact_change(common,dQ22=c,dQ31=b,dq4=a)
    cm,cc,*_=fn(f['pre13_mu'],f['pre13_Cpre'],f['pre13_K21'],f['pre13_k4']+k4,
               f['pre13_k22']+q22,'covariance',q31_override=base31+b)
    error=max(float(np.max(np.abs(cm-basem-dm))),float(np.max(np.abs((cc-baseC+cc.T-baseC.T)/2-dc))))
    assert error<=1e-12,error
    return dict(mean=dm,covariance=dc,next_mean=np.sum(w*(dc@w),axis=0)*f['response14']),error


def metrics(a,b,p):
    signal=float(np.sum(a*b));residual=float(np.sum((a-p)*(b-p)))
    risks=[float(np.sum((a-p)**2)/np.sum(a*a)),float(np.sum((b-p)**2)/np.sum(b*b))]
    ratio=residual/signal if signal>0 else None
    return dict(entries=a.size,target_cross_energy=signal,residual_cross_energy=residual,
                residual_cross_ratio=ratio,ordinary_risks=risks,
                source_energy=float(np.sum(p*p)),source_target_cross=float(np.sum(p*(a+b)/2)),
                passed=signal>0 and ratio<=.70 and max(risks)<1)


ii,jj=np.triu_indices(len(s),1)


def faces(k4,q31,q22):
    a=k4/scale4;b=q31/scale31;c=(q22+q22.T)/2/scale22
    directed=np.concatenate((b[ii,jj],b[jj,ii]))
    return dict(k4=a,q31=directed,q22=c[ii,jj],
                joint=np.concatenate((a,2*directed,np.sqrt(6)*c[ii,jj])))


zero=np.zeros_like(cap['delta31'])
pred,pred_error=change(cap['delta4'],cap['delta31'],zero)
pred_faces=faces(cap['delta4'],cap['delta31'],zero)
target={};target_faces={};records=[];seeds=[]
saved={f'prediction_{k}':v for k,v in pred.items()}
for stream in ('A','B'):
    path=ASTRA/f'fourth_trace_oracle_r35_{stream}_v1.npz'
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['passed'] and receipt['frozen_inputs_sha256']==input_hash and receipt['output_sha256']==sha(path)
    t=dict(np.load(path,allow_pickle=False));assert int(t['P'])==8000000
    seeds.append(int(t['seed_xor']))
    raw=(t['k4']-f['pre13_k4'],t['Q31']-f['pre13_k31'],t['Q22']-f['pre13_k22'])
    target[stream],error=change(*raw);target_faces[stream]=faces(*raw)
    reproduce=max(float(np.max(np.abs(target[stream][k]-old[f'target_{stream}_{k}']))) for k in pred)
    assert reproduce<=1e-12,reproduce
    records.append(dict(stream=stream,seed_xor=seeds[-1],direct_consumer_error=error,old_target_error=reproduce))
    for k,v in target[stream].items():saved[f'target_{stream}_{k}']=v
assert seeds[0]!=seeds[1]
consumer={k:metrics(target['A'][k],target['B'][k],pred[k]) for k in pred}
face_scores={k:metrics(target_faces['A'][k],target_faces['B'][k],pred_faces[k]) for k in pred_faces}
passed=all(consumer[k]['passed'] for k in ('covariance','next_mean'))
np.savez(HERE/'values.npz',**saved)
result=dict(passed=passed,consumers=consumer,faces=face_scores,records=records,
            baseline_direct_map_error=baseline_error,prediction_direct_map_error=pred_error,
            capture_sha256=sha(HERE/'capture.npz'),source_seal_sha256=sha(HERE/'seal.json'),
            values_sha256=sha(HERE/'values.npz'),wall_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc_finished=datetime.now(timezone.utc).isoformat(),
            qualification='One reused old rank35/pre13 state; target-free leading covariance-times-retained-third source, own scalar trace removed, k4/Q31 consumers only. Possible shared MPS/finite-consumer bias. Not an S31 endpoint or a complete fourth recurrence.',
            new_samples=False,candidate_built=False,check32_accessed=False,valid40_accessed=False)
with (HERE/'assessment.json').open('x') as o:json.dump(result,o,indent=2,allow_nan=False);o.write('\n')
print(json.dumps(result,indent=2),flush=True)
