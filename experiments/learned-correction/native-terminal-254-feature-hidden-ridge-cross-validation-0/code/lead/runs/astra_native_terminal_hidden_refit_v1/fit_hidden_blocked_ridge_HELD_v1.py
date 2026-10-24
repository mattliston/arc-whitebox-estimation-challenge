"""One fixed hidden128+encoder126 ridge CV after all TRAIN32 owners retire."""
from pathlib import Path
import json,hashlib,math,os,time
import numpy as np
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert os.getpriority(os.PRIO_PROCESS,0)==10
start=time.monotonic();receipt=json.loads((D/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text());assert receipt['approved']and receipt['reviewed_by']=='root'and receipt['complete_calls']==32 and receipt['actual_managed_exits_observed']and receipt['all_owned_PID_PPID_PGID_absent']and receipt['all_guards_clean']
c=json.loads((D/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text());rule=json.loads((D/'FIT_RULE_FROZEN_v1.json').read_text());assert c['fit_rule']==rule and rule['ridge_grid']==[.1,1.,10.,100.,1000.,10000.]
records=[]
for item in sorted(receipt['evaluations'],key=lambda e:e['seed']):
 assert sha(item['capture_path'])==item['capture_sha256'];ref=c['network_metadata'][item['seed']]['reference'];assert sha(ref['capture_path'])==ref['capture_sha256']
 with np.load(item['capture_path'],allow_pickle=False)as z:x=np.column_stack((z['hidden128'],z['encoder_input'])).astype('d')
 with np.load(ref['capture_path'],allow_pickle=False)as z:
  sigma=z['sigma'].astype('d');raw=z['raw_final_float64'].astype('d');target=z['target_final_float64'].astype('d');base=z['predictions_float32'][15].astype('d')
 assert x.shape==(1024,254)and np.isfinite(x).all()and np.all(sigma>0)and np.array_equal(np.maximum(raw,0).astype('f'),base)
 records.append(dict(rank=item['seed'],x=x,sigma=sigma,raw=raw,target=target,baseline_mse=float(np.mean((base-target)**2))))
assert [r['rank']for r in records]==list(range(32))
def system(rs):
 X=np.concatenate([r['x']for r in rs]);sigma=np.concatenate([r['sigma']for r in rs]);target=np.concatenate([(r['target']-r['raw'])/r['sigma']for r in rs]);center=X.mean(0);scale=np.maximum(X.std(0),1e-6);Z=np.column_stack((np.ones(len(X)),(X-center)/scale));w=sigma*sigma;w/=w.mean();return (Z.T*w)@Z/len(Z),Z.T@(w*target)/len(Z),center,scale

def solve(s,ridge):
 G,b,center,scale=s;pen=np.eye(len(b))*ridge;pen[0,0]=0;beta=np.linalg.solve(G+pen,b);coef=beta[1:]/scale;bias=float(beta[0]-center@coef);return coef,bias

def evaluate(r,coef,bias):
 delta=(r['x'][:,:128]@coef[:128]+r['x'][:,128:]@coef[128:])+bias;pred=np.maximum(r['raw']+r['sigma']*delta,0).astype('f').astype('d');mse=float(np.mean((pred-r['target'])**2));return dict(rank=r['rank'],mse=mse,baseline_mse=r['baseline_mse'],log_ratio=math.log(mse/r['baseline_mse']))
scores={r:[]for r in rule['ridge_grid']}
for f in range(8):
 held=list(range(4*f,4*f+4));s=system([r for r in records if r['rank']not in held])
 for ridge in rule['ridge_grid']:
  coef,bias=solve(s,ridge);scores[ridge].extend(evaluate(r,coef,bias)for r in records if r['rank']in held)
 assert time.monotonic()-start<120
summary=[dict(ridge=None,mean_log=0.,geometric_MSE_pct=0.)]
for ridge in sorted(scores,reverse=True):
 rows=scores[ridge];mean=float(np.mean([r['log_ratio']for r in rows]));summary.append(dict(ridge=ridge,mean_log=mean,geometric_MSE_pct=100*math.expm1(mean),networks_lower=sum(r['log_ratio']<0 for r in rows)))
winner=min(summary,key=lambda r:r['mean_log']);final=None
if winner['geometric_MSE_pct']<=-.5:
 coef,bias=solve(system(records),winner['ridge']);p=D/'FROZEN_HIDDEN_COEFFICIENTS_v1.npz';assert not p.exists();np.savez(p,coefficient254=coef,bias=np.asarray(bias));final=dict(path=str(p),sha256=sha(p))
status='CV_TARGET_PASSED_FROZEN_FIT'if final else'RETAIN_SMALL_NEGATIVE_FIT_PROPOSAL'if winner['mean_log']<0 else'BASELINE_ZERO_SELECTED_CLOSE'
r=dict(status=status,selected=winner,all_cv=summary,per_network={str(k):v for k,v in scores.items()},features='hidden128 then encoder126; h1 excluded',final=final,capture_retirement_sha256=sha(D/'ROOT_CAPTURE_RETIREMENT_v1.json'),source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start,no_independent_gain_claim=True,no_model_calls=True,no_variants=True)
with(D/'HIDDEN_BLOCKED_CV_RESULT_v1.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in r.items()if k!='per_network'}))
