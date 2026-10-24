"""Fixed ridge1 network-context residual atop frozen half-strength MLP; no estimator calls."""
from pathlib import Path
import hashlib,json,math,os,time
import numpy as np
D=Path(__file__).resolve().parent;M=D.parent/'astra_native_terminal_mlp16_v1';C=D.parent/'astra_native_terminal_linear_refit_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert os.getpriority(os.PRIO_PROCESS,0)==10
start=time.monotonic()
def write(name,x):
 with (D/name).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
p=C/'ROOT_CAPTURE_RETIREMENT_v1.json';assert sha(p)=='94dbc42175d4e7349217f6a3c37134ad83f454654fed4a17e1f58e27b6d6fde6'
r=json.loads(p.read_text());assert r['approved']and r['complete_calls']==32 and r['all_owned_PID_PPID_PGID_absent']
data=[]
for e in sorted(r['evaluations'],key=lambda x:x['seed']):
 assert sha(e['capture_path'])==e['capture_sha256']
 with np.load(e['capture_path'],allow_pickle=False)as z:
  x=z['features30'].astype('d');s=z['sigma'].astype('d');raw=z['raw_final_float64'].astype('d');y=z['target_final_float64'].astype('d');base=z['predictions_float32'][15].astype('d');state=z['state64'].astype('d');host=z['host'].astype('d');delta=z['delta'].astype('d')
 assert np.array_equal(np.maximum(raw,0).astype('f'),base)
 local=np.column_stack((np.ones(1024),x[:,5],x[:,6],x[:,3],raw/s,(x[:,3]**2-1)*x[:,6]))
 data.append(dict(rank=e['seed'],X=np.column_stack((x,state)),local=local,context=np.r_[x.mean(0),x.std(0)],s=s,raw=raw,y=y,host=host,delta=delta,base_mse=float(np.mean((base-y)**2))))
assert [d['rank']for d in data]==list(range(32))
def mlp_map(label):
 f=json.loads((M/f'{label}_FREEZE_v1.json').read_text());p=M/f'{label}_map_v1.npz';assert sha(p)==f['map_sha256']
 with np.load(p,allow_pickle=False)as z:a={k:z[k]for k in z.files}
 return a,f

def after_mlp(d,a):
 h=((d['X']-a['center'])/a['scale'])@a['w0'].T+a['b0'];h=h/(1+np.exp(-h));u=((h@a['w1'].T+a['b1']).reshape(-1)*float(a['target_rms']))*.5
 return d['host']+(d['delta']+d['s']*u)

def fit_context(train,a):
 cs=np.stack([d['context']for d in train]);cm=cs.mean(0);cv=np.maximum(cs.std(0),1e-6)
 def design(d):return (d['local'][:,:,None]*np.r_[1.,(d['context']-cm)/cv][None,None,:]).reshape(1024,-1)
 X=np.concatenate([design(d)for d in train]);xm=X.mean(0);xs=np.maximum(X.std(0),1e-6);X=np.column_stack((np.ones(len(X)),(X-xm)/xs))
 s=np.concatenate([d['s']for d in train]);target=np.concatenate([(d['y']-after_mlp(d,a))/d['s']for d in train]);w=s*s;w/=w.mean()
 G=(X.T*w)@X/len(X);b=X.T@(w*target)/len(X);pen=np.eye(len(b));pen[0,0]=0;beta=np.linalg.solve(G+pen,b)
 theta=(beta[1:]/xs).reshape(6,61);theta[0,0]+=beta[0]-np.sum(xm*beta[1:]/xs)
 W=np.empty((61,6));W[0]=theta[:,0]-theta[:,1:]@(cm/cv);W[1:]=(theta[:,1:]/cv).T
 for d in train[:1]:
  expanded=np.column_stack((np.ones(1024),(design(d)-xm)/xs))@beta;compact=d['local']@(np.r_[1.,d['context']]@W)
  assert np.allclose(expanded,compact,rtol=1e-10,atol=1e-12)
 return W

def score(d,raw):return float(np.mean((np.maximum(raw,0).astype('f').astype('d')-d['y'])**2))
rows=[];pins={};folds=[]
for fold in range(8):
 a,f=mlp_map(f'fold{fold}');train=[d for d in data if d['rank']//4!=fold];held=[d for d in data if d['rank']//4==fold];assert f['fit_ranks']==[d['rank']for d in train]
 W=fit_context(train,a);mp=D/f'fold{fold}_context_v1.npz';np.savez(mp,W=W);pins[str(mp)]=sha(mp)
 for d in held:
  raw=after_mlp(d,a);base=score(d,raw);stack=score(d,raw+d['s']*(d['local']@(np.r_[1.,d['context']]@W)))
  rows.append(dict(rank=d['rank'],mlp_log_vs_QR=math.log(base/d['base_mse']),stack_log_vs_QR=math.log(stack/d['base_mse']),stack_log_vs_MLP=math.log(stack/base)))
 folds.append(dict(fold=fold,map_sha256=f['map_sha256'],training_ranks=f['fit_ranks']))
 assert time.monotonic()-start<120
summary={k:float(np.mean([r[k]for r in rows]))for k in ('mlp_log_vs_QR','stack_log_vs_QR','stack_log_vs_MLP')};ok=summary['stack_log_vs_QR']<0 and summary['stack_log_vs_MLP']<0;final=None
if ok:
 a,f=mlp_map('all32');assert f['fit_ranks']==list(range(32));W=fit_context(data,a);mp=D/'all32_context_map_v1.npz';np.savez(mp,W=W);final=dict(path=str(mp),sha256=sha(mp),mlp_map_sha256=f['map_sha256'],ridge=1.,shrink=.5)
result=dict(status='PASS_BLOCKED_CV_REQUIRES_INDEPENDENT_NATIVE'if ok else'CLOSED_BLOCKED_CV_FAIL',summary_log=summary,summary_geometric_pct={k:100*math.expm1(v)for k,v in summary.items()},per_network=rows,folds=folds,fold_context_sha256=pins,final=final,source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start,no_estimator_calls=True,no_new_captures=True,no_independent_gain_claim=True)
write('STACK_CONTEXT_CV_RESULT_v1.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('per_network','folds','fold_context_sha256')}))
