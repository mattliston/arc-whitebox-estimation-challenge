"""One frozen tiny learned residual MLP; grouped independent-half validation."""
from pathlib import Path
import hashlib,json,math,resource,signal,time
import numpy as np

D=Path(__file__).resolve().parent; R=D.parent; O=R.parents[1]
H=R/'opus_headfit_v1'; START=time.monotonic()
PLAN=dict(architecture=[64,16,1],activation='SiLU',output='sigma * TRAIN residual/sigma RMS * scalar',
          input='state64 only; encoder/messages absent in146 saved captures',
          folds='sorted original TRAIN97[i::5]; held49 unopened unless meanCV<=-.5% and4/5folds improve',
          steps=400,batch=8192,optimizer='AdamW',learning_rate=.003,weight_decay=.01,
          adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,seed=20261016,
          standardization='fold-TRAIN input mean/std; minimum std1e-6, no label features',
          loss='average squared residual against(A+B)/2 weighted sigma²/networkAB_U, normalized weight mean1',
          initial_output='zero; hidden weights normal std1/sqrt64; hidden biases zero',
          no_grid=True,no_new_data=True,no_native=True,wall_cap_s=300,rss_cap_bytes=3_000_000_000)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(seed):
    p=H/f'cap/head_{seed}.npz'
    with np.load(p,allow_pickle=False) as z:
        y=z['y15'].astype(float);s=z['sigma'].astype(float);x=z['state'].astype('f4')
    assert x.shape==(1024,64) and np.isfinite(x).all() and np.all(s>0)
    tp=O/f'analyst/fresh/truth_seed{seed}.npz'
    with np.load(tp,allow_pickle=False) as z:A=z['mc_post_A'][-1].astype(float);B=z['mc_post_B'][-1].astype(float)
    U0=float(np.mean((y-A)*(y-B)));assert U0>0
    return dict(seed=seed,y=y,sigma=s,state=x,A=A,B=B,T=(A+B)/2,U0=U0,
                input_path=str(p),input_sha256=sha(p),truth_path=str(tp),truth_sha256=sha(tp))

def fit(rr,fold):
    rng=np.random.default_rng(PLAN['seed']+fold)
    X=np.concatenate([r['state'] for r in rr]);mean=X.mean(0,dtype='f8');std=X.std(0,dtype='f8').clip(1e-6)
    X=((X-mean)/std).astype('f4')
    t=np.concatenate([(r['T']-r['y'])/r['sigma'] for r in rr]);ts=float(np.sqrt(np.mean(t*t)))
    t=(t/ts).astype('f4')
    w=np.concatenate([r['sigma']**2/r['U0'] for r in rr]);w=(w/w.mean()).astype('f4')
    params=[rng.normal(0,1/8,(64,16)).astype('f4'),np.zeros(16,'f4'),np.zeros(16,'f4'),np.zeros(1,'f4')]
    moments=[np.zeros_like(p) for p in params];variances=[np.zeros_like(p) for p in params]
    losses=[]
    for step in range(1,PLAN['steps']+1):
        ii=rng.integers(0,len(X),PLAN['batch']);x=X[ii];target=t[ii];weight=w[ii]
        W,b,v,c=params;z=x@W+b
        sigmoid=1/(1+np.exp(-np.clip(z,-60,60)));h=z*sigmoid;pred=h@v+c[0]
        residual=pred-target; grad=2*weight*residual/len(ii)
        gz=grad[:,None]*v[None,:]*(sigmoid+z*sigmoid*(1-sigmoid))
        gradients=[x.T@gz,gz.sum(0),h.T@grad,np.array([grad.sum()],dtype='f4')]
        for j,(p,g) in enumerate(zip(params,gradients)):
            moments[j]*=.9;moments[j]+=.1*g
            variances[j]*=.999;variances[j]+=.001*g*g
            p*=1-PLAN['learning_rate']*PLAN['weight_decay']
            p-=PLAN['learning_rate']*(moments[j]/(1-.9**step))/(np.sqrt(variances[j]/(1-.999**step))+1e-8)
        if step in (1,100,200,400):losses.append([step,float(np.mean(weight*residual*residual))])
        if step%100==0:
            assert all(np.isfinite(p).all() for p in params)
            assert time.monotonic()-START<285
    return dict(W=params[0],b=params[1],v=params[2],c=params[3],center=mean,scale=std,target_scale=np.array(ts)),losses

def score(r,m):
    x=((r['state']-m['center'])/m['scale']).astype('f4');z=x@m['W']+m['b']
    h=z/(1+np.exp(-np.clip(z,-60,60)));out=h@m['v']+m['c'][0]
    dx=r['sigma']*float(m['target_scale'])*out.astype(float)
    y=np.maximum(r['y']+dx,0).astype('f4').astype(float)
    U=float(np.mean((y-r['A'])*(y-r['B'])));assert U>0
    return dict(seed=r['seed'],U0=r['U0'],U=U,log_ratio=math.log(U/r['U0']),shift_RMS=float(np.sqrt(np.mean(dx*dx))))

def summary(rows):
    lr=np.array([r['log_ratio'] for r in rows])
    return dict(n=len(rows),geometric_AB_U_pct=float(100*np.expm1(lr.mean())),
                log_SE_pct=float(100*lr.std(ddof=1)/np.sqrt(len(lr))),wins=int(sum(lr<0)),
                pooled_AB_U_pct=float(100*(sum(r['U'] for r in rows)/sum(r['U0'] for r in rows)-1)))

def main():
    assert not (D/'RESULT_v1.json').exists()
    (D/'PREREG_v1.json').write_text(json.dumps(PLAN,indent=2)+'\n')
    split=H/'REFIT_head15_v1.npz'
    with np.load(split,allow_pickle=False) as z:train=sorted(z['train'].tolist());held=sorted(z['held'].tolist())
    assert len(train)==97 and len(held)==49 and not set(train)&set(held)
    data={s:load(s) for s in train};folds=[train[i::5] for i in range(5)];rows=[];fr=[]
    for i,fold in enumerate(folds):
        m,losses=fit([data[s] for s in train if s not in fold],i)
        rr=[score(data[s],m) for s in fold];rows+=rr
        np.savez(D/f'fold{i}_v1.npz',**m)
        result=dict(fold=i,networks=fold,losses=losses,**summary(rr));fr.append(result)
        print(json.dumps(result),flush=True)
    cv=summary(rows);passed=cv['geometric_AB_U_pct']<=-.5 and sum(f['geometric_AB_U_pct']<0 for f in fr)>=4
    out=dict(status='TRAIN_PASS_PENDING_HELD' if passed else 'CLOSED_TRAIN_CV',CV=cv,folds=fr,per_network=rows,train=train,held=held,held_truth_opened=False,
             pins=[{k:r[k] for k in ['seed','input_path','input_sha256','truth_path','truth_sha256']} for r in data.values()],
             split_sha256=sha(split),script_sha256=sha(__file__),prereg_sha256=sha(D/'PREREG_v1.json'))
    (D/'TRAIN_CV_RESULT_v1.json').write_text(json.dumps(out,indent=2)+'\n')
    if passed:
        m,losses=fit(list(data.values()),5);rr=[score(load(s),m) for s in held];hs=summary(rr)
        np.savez(D/'all_train_v1.npz',**m)
        out.update(status='HELD_PASS_NEEDS_CURRENT_NATIVE' if hs['geometric_AB_U_pct']<=-.5 else 'CLOSED_HELD',held_truth_opened=True,held_summary=hs,held_per_network=rr,final_fit_losses=losses)
    out.update(seconds=time.monotonic()-START,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               no_native=True,no_GPU=True,no_new_data=True,runtime_estimate=dict(
                   matmul_flops=2*1024*64*16+2*1024*16,elementwise_flops_at_most=10*1024*16,
                   approximate_total_flops=2*1024*64*16+12*1024*16,
                   public_requests_estimate=9,notes='Center/std folds intoW,b. SiLU exp/add/divide/multiply plus scalar head and sigma; exact tariff ifpositive.'))
    assert out['seconds']<300 and out['max_RSS_bytes']<3_000_000_000
    (D/'RESULT_v1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['status','CV','held_truth_opened','seconds','max_RSS_bytes']},indent=2),flush=True)

if __name__=='__main__':
    signal.alarm(300)
    main()
