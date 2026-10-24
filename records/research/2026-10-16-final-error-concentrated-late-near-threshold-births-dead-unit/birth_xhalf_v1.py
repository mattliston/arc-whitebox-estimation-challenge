"""Noise-free (cross-half) final error + birth decomposition. U=mean((y-A)(y-B)); born from half A, projected on e15 vs half B."""
import json,sys,numpy as np
from pathlib import Path
from scipy.stats import norm
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W.parents[2]/'analyst/fresh'));import loader
H=json.loads((W/'HARVEST_v1.json').read_text());F={r['seed']:r for r in json.loads((W/'FEATURES_v1.json').read_text())}
tg=np.linspace(-6,6,4001);mm=norm.pdf(tg)+tg*norm.cdf(tg);e2=(1+tg**2)*norm.cdf(tg)+tg*norm.pdf(tg);rg=mm**2/e2
PRI=['b7090c35','2b10843a','204880a5']
out=[]
for s,recs in H.items():
    rec=None
    for sha in PRI+[None]:
        c=[r for r in recs if (sha is None or r['sha']==sha)]
        if c:rec=max(c,key=lambda x:x['mtime']);break
    p=Path(rec['path']);pp=p.with_name(p.stem+'_pred.npz')
    if not pp.exists():continue
    with np.load(pp) as z:
        if s not in z.files:continue
        y=z[s].astype(np.float64)
    t=loader.truth(int(s),check_weights=False);A,B,mp,vp=t['mc_post_A'],t['mc_post_B'],t['mc_post'],t['mc_post_var']
    U=float(((y[15]-A[15])*(y[15]-B[15])).mean());r=dict(seed=int(s),sha=rec['sha'],prod=rec['sha'] in PRI,U15=U,mse15=rec['mse'][15],scale=float(vp[15].mean()))
    if rec['sha'] in PRI:
        w=loader.weights(int(s));Phi=[norm.cdf(np.interp(np.clip(mp[l]**2/np.maximum(vp[l]+mp[l]**2,1e-300),rg[0],rg[-1]),rg,tg)) for l in range(16)]
        J=lambda l,v:Phi[l]*(w[l].T.astype(np.float64)@v)
        eA=y-A;eB15=y[15]-B[15];born=[eA[0]]+[eA[l]-J(l,eA[l-1]) for l in range(1,16)];sh=[]
        for l in range(16):
            v=born[l]
            for k in range(l+1,16):v=J(k,v)
            sh.append(float(v@eB15)/(U*1024))
        r['xshare']=sh
    out.append(r)
json.dump(out,open(W/'XHALF_v1.json','x'))
P=[r for r in out if 'xshare' in r];print(len(out),'seeds;',len(P),'production with birth')
print('noise-free/local final MSE: mean %.3f'%np.mean([r['U15']/r['mse15'] for r in out]))
yv=np.log([r['U15'] for r in out]);X=np.column_stack([np.ones(len(out)),np.log([r['scale'] for r in out])]);b=np.linalg.lstsq(X,yv,rcond=None)[0];res=yv-X@b
print('noise-free: slope on log scale %.3f, resid sd %.3f'%(b[1],res.std()))
cands=[]
for k in F[out[0]['seed']]:
    if k in('seed','mse15','mse_layers','sha','scale_var15','scale_m2_15'):continue
    x=np.array([F[r['seed']][k] for r in out]);
    if x.std()>0:cands.append((abs(np.corrcoef(x,res)[0,1]),k,np.corrcoef(x,res)[0,1]))
for a,k,c in sorted(cands,reverse=True)[:8]:print('  %-10s r %+.3f'%(k,c))
pr=np.array([r['xshare'] for r in P]);rp=np.array([res[i] for i,r in enumerate(out) if 'xshare' in r]);q=np.argsort(rp);n=len(P)//3
print('layer xshare all / hi-res / lo-res');print(' '.join('%d:%+.3f/%+.3f/%+.3f'%(l,pr[:,l].mean(),pr[q[-n:],l].mean(),pr[q[:n],l].mean()) for l in range(16)))
print('sum shares %.3f'%pr.sum(1).mean())
