"""Exact final-error birth decomposition vs MC truth (production versions only). e_l=pred-truth; b_l=e_l-J_l e_{l-1}; c_l=A_{l->15} b_l."""
import json,sys,numpy as np
from pathlib import Path
from scipy.stats import norm
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W.parents[2]/'analyst/fresh'));import loader
H=json.loads((W/'HARVEST_v1.json').read_text())
tg=np.linspace(-6,6,4001);mm=norm.pdf(tg)+tg*norm.cdf(tg);e2=(1+tg**2)*norm.cdf(tg)+tg*norm.pdf(tg);rg=mm**2/e2
PRI=['b7090c35','2b10843a','204880a5']
out=[]
for s,recs in H.items():
    rec=None
    for sha in PRI:
        c=[r for r in recs if r['sha']==sha]
        if c:rec=max(c,key=lambda x:x['mtime']);break
    if rec is None:continue
    p=Path(rec['path']);pp=p.with_name(p.stem+'_pred.npz')
    if not pp.exists():continue
    with np.load(pp) as z:y=z[s].astype(np.float64)
    t=loader.truth(int(s),check_weights=False);w=loader.weights(int(s));mp,vp=t['mc_post'],t['mc_post_var']
    e=y-mp;assert abs((e[15]**2).mean()/rec['mse'][15]-1)<1e-3
    Phi=[norm.cdf(np.interp(np.clip(mp[l]**2/np.maximum(vp[l]+mp[l]**2,1e-300),rg[0],rg[-1]),rg,tg)) for l in range(16)]
    J=lambda l,v:Phi[l]*(w[l].T.astype(np.float64)@v)
    born=[e[0]]+[e[l]-J(l,e[l-1]) for l in range(1,16)]
    contrib=[]
    for l in range(16):
        v=born[l]
        for k in range(l+1,16):v=J(k,v)
        contrib.append(v)
    C=np.array(contrib);e15=e[15];tot=(e15**2).sum()
    out.append(dict(seed=int(s),sha=rec['sha'],mse15=rec['mse'][15],scale=float(vp[15].mean()),proj_share=[float(c@e15/tot) for c in C],own_share=[float((c**2).sum()/tot) for c in C],remainder=float(((C.sum(0)-e15)**2).sum()/tot)))
json.dump(out,open(W/'BIRTH_v1.json','x'))
print(len(out),'seeds; mean linearization remainder %.4f'%np.mean([r['remainder'] for r in out]))
y=np.log([r['mse15'] for r in out]);X=np.column_stack([np.ones(len(out)),np.log([r['scale'] for r in out])]);res=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
P=np.array([r['proj_share'] for r in out]);q=np.argsort(res);n=len(out)//3;hi,lo=q[-n:],q[:n]
print('layer  proj-share(all)  hi-res  lo-res   corr(share*MSEnorm, resid)')
M=np.exp(res)
for l in range(16):print('%2d   %+.3f   %+.3f   %+.3f   %+.3f'%(l,P[:,l].mean(),P[hi,l].mean(),P[lo,l].mean(),np.corrcoef(P[:,l]*M,res)[0,1]))
