"""Noise-free born energy per unit at L12-15 by Gaussian-equivalent threshold t (cross-half product b_A*b_B)."""
import json,sys,numpy as np
from pathlib import Path
from scipy.stats import norm
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W.parents[2]/'analyst/fresh'));import loader
X=[r for r in json.loads((W/'XHALF_v1.json').read_text()) if 'xshare' in r];H=json.loads((W/'HARVEST_v1.json').read_text())
tg=np.linspace(-6,6,4001);mm=norm.pdf(tg)+tg*norm.cdf(tg);e2=(1+tg**2)*norm.cdf(tg)+tg*norm.pdf(tg);rg=mm**2/e2
bins=np.array([-9,-2,-1,-0.5,0,0.5,1,2,9]);rows=[]
for r in X:
    s=str(r['seed']);rec=max([x for x in H[s] if x['sha']==r['sha']],key=lambda x:x['mtime']);p=Path(rec['path'])
    with np.load(p.with_name(p.stem+'_pred.npz')) as z:y=z[s].astype(np.float64)
    t=loader.truth(r['seed'],check_weights=False);A,B,mp,vp=t['mc_post_A'],t['mc_post_B'],t['mc_post'],t['mc_post_var'];w=loader.weights(r['seed'])
    tt=[np.interp(np.clip(mp[l]**2/np.maximum(vp[l]+mp[l]**2,1e-300),rg[0],rg[-1]),rg,tg) for l in range(16)]
    E={}
    for l in (12,13,14,15):
        bA=(y[l]-A[l])-norm.cdf(tt[l])*(w[l].T.astype(np.float64)@(y[l-1]-A[l-1]));bB=(y[l]-B[l])-norm.cdf(tt[l])*(w[l].T.astype(np.float64)@(y[l-1]-B[l-1]))
        prod=bA*bB;k=np.digitize(tt[l],bins)-1;E[l]=[float(prod[k==j].sum()) for j in range(8)]+[float((k==j).mean()) for j in range(8)]
    rows.append(dict(seed=r['seed'],E=E))
json.dump(rows,open(W/'BORN_TBINS_v1.json','x'))
for l in (12,13,14,15):
    M=np.array([r['E'][l] for r in rows]);tot=M[:,:8].sum(1,keepdims=True);sh=(M[:,:8]/tot).mean(0);fr=M[:,8:].mean(0)
    print('L%d born-energy share by t-bin'%l,np.round(sh,3),' unit frac',np.round(fr,3))
print('bins',[f'{bins[j]}..{bins[j+1]}' for j in range(8)])
