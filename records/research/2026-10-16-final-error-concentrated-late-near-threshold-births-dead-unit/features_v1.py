"""Per-seed structure features from regenerated weights + MC truth (mean/var only). Read-only; no model calls."""
import json,sys,math
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
W=Path(__file__).resolve().parent;O=W.parents[2];sys.path.insert(0,str(O/'analyst/fresh'));import loader
H=json.loads((W/'HARVEST_v1.json').read_text())
# ratio m^2/E[y^2] for ReLU(N(t,1)) -> invert for t
tg=np.linspace(-6,6,4001);m=norm.pdf(tg)+tg*norm.cdf(tg);e2=(1+tg**2)*norm.cdf(tg)+tg*norm.pdf(tg);rg=m**2/e2
def tfit(mean,var):
    r=np.clip(mean**2/np.maximum(var+mean**2,1e-300),rg[0],rg[-1]);return np.interp(r,rg,tg)
rows=[]
for s in sorted(map(int,H)):
    t=loader.truth(s,check_weights=False);w=loader.weights(s)
    mp,vp=t['mc_post'],t['mc_post_var'];f=dict(seed=s)
    f['scale_var15']=float(vp[15].mean());f['scale_m2_15']=float((mp[15]**2).mean())
    for l in range(1,16):
        tt=tfit(mp[l],vp[l]);f[f'dead{l}']=float((norm.cdf(tt)<0.05).mean());f[f'near{l}']=float((np.abs(tt)<0.5).mean());f[f'tstd{l}']=float(tt.std())
        a=(w[l].astype(np.float64)**2)*vp[l-1][:,None];pr=a.sum(0)**2/(a**2).sum(0)
        f[f'pr_mean{l}']=float(pr.mean());f[f'pr_p10{l}']=float(np.percentile(pr,10))
    rec=max(H[str(s)],key=lambda x:x['mtime']);f['mse15']=rec['mse'][15];f['mse_layers']=rec['mse'];f['sha']=rec['sha']
    rows.append(f)
json.dump(rows,open(W/'FEATURES_v1.json','x'))
print(len(rows),'seeds')
