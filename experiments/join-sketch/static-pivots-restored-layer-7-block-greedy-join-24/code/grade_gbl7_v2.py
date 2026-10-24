"""Static-L7 GB128r vs GB128r controls on 24 seeds: 7103080-095 (dial_judge_v2; hunter res/) + 7103248-55 (screen judge; screen_FRAG).
Plain + noise-free (cross-half) final MSE, bill, net; rule: 24-seed xnet < 0 AND last-8 (7103248-55) xnet < 0."""
import json,math,statistics as st,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent;O=R.parents[1]
bill=lambda f:max(f/2**41,.1)
def pred(p,s):
    with np.load(str(p).replace('.json','_pred.npz')) as z:return z[str(s)].astype(np.float64)
rows=[]
for s in list(range(7103080,7103096))+list(range(7103248,7103256)):
    if s<7103248:a,b=D/f'res/sl7_{s}.json',R/f'opus_r256vgb_v1/res/gb128r_{s}.json'
    else:a,b=D/f'b/native_{s}.json',R/f'opus_joinx_v1/screen_FRAG/native_{s}.json'
    A,B=json.loads(a.read_text()),json.loads(b.read_text())
    assert A['sha256'].startswith('cb432f8c') and B['sha256'].startswith('4c484ee4'),(s,A['sha256'][:8],B['sha256'][:8])
    ar=A['rows'][0] if 'rows' in A else A;br=B['rows'][0] if 'rows' in B else B;ya,yb=pred(a,s),pred(b,s)
    t=np.load(O/f'analyst/fresh/truth_seed{s}.npz');TA,TB=t['mc_post_A'][-1],t['mc_post_B'][-1]
    ua,ub=np.mean((ya[-1]-TA)*(ya[-1]-TB)),np.mean((yb[-1]-TA)*(yb[-1]-TB))
    m,u,f=100*math.log(ar['mse_final']/br['mse_final']),100*math.log(ua/ub),100*math.log(bill(ar['flops'])/bill(br['flops']))
    rows.append(dict(seed=s,MSE=m,xMSE=u,bill=f,xnet=u+f,ops=ar.get('ops',0)-br.get('ops',0)))
def S(k,rs):v=[r[k] for r in rs];return dict(mean=st.mean(v),se=st.stdev(v)/len(v)**.5,lower=sum(x<0 for x in v),n=len(v))
last8=[r for r in rows if r['seed']>=7103248]
out=dict(candidate='cb432f8c (2b108+GB128r+static L7)',control='4c484ee4 (2b108+GB128r)',all24={k:S(k,rows) for k in ('MSE','xMSE','bill','xnet','ops')},last8={k:S(k,last8) for k in ('MSE','xMSE','bill','xnet')},rows=rows)
out['passed']=out['all24']['xnet']['mean']<0 and out['last8']['xnet']['mean']<0
p=D/'GBL7_SCI24_v2.json';p.open('x').write(json.dumps(out,indent=2)+'\n')
for r in rows:print('%d xMSE %+6.3f bill %+6.3f xnet %+6.3f ops %+5d'%(r['seed'],r['xMSE'],r['bill'],r['xnet'],r['ops']))
for g in ('all24','last8'):print(g,' '.join('%s %+.3f(se %.3f,%d/%d)'%(k,v['mean'],v['se'],v['lower'],v['n']) for k,v in out[g].items()))
print('PASSED',out['passed'],'sha',hashlib.sha256(p.read_bytes()).hexdigest()[:8])
