"""Full384 (e33563bc) vs 2b108 paired24 on current24: plain + noise-free (cross-half) final MSE, bill, net. Pre-registered:
noise-free net mean over all 24 < 0 AND over last 8 (7103264-71) < 0."""
import glob,json,math,statistics as st,hashlib,subprocess
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parent;R=F.parent;O=R.parents[1]
bill=lambda f:max(f/2**41,.1)
def arm(s):
    if s==7103248:return next(Path(x) for x in glob.glob(str(R/'astra_full384_smirnov_bill1_v1/run_*/native_7103248_ON.json')))
    if 7103249<=s<=7103255:return R/f'opus_lead_screen_v1/full384_all/native_{s}.json'
    if s in (7103256,7103257):return R/f'opus_lead_screen_v1/full384_screen5b/native_{s}.json'
    return F/f'sci_f384/native_{s}.json'
def ctl(s):
    if s<=7103257:return next(Path(x) for x in glob.glob(str(R/f'astra_smirnov_final_native_ten_v1/run_final_parity_s*_v1/native_{s}_ON.json')))
    return F/f'sci_2b108/native_{s}.json'
live={int(x) for x in subprocess.check_output(['/bin/ps','-axo','pid='],text=True).split()}
for g in glob.glob(str(F/'sci_*/guard_*.json')):
    d=json.loads(Path(g).read_text());assert d['exit_code']==0 and d['whole_group_retired'] and d['pid'] not in live,g
rows=[]
for s in range(7103248,7103272):
    a,b=json.loads(arm(s).read_text()),json.loads(ctl(s).read_text())
    assert a['sha256'].startswith('e33563bc') and b['sha256'].startswith('2b10843a'),(s,a['sha256'][:8],b['sha256'][:8])
    ar,br=a['rows'][0],b['rows'][0]
    ya=np.load(str(arm(s)).replace('.json','_pred.npz'))[str(s)].astype(np.float64);yb=np.load(str(ctl(s)).replace('.json','_pred.npz'))[str(s)].astype(np.float64)
    t=np.load(O/f'analyst/fresh/truth_seed{s}.npz');A,B=t['mc_post_A'][-1],t['mc_post_B'][-1]
    ua,ub=np.mean((ya[-1]-A)*(ya[-1]-B)),np.mean((yb[-1]-A)*(yb[-1]-B))
    m,u,f=100*math.log(ar['mse_final']/br['mse_final']),100*math.log(ua/ub),100*math.log(bill(ar['flops'])/bill(br['flops']))
    rows.append(dict(seed=s,MSE=m,xMSE=u,bill=f,net=m+f,xnet=u+f,ops_delta=ar.get('ops',0)-br.get('ops',0),first_diff_layer=next((l for l in range(16) if ya[l].astype(np.float32).tobytes()!=yb[l].astype(np.float32).tobytes()),16)))
def S(k,rs):v=[r[k] for r in rs];return dict(mean=st.mean(v),se=st.stdev(v)/len(v)**.5,lower=sum(x<0 for x in v),n=len(v))
last8=[r for r in rows if r['seed']>=7103264]
out=dict(candidate='e33563bc',reference='2b10843a',all24={k:S(k,rows) for k in ('MSE','xMSE','bill','net','xnet','ops_delta')},last8={k:S(k,last8) for k in ('MSE','xMSE','bill','net','xnet')},rows=rows)
out['passed']=out['all24']['xnet']['mean']<0 and out['last8']['xnet']['mean']<0
p=F/'SCI24_DECISION_v1.json';open(p,'x').write(json.dumps(out,indent=2)+'\n')
for r in rows:print('%d MSE %+6.3f xMSE %+6.3f bill %+6.3f xnet %+6.3f ops %+5d L%d'%(r['seed'],r['MSE'],r['xMSE'],r['bill'],r['xnet'],r['ops_delta'],r['first_diff_layer']))
for grp in ('all24','last8'):print(grp,' '.join('%s %+.3f(se %.3f,%d/%d)'%(k,v['mean'],v['se'],v['lower'],v['n']) for k,v in out[grp].items()))
print('PASSED',out['passed'],'sha',hashlib.sha256(p.read_bytes()).hexdigest()[:8])
