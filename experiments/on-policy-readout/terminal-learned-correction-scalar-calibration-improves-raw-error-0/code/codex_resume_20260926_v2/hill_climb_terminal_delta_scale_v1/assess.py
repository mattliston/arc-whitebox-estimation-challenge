"""Frozen one-scalar terminal correction fit/check; root guard required."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent;H=D.parent;P2=H.parent
stage=sys.argv[1];assert stage in ('fit2','check6')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def verify(x):
 if isinstance(x,dict):
  if all(k in x for k in ('path','sha256','bytes')):assert pin(x['path'])=={k:x[k] for k in ('path','sha256','bytes')}
  for v in x.values():verify(v)
 elif isinstance(x,list):
  for v in x:verify(v)
protocol=read(D/'protocol.json');verify(protocol)
assert protocol['source']['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
if stage=='check6':
 fit=read(D/'fit2_assessment.json');verify(fit['inputs']);assert fit['admitted']
 scale=float(fit['scale']);ranks=protocol['check_ranks'];size='6'
else:
 fit=None;scale=None;ranks=protocol['fit_ranks'];size='2'
folder=H/(D.name+'_'+stage+'_v1');block=protocol['base_blocks'][size];base=H/block['tag']
gpath=D/(stage+'_capture_guard.json');g=read(gpath)
assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['max_swap_growth_bytes']==0 and not g['remaining_rows']
assert g['max_rss_bytes']<7500000000
manifest=read(folder/'manifest.json');bm=read(base/'manifest.json')
assert manifest['sha256']==bm['sha256']==protocol['source']['sha256']
assert manifest['ranks']==bm['ranks']==ranks
rows=[json.loads(x) for x in (folder/'rows.jsonl').read_text().splitlines()]
brows=[json.loads(x) for x in (base/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in rows]==[x['rank'] for x in brows]==ranks
items=[];num=0.;den=0.;active_n=0
inputs=[pin(D/'protocol.json'),pin(D/'capture.py'),pin(D/'assess.py'),pin(gpath),pin(folder/'manifest.json'),pin(folder/'rows.jsonl'),pin(base/'manifest.json'),pin(base/'rows.jsonl')]
for rank,row,brow in zip(ranks,rows,brows):
 yp=folder/f'pred_{rank}.npy';bp=base/f'pred_{rank}.npy';cp=folder/f'terminal_{rank}.npz'
 assert yp.read_bytes()==bp.read_bytes(), 'all16 baseline byte parity'
 assert row['flops']==brow['flops'] and row['mse']==brow['mse'], 'baseline bill/score parity'
 y=np.load(yp);cap=np.load(cp);m=cap['physical'];delta=cap['delta']
 assert y.shape==(16,1024) and m.shape==delta.shape==(1024,)
 assert np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(delta).all()
 raw=m+delta;replay=np.maximum(raw,0.).astype(np.float32).astype(np.float64)
 assert replay.tobytes()==y[15].tobytes(), 'scale1 replay parity'
 tp=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
 truth=np.load(tp)['mc_post'][15].astype(np.float64)
 if stage=='fit2':
  active=raw>0;num+=float(np.sum((raw[active]-truth[active])*delta[active]));den+=float(np.sum(delta[active]**2));active_n+=int(active.sum())
 items.append((rank,m,delta,truth,y[15],row))
 inputs.extend([pin(yp),pin(bp),pin(cp),pin(tp)])
if stage=='fit2':
 assert den>0 and np.isfinite(num) and np.isfinite(den)
 scale=1-num/den
assert np.isfinite(scale)
per=[];base_errors=[];candidate_errors=[]
for rank,m,delta,truth,base_y,row in items:
 cand=np.maximum(m+scale*delta,0.).astype(np.float32).astype(np.float64)
 assert np.isfinite(cand).all()
 b=float(np.mean((base_y-truth)**2));c=float(np.mean((cand-truth)**2))
 assert abs(b/row['mse']-1)<1e-12
 base_errors.append(b);candidate_errors.append(c)
 per.append(dict(rank=rank,base_raw_mse=b,candidate_raw_mse=c,raw_pct=100*(c/b-1),changed_units=int(np.count_nonzero(cand!=base_y)),delta_rms=float(np.sqrt(np.mean(delta**2)))))
b=np.array(base_errors);c=np.array(candidate_errors);raw_pct=float(100*(c.mean()/b.mean()-1));wins=int((c<b).sum());maxreg=max(x['raw_pct'] for x in per)
gates=protocol['fit_gates'] if stage=='fit2' else protocol['check_gates']
checks=dict(raw_gain=raw_pct<=-gates['min_raw_gain_pct'],wins=wins>=gates['min_wins'],max_regression=maxreg<=gates['max_regression_pct'],positive_finite_scale=scale>0)
ci=None
if stage=='check6':
 rng=np.random.default_rng(gates['bootstrap_seed']);ix=rng.integers(0,len(ranks),size=(gates['bootstrap_resamples'],len(ranks)))
 ci=(100*(np.quantile(c[ix].mean(1)/b[ix].mean(1),[.025,.975])-1)).tolist()
 checks['check_only_ci_upper_negative']=ci[1]<0
 inputs.append(pin(D/'fit2_assessment.json'))
result=dict(schema='terminal-delta-one-scalar-'+stage+'-v1',stage=stage,admitted=all(checks.values()),scale=scale,raw_pct=raw_pct,raw_wins=wins,max_raw_regression_pct=maxreg,raw_ci95_pct=ci,checks=checks,per_network=per,fit_sums=dict(numerator=num,denominator=den,active_units=active_n) if stage=='fit2' else None,inputs=inputs,all16_baseline_file_bytes_exact=True,all_baseline_bills_exact=True,scale1_terminal_replay_bytes_exact=True,offline_only=True,no_native_candidate=True,no_serving_claim=True,source=protocol['source'],fit_networks=protocol['fit_ranks'],historically_reused_panels=True)
with (D/(stage+'_assessment.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:result[k] for k in ('stage','admitted','scale','raw_pct','raw_wins','max_raw_regression_pct','raw_ci95_pct','checks')}))
