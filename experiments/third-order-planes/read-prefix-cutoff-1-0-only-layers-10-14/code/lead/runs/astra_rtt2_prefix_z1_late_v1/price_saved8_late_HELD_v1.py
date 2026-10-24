from pathlib import Path
import json,hashlib
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_rtt2_prefix_z1_late_v1';D.mkdir(exist_ok=True);Z=O/'lead/runs/astra_rtt2_prefix_z1_v1'
s=(Z/'price_prefix_z1_HELD_v1.py').read_text();setup=s[:s.index('with np.load(R/')];setup=setup.replace('D=Path(__file__).resolve().parent;O=D.parents[2]',f'D=Path({str(Z)!r});O=D.parents[2]')
exec(compile(setup,str(Z/'price_prefix_z1_HELD_v1.py'),'exec'))
c=json.loads((Z/'PROTOCOL_REGISTERED_astra_v1.json').read_text());out=[];cache={}
def priced(a):
 key=tuple(a[k]for k in ('m','k','c15','c2','old_rank'))
 if key not in cache:cache[key]=price(dict(m=a['m'],young_k=a['k'],old_rank=a['old_rank'],old_columns=a['old_rank'],c2=a['c2']),a['c15'])
 return cache[key]
for seed in range(7103248,7103256):
 qp=next(x['certificate']for x in c['stack_control_reuse']if x['seed']==seed);zp=next(Z.glob(f'run*/native_{seed}_ON.json'))
 qr=json.loads(Path(qp).read_text())['rows'][0];zr=json.loads(zp.read_text())['rows'][0]
 qa={r['layer']:r for r in qr['stats']['_RTT_STATS']['reads']};za={r['layer']:r for r in zr['stats']['_RTT_STATS']['reads']}
 rows=[]
 for l in range(10,15):
  qv,zv=priced(qa[l]),priced(za[l]);counter=dict(qa[l],c15=min(qa[l]['c15'],za[l]['c15']));cv=priced(counter)
  rows.append(dict(layer=l,qr_shape=qa[l],z1_shape=za[l],qr_price=qv,z1_price=zv,observed_shape_price_difference=qv['flops']-zv['flops'],qr_geometry_z1_prefix_price=cv,fixed_qr_geometry_saving=qv['flops']-cv['flops']))
 saving=sum(r['fixed_qr_geometry_saving']for r in rows);observed=sum(r['observed_shape_price_difference']for r in rows)
 out.append(dict(seed=seed,qr_whole_flops=qr['flops'],z1_whole_flops=zr['flops'],rows=rows,fixed_qr_geometry_saved_flops=saving,fixed_qr_geometry_saved_pct=100*saving/qr['flops'],observed_shape_difference_pct=100*observed/qr['flops']))
 print(seed,out[-1]['fixed_qr_geometry_saved_pct'],out[-1]['observed_shape_difference_pct'],flush=True)
result=dict(rows=out,mean_fixed_qr_geometry_saved_pct=sum(x['fixed_qr_geometry_saved_pct']for x in out)/8,mean_observed_shape_difference_pct=sum(x['observed_shape_difference_pct']for x in out)/8,limits=['Actual saved native shape metadata, public virtual warm primitive tariffs. No per-layer native bill counters exist; these are priced forecasts, not measured per-layer native savings.','All-layer z1 changed upstream physics, so its late prefix counts are a proxy; fixed-QR geometry comparison isolates shape credit but cannot prove future late-only shapes.','Same QR/basis/oldinner and Q.T youngleft dimensions cancel in fixed geometry; all prefix, power refinement, tail, oldtail and concatenation shape-dependent charges included.','No model or candidate created.'])
f=O/'lead/runs/astra_rtt2_prefix_z1_late_v1/SAVED8_LATE_COST_v1.json';f.write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['mean_fixed_qr_geometry_saved_pct'],result['mean_observed_shape_difference_pct'],hashlib.sha256(f.read_bytes()).hexdigest())
