from pathlib import Path
import json,hashlib
import numpy as np
from inspect_fixed_band_oracle_HELD_v1 import basis,zero_diag
D=Path(__file__).resolve().parent;O=D.parents[2];C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1';old=json.loads((D/'TRAIN16_OLD_PROTECTED_QUARTER_v1.json').read_text());tail=json.loads((D/'TRAIN16_FIXED_BAND_ORACLE_v1.json').read_text());rows=[]
for ref in old['rows']:
 l=ref['layer'];path=C/f'L{l:02d}.npz'
 with np.load(path,allow_pickle=False)as z:
  H=z['old_inner'].astype(float);B=z['old_right'].astype(float);ids=z['rows'].astype(int);cols=z['cols'].astype(int)[:B.shape[0]];v=z['vc'].astype(float);mu=z['mu'].astype(float);g=z['gate'].astype(float)
 center=ref['exact_center'];outer=ids[center:];Q=basis(v[outer],mu[outer]);R=np.zeros_like(H);R[center:]=H[center:]-Q@(Q.T@H[center:]);E=R@B.T;zero_diag(E,ids,cols)
 d=np.exp(-.5*mu[ids]**2/v[ids])/np.sqrt(2*np.pi*v[ids]);w=d[:,None]*g[cols][None,:]
 result=dict(layer=l,center=center,outer_rows=len(outer),basis_rank=Q.shape[1],inner_error=float(np.sum(R*R)),old_prefix_error=float(np.sum(E*E)),weighted_error=float(np.sum((E*w)**2)))
 result['ratio_to_global_fit']={k:result[k]/ref[k]for k in ('inner_error','old_prefix_error','weighted_error')};assert result['inner_error']<=ref['inner_error']*(1+1e-12)
 rowtail=next(x for x in tail['per_layer']if x['layer']==l)['cases']['incumbent_QR']['weighted_directed'];result['incumbent_tail_weighted_error']=rowtail;result['weighted_error_vs_incumbent_tail']=0.25*result['weighted_error']/rowtail;rows.append(result)
pooled={k:sum(r[k]for r in rows)/sum(r[k]for r in old['rows'])for k in ('inner_error','old_prefix_error','weighted_error')};weighted_ratio=0.25*sum(r['weighted_error']for r in rows)/sum(r['incumbent_tail_weighted_error']for r in rows)
price=json.loads((D/'PROTECTED_OUTER_QUARTER_TARIFF_v1.json').read_text());projection_credit=sum(4*r['exact_center']*p['old_columns'] for r,p in zip(old['rows'],price['rows']));whole_bill=price['saving']/(price['saving_pct']/100)
report=dict(status='FIXED_OUTER_ONLY_LS_OFFLINE',rows=rows,pooled_error_ratio_to_closed_global_fit=pooled,pooled_weighted_error_vs_incumbent_tail=weighted_ratio,extra_projection_saving_FLOPs=projection_credit,extra_projection_saving_pct=100*projection_credit/whole_bill,warm_total_saving_pct=price['saving_pct']+100*projection_credit/whole_bill,limits=['TRAIN16 only; no truth, models, or amplitude/fraction/basis sweep.','Same exact center and two packed rows; Q restricted to approximated outer support.','No worse unweighted linear residual in exact arithmetic; covariance weighting, diagonal exclusion and future nonlinear feedback have no monotonic guarantee.','Old right captured over prefix only. Current negative tail and future transport omitted.','Warm forecast only; actual whole bill required.','Absolute tail comparison includes the common 1/2 K21-to-covariance coefficient; v1 omitted this factor in numerator only.'])
p=D/'TRAIN16_PROTECTED_OUTER_FIT_v2.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('rows','limits')}));print('sha256',hashlib.sha256(p.read_bytes()).hexdigest())
