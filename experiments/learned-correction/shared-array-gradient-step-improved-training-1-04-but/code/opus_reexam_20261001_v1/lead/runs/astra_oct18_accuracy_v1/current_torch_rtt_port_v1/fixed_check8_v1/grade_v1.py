"""Read-only fixed CHECK8 grade. Conditional projection only; no parameter changes."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
P=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1');O=P.parents[3]
BASE=1.4468565569762618e-9
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 pairs=json.loads(a.pairs.read_text());assert pairs['status']=='ALL_EIGHT_PAIRS_COMPLETE_AND_RETIRED';assert [r['seed'] for r in pairs['rows']]==list(range(7103472,7103480))
 rows=[]
 for pair in pairs['rows']:
  seed=pair['seed'];truth=O/f'analyst/fresh/truth_seed{seed}.npz'
  with np.load(truth,allow_pickle=False) as z:
   N=int(z['N']);assert N>=7995392
   target=np.asarray(z['mc_post'],dtype=np.float64)[15];noise=float(np.asarray(z['mc_post_var'],dtype=np.float64)[15].mean()/N)
  row=dict(seed=seed,N=N,truth_sha256=sha(truth),noise=noise)
  for arm in ('control','point'):
   x=pair[arm];assert sha(x['result'])==x['result_sha256']
   result=json.loads(Path(x['result']).read_text());pred=np.asarray(result['final_prediction'],dtype=np.float64);assert pred.shape==target.shape==(1024,)
   mse=float(np.square(pred-target).mean());assert math.isclose(mse,result['mse'],rel_tol=1e-8,abs_tol=1e-17)
   assert mse>noise
   row[arm]=dict(noisy_mse=mse,noise_corrected_mse=mse-noise)
  row['ratio']=row['point']['noise_corrected_mse']/row['control']['noise_corrected_mse'];rows.append(row)
 ratios=np.asarray([x['ratio'] for x in rows]);logs=np.log(ratios);geo=float(np.exp(logs.mean()));pooled=sum(x['point']['noise_corrected_mse'] for x in rows)/sum(x['control']['noise_corrected_mse'] for x in rows)
 receipt=dict(status='CURRENT_PORT_FIXED_CHECK8_SCREEN_ONLY',rows=rows,geomean_adjusted_percent_if_bill_unchanged=100*(geo-1),pooled_adjusted_percent_if_bill_unchanged=100*(pooled-1),log_standard_error_percent=float(logs.std(ddof=1)/math.sqrt(8)*100),wins=int((ratios<1).sum()),last_graded_adjusted=BASE,conditional_geomean_projection=BASE*geo,conditional_pooled_projection=BASE*pooled,native_bill_measured=False,native_confirmation_required=True,pairs_sha256=sha(a.pairs),parameters_updated=False)
 with a.out.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:receipt[k] for k in ('status','geomean_adjusted_percent_if_bill_unchanged','pooled_adjusted_percent_if_bill_unchanged','log_standard_error_percent','wins','conditional_geomean_projection')}))
if __name__=='__main__':main()
