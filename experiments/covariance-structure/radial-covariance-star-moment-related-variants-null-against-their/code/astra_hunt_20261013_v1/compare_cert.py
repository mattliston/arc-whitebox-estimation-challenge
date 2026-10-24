"""Compare native certificates and float32 prediction artifacts; no estimator execution."""
import json
from pathlib import Path
import sys
import numpy as np

base, trial, output = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parent
output.resolve().relative_to(root)
assert not output.exists()
a, b = json.loads(base.read_text()), json.loads(trial.read_text())
assert a['seeds'] == b['seeds']
pa = np.load(str(base).replace('.json', '_pred.npz'))
pb = np.load(str(trial).replace('.json', '_pred.npz'))
rows = []
for x, y in zip(a['rows'], b['rows']):
    assert x['seed'] == y['seed']
    d = pb[str(x['seed'])].astype(np.float64) - pa[str(x['seed'])].astype(np.float64)
    delta_mse = (d * d).mean(axis=1)
    base_final_mean_square = float(np.square(pa[str(x['seed'])][-1].astype(np.float64)).mean())
    bill_gain = 1 - y['flops'] / x['flops']
    d2_relative = float(delta_mse[-1] / x['mse_final'])
    rows.append(dict(seed=x['seed'], base_flops=x['flops'], trial_flops=y['flops'],
                     flops_saved=x['flops'] - y['flops'], bill_percent=-100*bill_gain,
                     base_ops=x['ops'], trial_ops=y['ops'], ops_delta=y['ops']-x['ops'],
                     residual_delta_s=5e-6*(y['ops']-x['ops']),
                     base_pred_sha=x['pred_sha'], trial_pred_sha=y['pred_sha'],
                     pred_sha_equal=x['pred_sha']==y['pred_sha'],
                     max_abs_delta=float(np.abs(d).max()), final_delta_mse=float(delta_mse[-1]),
                     base_final_prediction_rms=float(np.sqrt(base_final_mean_square)),
                     final_prediction_relative_rms=float(np.sqrt(delta_mse[-1] / base_final_mean_square)),
                     delta_mse_layers=delta_mse.tolist(), final_d2_relative=d2_relative,
                     final_mse_percent=100*(y['mse_final']/x['mse_final']-1),
                     gain_criterion_met=bill_gain>0 and (x['pred_sha']==y['pred_sha'] or d2_relative<bill_gain/2),
                     stats=y.get('stats')))
result = dict(base=str(base), trial=str(trial), base_sha256=a['sha256'], trial_sha256=b['sha256'], rows=rows)
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
