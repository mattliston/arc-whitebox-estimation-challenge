import json,statistics
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
N=D/'native_fixed2_v1'
P=D.parent/'fusion_strip5_v1/native_fixed2_v1'
new=json.loads((N/'RESULT_v1.json').read_text())
old=json.loads((P/'RESULT_v1.json').read_text())
byrank={r['rank']:r for r in old['rows']}
out=[]
for a in new['rows']:
    rank=a['rank'];b=byrank[rank]
    ratio=(1+a['raw_pct']/100)/(1+b['raw_pct']/100)
    bill=a['flops']/b['flops']
    p=np.fromfile(N/f'rank{rank}/pred_{rank}.bin',dtype='<f4').reshape(16,1024).astype('d')
    c=np.fromfile(P/f'rank{rank}/pred_{rank}.bin',dtype='<f4').reshape(16,1024).astype('d')
    out.append(dict(rank=rank,raw_delta_pct_vs_unsym=100*(ratio-1),bill_delta_pct_vs_unsym=100*(bill-1),
        adjusted_delta_pct_vs_unsym=100*(ratio*bill-1),extra_flops=a['flops']-b['flops'],extra_requests=a['requests']-b['requests'],
        residual_instructions_delta=a['residual_instructions']-b['residual_instructions'],
        prediction_rms=float(np.sqrt(np.mean((p-c)**2))),per_layer_prediction_rms=np.sqrt(np.mean((p-c)**2,axis=1)).tolist(),
        final_raw_pct_vs95c=a['raw_pct'],final_adjusted_pct_vs95c=a['adjusted_pct_vs_science']))
result=dict(status='FIXED_PAIRED_SYMMETRIZATION_DIAGNOSTIC',rows=out,
    mean={k:statistics.mean(v[k] for v in out) for k in ('raw_delta_pct_vs_unsym','bill_delta_pct_vs_unsym','adjusted_delta_pct_vs_unsym','extra_flops','extra_requests')},
    interpretation='Sym and unsym groups both used fresh individual rank35/rank200. Pairwise comparison therefore has matched cache-group structure. These two data points do not validate hosted runtime or population accuracy.')
with(N/'SYMMETRIZATION_COMPARISON_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
