import hashlib,json,statistics
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent
R=D.parents[2]
O=R.parents[1]
N=D/'native_fixed2_v1'
Q=R/'astra_oct18_lead_v1/compact10_repair_qualification_v1'
C=Q/'science32/candidate32'
T=Q/'candidate_normal'
F=D.parent/'fusion_native2_v1'
assert json.loads((N/'COMPLETE_v1.json').read_text())['all_groups_retired']
rows=lambda p:{v['rank']:v for v in map(json.loads,(p/'rows.jsonl').read_text().splitlines()) if 'rank' in v}
control,runtime,fusion=rows(C),rows(T),rows(F)
out=[]
for rank in (35,200):
    P=N/('rank'+str(rank))
    g=json.loads((P/'GUARD_v1.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
    a=rows(P)[rank]; b=control[rank]; rt=runtime[rank]
    p=np.fromfile(P/f'pred_{rank}.bin',dtype='<f4').reshape(16,1024).astype('d')
    c=np.fromfile(C/f'pred_{rank}.bin',dtype='<f4').reshape(16,1024).astype('d')
    A=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{rank:04d}_A32M.npz')['mc_post'].astype('d')
    B=np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{rank:04d}_B32M.npz')['mc_post'].astype('d')
    u=np.mean((p-A)*(p-B),axis=1); u0=np.mean((c-A)*(c-B),axis=1)
    raw=float(u[-1]/u0[-1]); bill=a['flops']/b['flops']
    out.append(dict(rank=rank,raw_pct=100*(raw-1),bill_pct_vs_science=100*(bill-1),adjusted_pct_vs_science=100*(raw*bill-1),
        bill_pct_vs_runtime=100*(a['flops']/rt['flops']-1),flops=a['flops'],control_flops=b['flops'],runtime_control_flops=rt['flops'],
        requests=a['requests'],requests_delta_vs_science=a['requests']-b['requests'],requests_delta_vs_runtime=a['requests']-rt['requests'],
        requests_delta_vs_fusion=a['requests']-fusion[rank]['requests'],wall_s=a['wall_s'],control_runtime_wall_s=rt['wall_s'],
        residual_instructions=a.get('residual_instructions'),nonfinite=a['nonfinite'],rescue=a.get('fast_rescue_stats'),
        fusion_state=a.get('oct18_fusion_state'),covdiag_state=a.get('oct18_covdiag_state'),
        prediction_rms=float(np.sqrt(np.mean((p-c)**2))),max_abs_dy=float(np.max(np.abs(p-c))),
        guard={k:g.get(k) for k in ('completed','whole_group_retired','max_rss_bytes','max_footprint_bytes','min_pressure_free_pct','elapsed_s')}))
means={k:statistics.mean(v[k] for v in out) for k in ('raw_pct','bill_pct_vs_science','adjusted_pct_vs_science','requests_delta_vs_runtime')}
result=dict(status='FIXED_TWO_NETWORK_DIRECT_COVARIANCE_DIAGONAL_SCREEN',candidate_sha256=hashlib.sha256((D/'candidate_95c_direct_covdiag_HELD_v1.py').read_bytes()).hexdigest(),
    baseline_submission=336935,baseline_adjusted=1.4468565569762618e-9,rows=out,mean=means,
    conditional_cold_biased_adjusted=1.4468565569762618e-9*(1+means['adjusted_pct_vs_science']/100),
    caveats=['Both ranks ran in separate fresh processes. Rank35 matches cold control; rank200 control is warm, so its charged bill and request comparison include extra cold aliases.',
        'The historical estimated cold200 bill bias is approximately +0.14 percentage points; it is not subtracted from measured results.',
        'Only hidden covariance diagonals are replaced by direct float32-factor products accumulated in float64; no byte-equivalence claim.',
        'This fixed2 screen does not establish population accuracy or final release eligibility.'])
with (N/'RESULT_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],mean=means,conditional_cold_biased_adjusted=result['conditional_cold_biased_adjusted'])))
for v in out: print(json.dumps({k:v[k] for k in ('rank','raw_pct','bill_pct_vs_science','adjusted_pct_vs_science','requests','requests_delta_vs_runtime','requests_delta_vs_fusion','wall_s','covdiag_state','guard')}))
