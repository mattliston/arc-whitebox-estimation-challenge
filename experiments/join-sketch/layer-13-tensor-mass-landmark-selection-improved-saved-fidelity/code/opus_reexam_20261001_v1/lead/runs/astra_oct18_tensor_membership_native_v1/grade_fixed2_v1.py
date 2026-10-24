"""Grade the fixed two cold diagnostics, never authorize a submission."""
import json, statistics
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
N = D / 'native_fixed2_v1'
BASE = 1.4468565569762618e-9
controls = {
    35: O / 'lead/runs/astra_oct18_lead_v1/compact10_repair_qualification_v1/science32/candidate32',
    200: O / 'lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/native_control_split_fixed_v1/control200',
}
rows = []
for rank in (35, 200):
    c = N / f'rank{rank}'
    g = json.loads((c/'GUARD_v1.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and not g.get('remaining_rows')
    a = next(json.loads(s) for s in (c/'rows.jsonl').read_text().splitlines() if json.loads(s).get('rank') == rank)
    b = next(json.loads(s) for s in (controls[rank]/'rows.jsonl').read_text().splitlines() if json.loads(s).get('rank') == rank)
    assert a['nonfinite'] == 0 and a['fast_rescue_stats']['calls'] == 0
    assert a['oct18_tensor_membership_state']['calls'] == 1
    p = np.fromfile(c/f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d')
    q = np.fromfile(controls[rank]/f'pred_{rank}.bin', dtype='<f4').reshape(16,1024).astype('d')
    assert np.array_equal(p[:13], q[:13]), 'L13-only source changed earlier predictions'
    A = np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_A32M.npz')['mc_post'].astype('d')
    B = np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_B32M.npz')['mc_post'].astype('d')
    u = np.mean((p-A)*(p-B), axis=1)
    v = np.mean((q-A)*(q-B), axis=1)
    fr = a['flops']/b['flops']
    raw = 100*(u[-1]/v[-1]-1)
    adj = 100*(u[-1]/v[-1]*fr-1)
    rows.append(dict(rank=rank, raw_pct=float(raw), bill_pct=100*(fr-1), adjusted_pct=float(adj),
        conditional_adjusted_score=BASE*(1+adj/100), requests_delta=a['requests']-b['requests'],
        earlier_layers_byte_identical=True, layer_raw_pct=(100*(u/v-1)).tolist(),
        stats=a['oct18_tensor_membership_state'], candidate_wall_s=a['wall_s'],
        control_wall_s=b['wall_s'], residual_instructions_delta_pct=100*(a['residual_instructions']/b['residual_instructions']-1),
        candidate_flops=a['flops'],control_flops=b['flops'], guard=g))
mean = statistics.mean(r['adjusted_pct'] for r in rows)
result = dict(status='FIXED_TWO_DIAGNOSTIC_NOT_BANKED',baseline_submission=336935,
    baseline_adjusted_score=BASE,rows=rows,mean_adjusted_pct=mean,
    conditional_adjusted_score=BASE*(1+mean/100))
with (N/'RESULT_v1.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
for r in rows: print(json.dumps({k:v for k,v in r.items() if k not in ('guard','stats')}))
