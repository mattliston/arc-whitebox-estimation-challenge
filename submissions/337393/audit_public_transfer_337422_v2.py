"""Saved-output decomposition only. No fitting, native calls, or pointer changes."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
A = HERE.parent
K = A.parent
R = K.parent
O = R.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def summarize(rows):
    n = len(rows)
    s0 = sum(r['u0'] * r['f0'] for r in rows)
    s1 = sum(r['u1'] * r['f1'] for r in rows)
    out = dict(n=n, lower=sum(r['u1'] * r['f1'] < r['u0'] * r['f0'] for r in rows))
    for name, values in (
        ('raw', [100 * (r['u1'] / r['u0'] - 1) for r in rows]),
        ('bill', [100 * (r['f1'] / r['f0'] - 1) for r in rows]),
        ('adjusted', [100 * (r['u1'] * r['f1'] / (r['u0'] * r['f0']) - 1) for r in rows]),
    ):
        out[name + '_paired_mean_pct'] = st.mean(values)
        out[name + '_paired_se_pct'] = st.stdev(values) / math.sqrt(n)
    out.update(pooled_adjusted_pct=100 * (s1 / s0 - 1),
               pooled_raw_mse_pct=100 * (sum(r['u1'] for r in rows) / sum(r['u0'] for r in rows) - 1),
               weighted_raw_contribution_pp=100 * sum((r['u1'] - r['u0']) * r['f0'] for r in rows) / s0,
               weighted_bill_contribution_pp=100 * sum(r['u0'] * (r['f1'] - r['f0']) for r in rows) / s0,
               weighted_cross_contribution_pp=100 * sum((r['u1'] - r['u0']) * (r['f1'] - r['f0']) for r in rows) / s0)
    return out


def mac_rows(directory):
    grade = {r['rank']: r for r in read(directory / 'candidate_all32/GRADE_STORE_v1.json')}
    bills = {}
    for path in directory.glob('candidate_part*/rows.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['rank'] not in bills
            assert row['fast_rescue_stats']['calls'] == 0 and row['nonfinite'] == 0
            bills[row['rank']] = row['flops']
    assert set(grade) == set(bills) and len(grade) == 32
    return {rank: (row['u1'], bills[rank]) for rank, row in grade.items()}


p0 = HERE / 'public_monitor_20261021T002127Z/eval_337421_raw.json'
p1 = HERE / 'public_monitor_20261021T002129Z/eval_337422_raw.json'
old, new = read(p0), read(p1)
pub, tele = {}, {}
for sid, data in ((337421, old), (337422, new)):
    assert data['submission_status'] == 'GRADED'
    ev = data['evaluation']
    assert ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
    result = ev['results']
    assert len(result['setup_executions']) == 5
    assert all(x['error_code'] == 'SUCCESS' for x in result['setup_executions'])
    tele[sid] = {x['mlp_index']: x for x in result['per_mlp']}
    pub[sid] = {x['mlp_index']: x['scores'] for x in result['public_scores']}
    assert len(tele[sid]) == 100 and len(pub[sid]) == 50
    assert all(x['state'] == 'completed' and x['error_code'] in (None, 'SUCCESS') for x in tele[sid].values())
    assert all(x['telemetry']['effective_compute'] == x['telemetry']['flops_used'] for x in tele[sid].values())
    ids = sorted(pub[sid])
    assert st.mean(tele[sid][i]['telemetry']['effective_compute'] for i in ids) == result['aggregates']['public']['mean_effective_compute']
    assert min(x['telemetry']['effective_compute'] for x in tele[sid].values()) / 2**41 > .1
for i in range(100):
    assert tele[337421][i]['mlp_name'] == tele[337422][i]['mlp_name']
    assert tele[337421][i]['dataset_row'] == tele[337422][i]['dataset_row']
public = [dict(rank=i, u0=pub[337421][i]['final_layer_mse'], u1=pub[337422][i]['final_layer_mse'],
               f0=tele[337421][i]['telemetry']['effective_compute'],
               f1=tele[337422][i]['telemetry']['effective_compute']) for i in sorted(pub[337421])]
public_summary = summarize(public)
public_summary['actual_platform_score_pct'] = 100 * (new['submission_score'] / old['submission_score'] - 1)
public_summary['mse_precision'] = 'Public MSE is rounded to four significant digits. Decomposition is approximate; exact top-level score is authoritative.'
def half_unit(value):
    return .5 * 10**(math.floor(math.log10(abs(value))) - 3)
lo = sum((r['u1'] - half_unit(r['u1'])) * r['f1'] for r in public) / sum((r['u0'] + half_unit(r['u0'])) * r['f0'] for r in public)
hi = sum((r['u1'] + half_unit(r['u1'])) * r['f1'] for r in public) / sum((r['u0'] - half_unit(r['u0'])) * r['f0'] for r in public)
public_summary['rounded_mse_ratio_bounds_pct'] = [100 * (lo - 1), 100 * (hi - 1)]
assert public_summary['rounded_mse_ratio_bounds_pct'][0] <= public_summary['actual_platform_score_pct'] <= public_summary['rounded_mse_ratio_bounds_pct'][1]

m0 = mac_rows(R / 'astra_oct20_bill_v1/science32_eh14_flatload_revert_parallel5_v1')
m1 = mac_rows(R / 'astra_oct20_vm_frontier_v1/science32_msg8_loaderfix_parallel5_v1')
mac = [dict(rank=i, u0=m0[i][0], f0=m0[i][1], u1=m1[i][0], f1=m1[i][1]) for i in sorted(m0)]
fresh_source = A / 'cloud_fresh4_v1/MSG8_FRESH_VS_VALID337421_v1.json'
fresh = [dict(rank=r['rank'], u0=r['reference_u'], u1=r['candidate_u'], f0=r['reference_flops'], f1=r['candidate_flops']) for r in read(fresh_source)['rows']]
pointer_path = K / 'BEST_GRADED_REFERENCE_v1.json'
pointer = read(pointer_path)
assert pointer['submission_id'] == 337422 and pointer['platform_score'] == new['submission_score']
assert pointer['platform_receipt_sha256'] == sha(p1)
assert sha(pointer['science_source']) == pointer['science_sha256'] == '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
archive = O / 'release/runs/astra_msg8_loaderfix_e41095e4_bank_v1/submission.tar.gz'
assert sha(archive) == '8b1d3c1e68810fac6c41723bbc9b9f9205b34a115a29939ea49f31a290a0bd82'
with tarfile.open(archive) as tar:
    assert hashlib.sha256(tar.extractfile('estimator.py').read()).hexdigest() == pointer['release_sha256'] == 'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
receipt = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
               status='SAVED_OUTPUT_PUBLIC_TRANSFER_AUDIT', reference=337421, candidate=337422,
               public50_rounded_mse_exact_compute=public_summary, original_mac32=summarize(mac),
               linux_fresh4=summarize(fresh), public_rows=public, mac_rows=mac, fresh_rows=fresh,
               all100_zero_failures=True, five_setup_successes=True, all100_effective_equals_predict_flops=True,
               no_multiplier_floor_active=True, public_raw_available_count=50, private_raw_available_count=0,
               pointer_and_archive_pins_pass=True, no_pointer_mutation=True,
               public_sources={str(p0): sha(p0), str(p1): sha(p1)}, fresh_receipt_sha256=sha(fresh_source),
               pointer_sha256=sha(pointer_path), archive_sha256=sha(archive), script_sha256=sha(__file__),
               conclusion='Observed transfer loss is predominantly the accuracy component: Mac32 improves raw while public50 rounded raw worsens. Billing savings remain similar. Mean-ratio versus weighted aggregation is quantified but does not explain the discrepancy. No evidence of hidden setup or residual billing. Different cohorts/truth estimators are possible causes, not established mechanisms.',
               no_fitting=True, no_new_model_evaluation=True)
out = HERE / 'PUBLIC_337422_TRANSFER_COMPOSITION_v1.json'
with out.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: receipt[k] for k in ('public50_rounded_mse_exact_compute', 'original_mac32', 'linux_fresh4')}))
print(json.dumps(dict(receipt=str(out), sha256=sha(out))))
