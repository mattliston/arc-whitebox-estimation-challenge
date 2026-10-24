"""Retain the passed TRAIN8 component and check fixed S32 runtime; scalar stdlib only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

D = Path(__file__).resolve().parent
H = D.parent
def pin(p):
    p = Path(p).resolve(); raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
def read(p): return json.loads(Path(p).read_text())
assessor = D/'boundary_assess_s34.py'
assert pin(assessor)['sha256'] == '7f5bdfab67f9e32282f42982999437f5c7ace6851b12e7b90c2dc79dac64fc67'
spec = importlib.util.spec_from_file_location('packed_pair_saved_boundary_helpers', assessor)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
_, q = module.helpers()
bp = pin(D/'boundary_assessment.json'); boundary = json.loads(q.checked(bp))
np = pin(D/'native8_assessment.json'); training = json.loads(q.checked(np))
assert boundary['passed'] and boundary['receipt_checks_passed'] and all(boundary['gates'].values())
assert training['admitted'] and training['passed_integrity'] and all(training['gates'].values())
assert training['n'] == 8 and training['boundary'] == bp and training['candidate'] == boundary['candidate']
assert training['config'] == boundary['native_config'] == pin(D/'native_screen_continuation.json')
nc = json.loads(q.checked(training['config'])); bc = json.loads(q.checked(boundary['config']))
assert nc['arms']['candidate']['source'] == training['candidate'] and nc['arms']['base']['source'] == training['comparator']
assert nc['arms']['candidate']['source'] == pin(D/'candidate.py')
q.guard(pin(D/'native8_assess_guard.json'), [str(q.NATIVE), '-B', nc['engine']['path'],
    training['config']['path'], training['config']['sha256'], '8'], 120)
q.guard(pin(D/'boundary_assess_guard.json'), nc['boundary_before8']['assessment_command'], 120)
assert boundary['native8_runtime_recheck_contract'] == 'direct_candidate_over_pinned_S32_v2'
reference_rows, reference = module.runtime_controls(q, bc)
assert reference == boundary['runtime_reference']
candidate_rows = {}; inputs = {}
for block in training['blocks']:
    size = block['size']; assert size in (2, 6) and size not in candidate_rows
    candidate_rows[size] = module.native_arm(q, nc['arms']['candidate'], size, block['arms']['candidate'])
    inputs[str(size)] = dict(candidate=block['arms']['candidate']['rows'], reference=reference['native_controls'][str(size)]['rows'])
assert set(candidate_rows) == {2, 6}
candidate_cpu = math.fsum(row['cpu_s'] for size in (2, 6) for row in candidate_rows[size])
reference_cpu = math.fsum(row['cpu_s'] for size in (2, 6) for row in reference_rows[size])
ratio = candidate_cpu/reference_cpu
rt = boundary['runtime']
ratio2 = math.fsum(row['cpu_s'] for row in candidate_rows[2])/math.fsum(row['cpu_s'] for row in reference_rows[2])
assert ratio2 == rt['native2_cpu_ratio']
assert rt['gate_s'] == 105 and rt['native2_boundary_disagreement_pp'] <= 5
growth = max(0., ratio-1, rt['native2_cpu_ratio']-1, rt['boundary_mean_cpu_ratio']-1)
requests = rt['nonnegative_mean_request_delta']
official_pin = q.fixed(H/'eval_332738.json', '6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6')
official = json.loads(q.checked(official_pin)); ev = official['evaluation']
assert official['submission_id'] == 332738 and ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
telemetry = [row['telemetry'] for row in ev['results']['per_mlp']]; assert len(telemetry) == 100
assert all(math.isfinite(t[k]) and t[k] >= 0 for t in telemetry for k in ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s'))
grid = [dict(native_fraction=f, request_cost_s=p, projected_worst_wall_s=max(
    t['participant_predict_wall_s']+f*growth*t['flopscope_server_kernel_wall_s']+requests*p for t in telemetry))
    for f in (.60, .78, .95) for p in (.00030, .00041, .00050)]
wall = max(row['projected_worst_wall_s'] for row in grid)
runtime = dict(contract='direct_candidate_over_pinned_S32_v2', passed=wall <= 105,
    reference=reference, official=official_pin, native8_ratio_inputs=inputs,
    cpu_ratio_rule='sum(candidate TRAIN2+TRAIN6 cpu_s) / sum(pinned S32 TRAIN2+TRAIN6 cpu_s)',
    candidate_cpu_s=candidate_cpu, reference_cpu_s=reference_cpu, native8_cpu_ratio=ratio,
    native2_cpu_ratio=ratio2, boundary_mean_cpu_ratio=rt['boundary_mean_cpu_ratio'],
    used_nonnegative_cpu_change=growth, nonnegative_mean_request_delta=requests,
    projection_grid=grid, projected_worst_wall_s=wall, gate_s=105, timing_rerun=False)
assert runtime['passed'], runtime
byte_checks = []
for size in (2, 6):
    child = nc['arms']['candidate']['blocks'][str(size)]
    base = nc['arms']['base']['blocks'][str(size)]
    for rank in child['ranks']:
        cp = pin(H/child['tag']/('pred_%s.npy' % rank))
        pp = pin(H/base['tag']/('pred_%s.npy' % rank))
        byte_checks.append(dict(rank=rank, candidate=cp, base=pp,
                               whole_npy_bytes_exact=q.checked(cp) == q.checked(pp)))
assert len(byte_checks) == 8
result = dict(status='retained_for_combination', candidate=training['candidate'], comparator=training['comparator'],
    native8=np, native8_guard=pin(D/'native8_assess_guard.json'), native_config=training['config'],
    boundary=bp, boundary_guard=pin(D/'boundary_assess_guard.json'), writer=pin(__file__),
    adjusted_pct=training['adjusted_pct'], adjusted_ci95_pct=training['adjusted_ci95_pct'],
    adjusted_wins=training['adjusted_wins'], raw_pct=training['raw_pct'], bill_pct=training['bill_pct'],
    additional6_adjusted_pct=training['additional6']['adjusted_pct'], score_gates=training['gates'],
    native8_runtime_recheck=runtime, prediction_file_byte_checks=byte_checks, all16_all8_file_bytes_exact=all(x['whole_npy_bytes_exact'] for x in byte_checks), projected_residual_s=boundary['residual']['projected_s'],
    setup_s=boundary['setup']['setup_wall_s'], setup_counts=nc['arms']['candidate']['setup']['counts'],
    source_pins_checked=len(q.EVIDENCE), scope='Fixed TRAIN8 and own boundary only; retain as a complementary component.',
    broad_qualified=False, serving_qualified=False, release_qualified=False, submitted=False,
    bank_min_gain_pct=1.0, standalone_bank_minimum_met=False, arrays_read=False, model_executed=False)
for record in list(q.EVIDENCE.values()): q.checked(record)
with (D/'retained_component.json').open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps(dict(receipt=pin(D/'retained_component.json'), native8_cpu_ratio=ratio,
    projected_worst_wall_s=wall, runtime_passed=runtime['passed'])))
