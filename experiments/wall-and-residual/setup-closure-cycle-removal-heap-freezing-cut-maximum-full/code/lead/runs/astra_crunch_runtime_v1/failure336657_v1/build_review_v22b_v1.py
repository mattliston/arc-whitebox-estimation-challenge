from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path('/Users/matt/arc-white-box-estimation')
O = ROOT/'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
R = O/'lead/runs'
D = R/'astra_crunch_runtime_v1/failure336657_v1'
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def session(path):
    manifest = load(path/'manifest.json')
    manifest['rows'] = [json.loads(x) for x in (path/'rows.jsonl').read_text().splitlines()]
    return manifest

baseline_path = D/'natural_sequence_v1'
candidate_path = D/'natural_v22b_sequence_v1'
base, candidate = map(session, (baseline_path, candidate_path))
source = D/'candidate_setup_freeze_v22b.py'
archive = O/'release/runs/a_astra_runtime_v22b_1fdc8d9b/submission.tar.gz'
assert candidate['sha256'] == sha(source)
assert len(base['rows']) == len(candidate['rows']) == 10
rows = []
for before, after in zip(base['rows'], candidate['rows']):
    rank = after['rank']
    assert rank == before['rank']
    byte_parity = (baseline_path/f'pred_{rank}.bin').read_bytes() == (candidate_path/f'pred_{rank}.bin').read_bytes()
    flop_parity = before['flops'] == after['flops']
    assert byte_parity and flop_parity and after['nonfinite'] == 0
    assert after['natural_gc']['enabled'] is True
    assert after['alias_hold_lifetime']['live'] == 0
    rows.append(dict(rank=rank, byte_parity=byte_parity, flop_parity=flop_parity,
        flops=after['flops'], requests=after['requests'], request_delta=after['requests']-before['requests'],
        residual_instructions=after['residual_instructions'],
        residual_instruction_ratio=after['residual_instructions']/before['residual_instructions'],
        wall_s=after['wall_s'], kernel_s=after['kernel_s'], peak_handles=after['peak_handles'],
        frozen_objects=after['natural_gc']['frozen_objects'],
        gen2_events=[x for x in after['natural_gc']['events'] if x['generation'] == 2],
        outside_gc_s=sum(x['duration_s'] for x in after['natural_gc']['events'] if x['dispatch_depth'] == 0)))

before_events = [dict(rank=r['rank'], **x) for r in base['rows'] for x in r['natural_gc']['events'] if x['generation'] == 2]
after_events = [dict(rank=r['rank'], **x) for r in candidate['rows'] for x in r['natural_gc']['events'] if x['generation'] == 2]
before_max = max(x['duration_s'] for x in before_events)
after_max = max(x['duration_s'] for x in after_events)
stress = load(candidate_path/'end_gc_stress.json')
assert stress['automatic_gc_enabled'] and stress['threshold'] == [700, 10, 10]
guard = load(D/'GUARD_NATURAL_V22B_SEQUENCE_v1.json')
assert guard['exit_code'] == 0 and guard['whole_group_retired'] and guard['error'] is None
startup = load(D/'CONCURRENT_V22B_v1_SUMMARY.json')
assert startup['passed'] and all(x['start_envelope_s'] < 5 for x in startup['results'])
stage = load(archive.parent/'stage_a.json')
assert stage['passed'] and stage['candidate_sha256'] == sha(source)

# This is only a transparent sensitivity calculation. It does not establish that
# either remote failure was caused by GC, nor replace a completed-platform grade.
warm = candidate['rows'][1:]
warm_mean_instructions = statistics.mean(r['residual_instructions'] for r in warm)
stress_ratio = stress['client']['instructions']/warm_mean_instructions
public = load(D/'PUBLIC_COMPARISON_v1.json')['rows']
ordinary_warm = [r['residual'] for r in public if not r['cold'] and r['state'] == 'completed']
ordinary_max = max(ordinary_warm)
ordinary_mean = statistics.mean(ordinary_warm)
sensitivity = dict(reference='336657 completed warm residuals, excluding the two failed outliers',
    ordinary_max_s=ordinary_max, ordinary_mean_s=ordinary_mean,
    local_warm_mean_instructions=warm_mean_instructions,
    end_stress_instructions=stress['client']['instructions'],
    stress_to_ordinary_instruction_ratio=stress_ratio,
    additive_stress_at_ordinary_max_s=ordinary_max*stress_ratio,
    ordinary_max_plus_one_stress_s=ordinary_max*(1+stress_ratio),
    caution='Conditional CPU scaling only; local and platform hardware differ, and original failure causality is unproven.')

result = dict(status='REVIEW_READY_PENDING_GC_IMPORT_EXCEPTION', uploaded=False,
    source=str(source), source_sha256=sha(source), archive=str(archive), archive_sha256=sha(archive),
    stage_a=str(archive.parent/'stage_a.json'), baseline=str(baseline_path), candidate_sequence=str(candidate_path),
    source_changes=['Break setup-owned recursive closure and facade cycles without changing public calls.',
                    'After successful setup, explicitly collect temporary cycles and freeze the live setup graph.',
                    'Before repeated setup, unfreeze our previous graph before parent reset; prediction code is unchanged.'],
    gc_policy=dict(automatic_gc_enabled=True, thresholds=[700,10,10],
                   collect_and_freeze_in_setup_only=True, no_predict_hook=True, no_meter_or_dispatch_hook_in_source=True,
                   frozen_objects_after_setup=candidate['setup']['gc_state']['frozen_objects']),
    numerical=dict(all_10_byte_exact=True, all_10_flops_exact=True, rows=rows),
    natural_gc=dict(baseline_events=before_events, candidate_events=after_events,
                    baseline_max_s=before_max, candidate_max_s=after_max,
                    maximum_pause_reduction=1-after_max/before_max,
                    baseline_outside_gen2=[x for x in before_events if x['dispatch_depth'] == 0],
                    candidate_outside_gen2=[x for x in after_events if x['dispatch_depth'] == 0],
                    causal_reproduction=False),
    end_gc_stress=stress, conditional_residual_sensitivity=sensitivity,
    startup=startup, startup_guard=load(D/'GUARD_CONCURRENT_V22B_v1.json'), prediction_guard=guard,
    parity_and_lifetime_proofs=[str(R/'astra_crunch_physics_v1/gc_repair336657_v1/OWNED_CYCLES_TINY_v1.json'),
                                str(R/'astra_crunch_physics_v1/gc_repair336657_v1/FREEZE_WRAPPER_TINY_v1.json')],
    policy_gate=dict(status='BLOCKED_PENDING_USER_EXCEPTION', new_import='gc',
                     reason='User-specific handoff import allowlist excludes gc. Root must obtain a narrow explicit exception; the checker has not been bypassed.'),
    limitations=['No outside-dispatch gen2 reproduced in the unchanged natural baseline; GC remains a plausible mechanism, not an established remote cause.',
                 'The ten measured predictions are existing local fixtures, not the inaccessible failed public weights.',
                 'Harness callbacks and reporting allocate objects; collection is natural under instrumentation, not a zero-overhead replica.',
                 'Only60ms worst local concurrent startup margin; the unchanged routing parent passed actual remote setup4.07–4.216s.',
                 'The end GC scan is a separately labeled diagnostic, excluded from prediction metrics and score claims.'])
(D/'REVIEW_v22b_v1.json').write_text(json.dumps(result, indent=1)+'\n')
print(json.dumps({k:result[k] for k in ('status','source_sha256','archive','archive_sha256')}))
print(json.dumps(dict(gen2_max_before=before_max, gen2_max_after=after_max,
    reduction=1-after_max/before_max, stress_wall_s=stress['wall_s'], stress_instructions=stress['client']['instructions'],
    conditional=sensitivity)))
