"""Prospective prefix128 resource-only boundary2: current-winner residual reference, unchanged S32 wall/CPU recipe.

LIGHT -B assess.py STUDY/boundary_config.json CONFIG_SHA. Native2 is sufficient
for this early rejection gate. Score and runtime comparators are explicit;
actual native8 timing must later be divided by the same pinned S32 controls.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys

H2 = Path(__file__).resolve().parent.parent
V1 = H2/'hill_climb_precision_g4_symdiag_v1'
HELPER_SHA = '652650baaa5f525f09d43d08cf8cf6c181baacf4ade5637e424b63802432f2d2'
CONTROL_SHA = '9091ca7be8089e8e170540309dd558433a8116d76648b6b854d068af526afbce'
CONTROL_GUARD_SHA = '564a6609d319ed2e996cd5e408f9b6f205a82b30157a1fb35aa64d3c02770366'
RUNTIME_CONFIG_SHA = '5c4bf41c38012ee1a4e4af0d706d33f6f8902f52f4a32c899c66a3506aaf8c8d'
ROSTERS = {2: [35, 100], 6: [36, 45, 52, 58, 64, 70]}
LIMITS = dict(seconds=240, rss_bytes=7500000000, added_swap_bytes=0,
              setup_s_max=4.1, residual_s_max=.4, projected_wall_s_max=105,
              cpu_disagreement_pp=5)


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def helpers():
    path = V1/'paired_boundary_assess.py'
    assert pin(path)['sha256'] == HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_boundary_pair_helpers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    q = module.helper()  # Frozen stdlib definitions only; no assessor main runs.
    q.checked(pin(path))
    return module, q


def native_arm(q, arm, size, records, cache_limit=512000000):
    block = arm['blocks'][str(size)]; folder = H2/block['tag']
    ranks = ROSTERS[size]; seconds = 150 if size == 2 else 300
    assert block['ranks'] == ranks
    for key, path in (('manifest', folder/'manifest.json'), ('rows', folder/'rows.jsonl'),
                      ('guard', Path(block['guard']))):
        assert records[key] == pin(path); q.checked(records[key])
        if 'artifacts' in block: assert records[key] == block['artifacts'][key]
    manifest = json.loads(q.checked(records['manifest']))
    assert manifest['candidate'] == arm['source']['path']
    assert manifest['sha256'] == arm['source']['sha256'] and manifest['ranks'] == ranks
    assert manifest['runtime'] == str(q.NATIVE) and manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == dict.fromkeys(
        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'), '1')
    command = [str(q.NATIVE)] + (['-B'] if block['no_bytecode'] else [])
    command += [block['driver']['path'], arm['source']['path'], ','.join(map(str, ranks)), block['tag']]
    q.checked(block['driver']); q.checked(arm['source'])
    assert block['limits'] == dict(seconds=seconds, rss_bytes=7500000000)
    guard_normalized(q, records['guard'], command, seconds)
    assert 0 <= manifest['setup']['setup_s'] <= 4.1
    for key, count in arm['setup']['counts'].items():
        assert [r[1] for r in manifest['setup']['pb_log'] if r[0] == key] == [count]
    rows = [json.loads(line) for line in q.checked(records['rows']).decode().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    for row in rows:
        assert math.isfinite(row['cpu_s']) and row['cpu_s'] > 0
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        if arm['ubc_retired_required']:
            assert row['ubc']['retired'] is True
            assert row['ubc']['retained_cache_bytes'] == row['ubc']['live_cache_bytes'] == 0
            assert 0 <= row['ubc']['max_resident_bytes'] <= cache_limit
    return rows


def native_two(q, receipt, cfg):
    assert len(receipt['blocks']) == 1 and receipt['blocks'][0]['size'] == 2
    outputs = {role: native_arm(q, cfg['arms'][role], 2, receipt['blocks'][0]['arms'][role], 512000000)
               for role in ('candidate', 'base')}
    ratio = math.fsum(r['cpu_s'] for r in outputs['candidate']) / math.fsum(r['cpu_s'] for r in outputs['base'])
    return outputs, ratio


def runtime_controls(q, c):
    reference = c['runtime_reference']
    assert reference['official_submission_id'] == 332738 and reference['native_arm'] == 'base'
    assert reference['source'] == q.fixed(H2/'hill_climb_star_projection32_v1/candidate.py', q.S32_SHA)
    assert reference['native_config'] == q.fixed(
        H2/'hill_climb_precision_g3_symdiag128_v1/native_screen.json', RUNTIME_CONFIG_SHA)
    original = json.loads(q.checked(reference['native_config']))
    assert original['schema'] == 'native-screen-v1' and original['engine'] == c['control_native_engine']
    assert original['arms']['base']['source'] == reference['source']
    arm = original['arms']['base']; rows = {}; artifacts = {}
    for size in (2, 6):
        records = {key: arm['blocks'][str(size)]['artifacts'][key] for key in ('manifest', 'rows', 'guard')}
        rows[size] = native_arm(q, arm, size, records)
        artifacts[str(size)] = records
    return rows, dict(**reference, native_controls=artifacts,
        native8_cpu_ratio_rule='sum(candidate TRAIN2+TRAIN6 cpu_s) / sum(pinned S32 TRAIN2+TRAIN6 cpu_s)',
        score_comparator_ratio_is_not_runtime_ratio=True)


def guard_normalized(q, record, command, seconds):
    # Match actual nice child argv semantics; raw receipt remains unchanged.
    g = json.loads(q.checked(record))
    actual = g['command'][1:] if g['command'][:1] == ['--'] else g['command']
    assert actual == command
    assert g['completed'] and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows']
    assert g['rss_trigger_bytes'] == 7500000000 and 0 < g['max_rss_bytes'] < 7500000000
    assert g['max_swap_growth_bytes'] == 0 and 0 < g['elapsed_s'] <= seconds
    return g


def verify_cheap_evidence(q, c):
    d = Path(c['candidate']['path']).parent
    cheap = json.loads(q.checked(c['cheap_gate']))
    root = json.loads(q.checked(c['root_cheap_gate']))
    assert c['root_cheap_gate']['sha256'] == '7b20f949b6f34d759985975e77b553395247afdb808e063c45309a6f6e14ad71'
    assert cheap['candidate'] == root['candidate'] == c['candidate']
    assert cheap['base'] == c['native_comparator'] and cheap['passed'] and root['passed']
    assert cheap['all16_bytes_equal'] and cheap['flops_equal'] and cheap['requests_lower_each_rank']
    source_protocol = json.loads(q.checked(c['source_protocol']))
    q.checked(source_protocol['degraded_probe'])
    q.checked(root['source_review'])
    assert root['protocol'] == c['source_protocol']
    for record in cheap['evidence']:
        q.checked(record)
    normal_driver = H2/'hill_climb_essential_fusion_v1/probe_driver.py'
    assert pin(normal_driver)['sha256'] == '2fb0035189788e763ecec94a00c25dc0d71cfdbdd480b847e11c595d55e93cbe'
    q.checked(pin(normal_driver))
    for i, block in enumerate(root['normal_setup3'], 1):
        tag = d.name+f'_normal_setup{i}_v1'
        assert block['receipt'] == pin(H2/(tag+'_setup.json'))
        s = json.loads(q.checked(block['receipt']))
        assert s['candidate'] == c['candidate']['path'] and s['candidate_sha256'] == c['candidate']['sha256']
        assert s['mode'] == 'normal' and s['phase1_deadline'] == 4.0
        assert 0 <= s['setup_wall_s'] <= 4.1 and block['setup_s'] == s['setup_wall_s'] == cheap['normal_setup_seconds'][i-1]
        assert s['fused_entries'] == 649 and all(x['fused'] for x in s['essential'])
        for key, count in c['setup_counts'].items():
            assert [x[1] for x in s['setup_log'] if x[0] == key] == [count]
        guard_normalized(q, pin(d/f'normal_setup{i}_guard.json'),
            [str(q.NATIVE), '-B', str(normal_driver), c['candidate']['path'], '-', tag, 'normal'], 60)
    assert len(root['normal_setup3']) == 3
    a = H2/'hill_climb_all_eager_fusion_v1'
    driver = a/'probe_driver_degraded_v1.py'
    assert pin(driver)['sha256'] == 'fde8e8ab03a48b6c135992ed3b88d6843d2c89f144bf6b65e7f2e24aa742a700'
    q.checked(pin(driver))
    datasets = []
    for tag, source, gp, fused in (
            (a.name+'_phase1_3s_base_v1', c['native_comparator'], a/'phase1_3s_base_guard_v1.json', 0),
            (d.name+'_phase1_3s_candidate_v1', c['candidate'], d/'degraded_candidate_guard.json', 128)):
        s = json.loads(q.checked(pin(H2/(tag+'_setup.json'))))
        assert s['candidate_sha256'] == source['sha256'] and s['mode'] == 'phase1_3s'
        assert s['phase1_deadline'] == 4.0 and 0 <= s['setup_wall_s'] <= 4.1
        sim = s['phase1_simulation']
        assert sim['requested_phase1_budget_s'] == 3.0 and sim['restored_deadline_s'] == 4.0
        assert len(sim['phase1_bound_roster']) == 649 and len(sim['phase1_fused_roster']) == fused
        assert s['bound_entries'] == 649 and s['fused_entries'] == fused
        if fused:
            assert sim['phase1_fused_roster'] == source_protocol['prefix_keys']
        launch = json.loads(q.checked(pin(H2/(tag+'_launcher.json'))))
        assert launch['candidate'] == source['path'] and launch['candidate_sha256'] == source['sha256']
        assert launch['client_exit'] == 0 and launch['server_retired']
        guard_normalized(q, pin(gp), [str(q.NATIVE), '-B', str(driver), source['path'], '35,100', tag, 'phase1_3s'], 240)
        rows = json.loads(q.checked(pin(H2/(tag+'_census.json'))))
        assert [r['rank'] for r in rows] == [35, 100]
        for row in rows:
            p = pin(H2/f"{tag}_pred_{row['rank']}.bin")
            q.checked(p)
            assert p['sha256'] == row['prediction_sha256'] and p['bytes'] == 65536
            assert row['prediction_shape'] == [16, 1024] and row['prediction_dtype'] == 'float32' and row['nonfinite'] == 0
        datasets.append(rows)
    for b, x, check in zip(*datasets, root['rows']):
        assert b['prediction_sha256'] == x['prediction_sha256'] and b['flops'] == x['flops'] == x['flops_ctx']
        assert x['requests'] < b['requests'] and check['requests_saved'] == b['requests']-x['requests']
        assert check['rank'] == x['rank'] and check['all16_bytes_equal'] and check['flops_equal']
    failure = json.loads(q.checked(c['official_parent_failure']))
    # Preserved official failure is provenance, never a successful bank or timing anchor.
    assert failure['submission_id'] == 333099


def require_preserved_failure(old):
    boundary_keys = {'passed', 'receipt_checks_passed'}
    native_keys = {'admitted', 'passed_integrity'}
    boundary_present = boundary_keys.intersection(old)
    native_present = native_keys.intersection(old)
    if boundary_present == boundary_keys and not native_present:
        assert old['passed'] is False and old['receipt_checks_passed'] is True
    elif native_present == native_keys and not boundary_present:
        assert old['admitted'] is False and old['passed_integrity'] is True
    else:
        raise AssertionError('unknown, incomplete or mixed preserved-failure schema')


def assess(config_path, digest):
    m, q = helpers(); config_pin = pin(config_path)
    assert config_pin['sha256'] == digest and len(digest) == 64
    c = json.loads(q.checked(config_pin)); d = config_path.parent
    assert c['schema'] == 'prefix128-resource-s34-boundary-native2-v1' and c['prospective_authorization'] is True
    assert c['engine'] == pin(__file__) and c['limits'] == LIMITS and c['ranks'] == [35, 100]
    assert c['native8_runtime_recheck_required'] is True and c['no_automatic_retry'] is True
    assert c['tag'] == d.name+'_boundary_feasibility_v1'
    for key in ('engine', 'candidate', 'native_comparator', 'build', 'source_protocol', 'native_engine', 'driver', 'harness'):
        q.checked(c[key])
    assert c['driver'] == q.fixed(H2/'run_boundary_census.py', q.DRIVER_SHA)
    assert c['harness'] == q.fixed(H2/'census_boundary_setup_snapshot_v1.py', q.HARNESS_SHA)
    q.fixed(q.P2/'fable_porter_v1/census_boundary_setup_v2.py', q.HARNESS_SHA)
    build = json.loads(q.checked(c['build']))
    assert build['candidate'] == c['candidate'] and build['source_inverse']['exact_inverse'] is True
    admission_pin = pin(d/'boundary_admission.json'); a = json.loads(q.checked(admission_pin))
    assert a['root_admitted'] is True and a['source_changes'] is False and a['serving_qualified'] is False
    assert a['config'] == config_pin and a['assessor'] == c['engine'] and a['candidate'] == c['candidate']
    for key in ('native_config', 'native2', 'native2_guard', 'source_proof'):
        q.checked(a[key])
    assert a['native_config'] == pin(c['native_config_path'])
    nc = json.loads(q.checked(a['native_config']))
    assert nc['schema'] == 'native-screen-v1' and nc['engine'] == c['native_engine']
    assert nc['prospective_authorization'] is True and nc['future_only'] is True
    assert nc['arms']['candidate']['source'] == c['candidate']
    assert nc['arms']['base']['source'] == c['native_comparator']
    expected = [str(q.LIGHT), '-B', str(Path(__file__).resolve()), str(config_path), digest]
    assert nc['boundary_before8']['assessment_command'] == expected
    assert nc['boundary_before8']['assessor'] == c['engine']
    assert nc['boundary_before8']['receipt'] == str(d/'boundary_assessment.json')
    assert nc['boundary_before8']['assessment_guard'] == str(d/'boundary_assess_guard.json')
    assert nc['boundary_before8']['assessment_limits'] == dict(seconds=120, rss_bytes=7500000000)
    stage = nc['stages']['2']
    assert a['native2'] == pin(stage['result']) and a['native2_guard'] == pin(stage['assessment_guard'])
    r = json.loads(q.checked(a['native2']))
    assert r['passed_integrity'] is True and r['admitted'] is True and r['n'] == 2 and r['ranks'] == [35, 100]
    assert r['config'] == a['native_config'] and r['engine'] == c['native_engine']
    assert r['candidate'] == c['candidate'] and r['comparator'] == nc['arms']['base']['source']
    assert r['all16_predictions_finite'] is True and all(r['gates'].values())
    assert stage['gates'] == c['native2_gates']
    ng = c['native2_gates']; value = r['adjusted_pct']; cutoff = ng['adjusted_pct_at_most']
    assert value < cutoff if ng['strict_adjusted'] else value <= cutoff
    assert r['adjusted_wins'] >= ng['wins_min']
    assert stage['assessment_limits'] == dict(seconds=120, rss_bytes=7500000000)
    q.guard(a['native2_guard'], [str(q.NATIVE), '-B', c['native_engine']['path'],
            a['native_config']['path'], a['native_config']['sha256'], '2'], 120)
    native, component_cpu_ratio = native_two(q, r, nc)
    runtime_native, runtime_reference = runtime_controls(q, c)
    candidate_cpu_s = math.fsum(row['cpu_s'] for row in native['candidate'])
    s32_cpu_s = math.fsum(row['cpu_s'] for row in runtime_native[2])
    native2_ratio = candidate_cpu_s / s32_cpu_s
    # Exact prefix128 setup schedule; locally qualified FG control, fixed S34/S32 resources.
    assert a['source_proof'] == c['build']
    assert c['native_comparator'] == build['source_inverse']['parent'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert c['native_comparator']['sha256'] == '1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
    source = q.checked(c['candidate']).decode()
    inv = build['source_inverse']
    assert inv['parent'] == c['native_comparator'] and inv['exact_inverse']
    assert len(inv['edits']) == 1 and build['only_one_condition_changed'] and build['bind_order_unchanged']
    for edit in reversed(inv['edits']):
        assert source.count(edit['after']) == 1
        source = source.replace(edit['after'], edit['before'], 1)
    assert source == q.checked(c['native_comparator']).decode()
    ei = json.loads(q.checked(c['engine_inverse']))
    assert ei['engine'] == c['native_engine'] == nc['engine']
    assert ei['parent'] == c['control_native_engine'] and ei['exact_inverse']
    engine_source = q.checked(ei['engine']).decode()
    assert len(ei['edits']) == 3
    for edit in reversed(ei['edits']):
        assert engine_source.count(edit['after']) == 1
        engine_source = engine_source.replace(edit['after'], edit['before'], 1)
    assert engine_source == q.checked(ei['parent']).decode()
    assert ei['parent']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    rp = json.loads(q.checked(c['resource_protocol']))
    assert rp['candidate'] == c['candidate'] and rp['source_build'] == c['build'] and rp['engine_inverse'] == c['engine_inverse']
    assert rp['resource_only'] and rp['native2_and8_gates'] == c['native2_gates']
    assert nc['exact_flops_required'] and r['per_network_flops_equal']
    assert r['adjusted_pct'] == r['raw_pct'] == 0
    verify_cheap_evidence(q, c)
    assert c['candidate_cache_limit_bytes'] == c['control_cache_limit_bytes'] == c['score_parent_cache_limit_bytes'] == 512000000
    parent_n8 = json.loads(q.checked(c['score_parent_native8']))
    assert parent_n8['candidate'] == c['native_comparator'] and parent_n8['admitted'] and parent_n8['passed_integrity'] and all(parent_n8['gates'].values())
    assert parent_n8['all16_parent_bytes_exact']
    parent_cfg = json.loads(q.checked(parent_n8['config']))
    parent_ng = c['score_parent_native8_guard']
    q.guard(parent_ng, [str(q.NATIVE), '-B', parent_cfg['engine']['path'], parent_n8['config']['path'], parent_n8['config']['sha256'], '8'], 120)
    serving = json.loads(q.checked(c['score_parent_serving']))
    assert serving['passed'] and serving['receipt_checks_passed'] and serving['worker_qualified'] and serving['release_qualified']
    assert serving['candidate'] == c['native_comparator'] and serving['native8'] == c['score_parent_native8']
    assert serving['long_worker']['networks'] == 20 and serving['full_estimator_contract']
    assert serving['resources']['maximum_group_rss_bytes'] < 7500000000 and serving['resources']['maximum_swap_growth_bytes'] == 0
    q.guard(c['score_parent_serving_guard'], [str(q.NATIVE), '-B', serving['assessor']['path'], serving['config']['path'], serving['config']['sha256'], serving['validation_admission']['sha256']], 120)
    for record in (serving['assessor'], serving['config'], serving['validation_admission'], *serving['resources']['guards']):
        q.checked(record)
    for size in (2, 6):
        parent_records = next(block for block in parent_n8['blocks'] if block['size'] == size)['arms']['candidate']
        for role in ('base', 'parity_parent'):
            native_arm(q, nc['arms'][role], size, parent_records, 512000000)
            for entry in nc['arms'][role]['blocks'][str(size)]['artifacts']['predictions'].values():
                q.checked(entry)
    assert r['all16_parent_bytes_exact'] is True
    assert nc['parity_layers'] == r['parity_layers'] == list(range(16))
    assert r['configured_parent_bytes_exact'] is True
    # The continuation only re-assesses the same immutable pilot outputs.
    continuation = nc['continuation']
    original_config = json.loads(q.checked(continuation['pilot_config']))
    original_result = json.loads(q.checked(continuation['pilot_result']))
    assert original_config['prospective_scope'] == 'native2_only_until_own_boundary_continuation'
    for key in ('schema', 'engine', 'parity_layers', 'future_only', 'raw_error_veto'):
        assert nc[key] == original_config[key]
    assert nc['arms'] == original_config['arms']
    for stage_id in ('2', '8'):
        for key in ('gates', 'assessment_limits'):
            assert nc['stages'][stage_id][key] == original_config['stages'][stage_id][key]
    assert original_result['config'] == continuation['pilot_config']
    assert continuation['pilot_result'] == pin(original_config['stages']['2']['result'])
    assert continuation['pilot_assessment_guard'] == pin(d/'native2_assess_guard.json')  # Actual preserved guard; v2 config anticipated a different filename.
    q.guard(continuation['pilot_assessment_guard'], [str(q.NATIVE), '-B', c['native_engine']['path'],
        continuation['pilot_config']['path'], continuation['pilot_config']['sha256'], '2'], 120)
    excluded = {'config', 'evidence'}
    assert {k: v for k, v in r.items() if k not in excluded} == {k: v for k, v in original_result.items() if k not in excluded}
    for failure in c['preserved_failures']:
        old = json.loads(q.checked(failure))
        require_preserved_failure(old)

    control_pin = q.fixed(V1/'paired_boundary_assessment.json', CONTROL_SHA)
    paired = json.loads(q.checked(control_pin))
    cg = q.fixed(V1/'paired_boundary_assess_guard.json', CONTROL_GUARD_SHA)
    q.guard(cg, [str(q.LIGHT), '-B', str(V1/'paired_boundary_assess.py')], 120)
    assert paired['passed'] is False and paired['receipt_checks_passed'] is True
    control = paired['jobs'][0]
    assert control['role'] == 's32' and control['source'] == runtime_reference['source']
    baseline, bg = q.boundary(control['census'], control['launcher'], control['guard'], control['source'],
                             'hill_climb_precision_g4_symdiag_v1_boundary_fresh_s32_v1')
    base_setup = m.setup_record(q, control['log']['path'], 624, 600)
    census = pin(H2/(c['tag']+'_census.json')); launcher = pin(H2/(c['tag']+'_launcher.json'))
    gp = pin(d/'boundary_guard.json')
    current, g = q.boundary(census, launcher, gp, c['candidate'], c['tag'])
    setup = m.setup_record(q, d/'boundary.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])
    assert nc['arms']['candidate']['setup']['counts'] == c['setup_counts']
    for key, count in c['setup_counts'].items():
        assert [r[1] for r in setup['setup_log'] if r[0] == key] == [count]
    comparisons = []
    for i, (child, base) in enumerate(zip(current, baseline)):
        assert child['flops'] == native['candidate'][i]['flops'] and base['flops'] == runtime_native[2][i]['flops']
        comparisons.append(dict(rank=child['rank'], full_native_bills_exact=True,
            candidate_residual_s=child['residual_s'], control_residual_s=base['residual_s'],
            residual_delta_s=child['residual_s']-base['residual_s'], request_delta=child['requests']-base['requests'],
            server_cpu_ratio=child['server_cpu_s']/base['server_cpu_s']))
    official_pin = q.fixed(H2/'eval_332738.json', '6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6')
    official = json.loads(q.checked(official_pin)); ev = official['evaluation']
    assert official['submission_id'] == 332738 and ev['submission_id'] == '332738'
    assert ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
    telemetry = [x['telemetry'] for x in ev['results']['per_mlp']]; assert len(telemetry) == 100
    assert all(math.isfinite(t[k]) and t[k] >= 0 for t in telemetry for k in
               ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s', 'residual_wall_time_s'))
    maximum = max(t['residual_wall_time_s'] for t in telemetry)
    assert math.isclose(maximum, .285450751, rel_tol=0, abs_tol=1e-12)
    boundary_ratio = statistics.mean(x['server_cpu_ratio'] for x in comparisons)
    disagreement = abs(100*(boundary_ratio-native2_ratio))
    growth = max(0., native2_ratio-1, boundary_ratio-1)
    requests = max(0., statistics.mean(x['request_delta'] for x in comparisons))
    grid = [dict(native_fraction=f, request_cost_s=p, projected_worst_wall_s=max(
        t['participant_predict_wall_s']+f*growth*t['flopscope_server_kernel_wall_s']+requests*p for t in telemetry))
        for f in (.60, .78, .95) for p in (.00030, .00041, .00050)]
    wall = max(x['projected_worst_wall_s'] for x in grid)
    bank = json.loads(q.checked(c['s34_bank']))
    assert c['s34_bank']['sha256'] == '8b2cba1ddc5381b4ce70948107511bb888fe86fb3f8766d80abac4417f37d0e2' and bank['submission_id'] == 333050
    assert bank['candidate'] == c['parent'] and bank['source_sha256'] == c['parent']['sha256']
    assert nc['arms']['base']['source'] == nc['arms']['parity_parent']['source'] == c['native_comparator']
    assert c['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    q.checked(c['candidate']); q.checked(c['parent'])
    assert bank['authenticated_poll']['grading_status_cd'] == 'graded'
    assert bank['evaluation_receipt'] == c['s34_evaluation']
    evaluation = json.loads(q.checked(c['s34_evaluation']))
    assert evaluation['submission_id'] == 333050 and evaluation['grading_status_cd'] is None
    ev = evaluation['evaluation']
    assert ev['state'] == 'completed' and ev['error'] is None
    assert ev['submission_id'] == '333050' and ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
    s34_telemetry = [row['telemetry'] for row in ev['results']['per_mlp']]; assert len(s34_telemetry) == 100
    assert all(math.isfinite(t['residual_wall_time_s']) and t['residual_wall_time_s'] >= 0 for t in s34_telemetry)
    s34_maximum = max(t['residual_wall_time_s'] for t in s34_telemetry)
    assert s34_maximum == bank['telemetry']['max_residual_wall_s']
    parent_receipt = json.loads(q.checked(c['parent_boundary']))
    assert parent_receipt['receipt_checks_passed'] is True and parent_receipt['candidate'] == c['parent']
    assert parent_receipt['setup']['setup_wall_s'] <= 4.1
    tag = Path(c['parent']['path']).parent.name+'_boundary_feasibility_v1'
    parent_rows, parent_guard = q.boundary(parent_receipt['current_census'], parent_receipt['launcher'],
        parent_receipt['guard'], c['parent'], tag)
    s34_cfg = json.loads(q.checked(c['s34_native_config']))
    assert s34_cfg['engine'] == c['control_native_engine'] and s34_cfg['arms']['base']['source'] == c['parent']
    s34_arm = s34_cfg['arms']['base']
    s34_records = {key: s34_arm['blocks']['2']['artifacts'][key] for key in ('manifest', 'rows', 'guard')}
    native_parent_rows = native_arm(q, s34_arm, 2, s34_records)
    assert [row['rank'] for row in parent_rows] == [row['rank'] for row in current] == [35, 100]
    assert [(row['rank'], row['flops']) for row in parent_rows] == [(row['rank'], row['flops']) for row in native_parent_rows]
    residual_comparisons = [dict(rank=x['rank'], candidate_residual_s=x['residual_s'], parent_residual_s=y['residual_s'],
        residual_delta_s=x['residual_s']-y['residual_s']) for x, y in zip(current, parent_rows)]
    residual = s34_maximum+4*max(0., *(x['residual_delta_s'] for x in residual_comparisons))
    gates = dict(setup=max(setup['setup_wall_s'], base_setup['setup_wall_s']) <= 4.1,
        residual_projection=residual < .4, runtime_projection=wall <= 105, cpu_agreement=disagreement <= 5)
    for record in list(q.EVIDENCE.values()): q.checked(record)
    return dict(passed=all(gates.values()), receipt_checks_passed=True, stage='EARLY_BOUNDARY2_AFTER_NATIVE2',
        candidate=c['candidate'], native_comparator=c['native_comparator'],
        runtime_reference=runtime_reference, assessor=c['engine'], config=config_pin, admission=admission_pin,
        native2=a['native2'], native_config=a['native_config'], source_proof=a['source_proof'],
        control_resolution=control_pin, fixed_fresh_control=control, current_census=census, launcher=launcher,
        guard=gp, setup=setup, control_setup=base_setup, comparisons=comparisons, gates=gates,
        residual=dict(official_s34_max_s=s34_maximum, multiplier=4, projected_s=residual, gate_s=.4, strict_upper_bound=True),
        residual_reference=dict(source=c['parent'], official_bank=c['s34_bank'], official_evaluation=c['s34_evaluation'],
            parent_boundary=c['parent_boundary'], parent_census=parent_receipt['current_census'], parent_guard=parent_receipt['guard'],
            candidate_census=census, comparisons=residual_comparisons),
        runtime=dict(native2_cpu_ratio=native2_ratio, native2_cpu_reference=runtime_reference['source'],
            native2_ratio_inputs=dict(candidate_rows=r['blocks'][0]['arms']['candidate']['rows'],
                reference_rows=runtime_reference['native_controls']['2']['rows'],
                candidate_cpu_s=candidate_cpu_s, reference_cpu_s=s32_cpu_s, ranks=ROSTERS[2]),
            score_comparator_cpu_ratio=component_cpu_ratio, score_comparator=c['native_comparator'],
            boundary_mean_cpu_ratio=boundary_ratio,
            native2_boundary_disagreement_pp=disagreement, disagreement_limit_pp=5, used_nonnegative_cpu_change=growth,
            nonnegative_mean_request_delta=requests, projection_grid=grid, projected_worst_wall_s=wall, gate_s=105),
        maximum_boundary_rss_bytes=max(g['max_rss_bytes'], bg['max_rss_bytes'], parent_guard['max_rss_bytes']), evidence=list(q.EVIDENCE.values()),
        preserved_failures=c['preserved_failures'], no_control_reselection=True, no_automatic_retry=True,
        native8_runtime_recheck_required=True, native8_runtime_recheck_contract='direct_candidate_over_pinned_S32_v2',
        existing_v1_broad_serving_engines_compatible=False, output_parity_claim=False, worker20_qualified=False,
        setup_stress_qualified=False, release_qualified=False, submitted=False, arrays_read=False, model_executed=False)


def main():
    assert len(sys.argv) == 3
    config = Path(sys.argv[1]).resolve(); target = config.parent/'boundary_assessment.json'
    assert not target.exists()
    try: result = assess(config, sys.argv[2])
    except Exception as exc:
        result = dict(passed=False, receipt_checks_passed=False, assessor=pin(__file__),
            error=dict(type=type(exc).__name__, message=str(exc)), release_qualified=False,
            submitted=False, arrays_read=False, model_executed=False)
        with target.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        raise
    with target.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'receipt_checks_passed', 'gates')}))


if __name__ == '__main__': main()
