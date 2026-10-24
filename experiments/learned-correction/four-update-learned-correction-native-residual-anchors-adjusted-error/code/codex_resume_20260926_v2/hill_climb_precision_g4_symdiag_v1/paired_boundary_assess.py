"""One fixed contemporary boundary pair; JSON/source only, no arrays or model.

This outcome-informed resolution preserves the original failed assessment.
Run once under the root's assessment guard; no retries or automatic progression.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys

D = Path(__file__).resolve().parent
PROTOCOL_SHA = '06dedc12838981b01473ba5b9c7050debf2e54b6b0e6269deb4112af26b05f84'
HELPER_SHA = 'a8cc798f6d69ed378b0086520b56cad98d93d31337f6abdf02efc08c416a3f03'
FAILURE_SHA = 'f8ab9902712e8d2f903713c73f0e2d3349300e82bb298d88cac54e1524903e22'
FAILURE_GUARD_SHA = 'fc146b75db167f9bbd055be5ee8335675cefe24abd1470b5c870018d719d819a'
LIMITS = dict(seconds=240, rss_bytes=7500000000, added_swap_bytes=0,
              setup_s_max=4.1, residual_s_max=.35, projected_wall_s_max=105,
              cpu_disagreement_pp=5)


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def helper():
    path = D/'boundary_feasibility_assess.py'
    assert pin(path)['sha256'] == HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_boundary_feasibility_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Definitions and stdlib imports only; no main().
    module.checked(pin(path))
    return module


def setup_record(q, path, bound, fused):
    text = q.checked(pin(path)).decode(); decoder = json.JSONDecoder(); records = []
    for start, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'setup_wall_s' in value:
            records.append(value)
    assert len(records) == 1
    value = records[0]
    assert math.isfinite(value['setup_wall_s']) and value['setup_wall_s'] >= 0
    assert [r[1] for r in value['setup_log'] if r[0] == 'bound'] == [bound]
    assert [r[1] for r in value['setup_log'] if r[0] == 'lvfuse'] == [fused]
    return value


def assess():
    q = helper()
    protocol_pin = q.fixed(D/'paired_boundary_protocol.json', PROTOCOL_SHA)
    protocol = json.loads(q.checked(protocol_pin))
    assert protocol['ranks'] == [35, 100] and protocol['limits'] == LIMITS
    assert protocol['selection_is_outcome_informed'] is True
    assert protocol['fresh_outcomes_seen'] is False and protocol['original_failure_preserved'] is True
    assert protocol['no_automatic_retry'] is True and protocol['full_serving_still_required'] is True
    for name in ('candidate','parent','original_assessment','original_assessor',
                 'original_assess_guard','extension','driver','harness','native8'):
        q.checked(protocol[name])
    candidate = q.fixed(D/'candidate.py', q.CANDIDATE_SHA)
    parent = q.fixed(q.H2/'hill_climb_star_projection32_v1/candidate.py', q.S32_SHA)
    assert protocol['candidate'] == candidate and protocol['parent'] == parent
    assert protocol['driver'] == q.fixed(q.H2/'run_boundary_census.py', q.DRIVER_SHA)
    assert protocol['harness'] == q.fixed(q.H2/'census_boundary_setup_snapshot_v1.py', q.HARNESS_SHA)
    q.fixed(q.P2/'fable_porter_v1/census_boundary_setup_v2.py', q.HARNESS_SHA)
    assert protocol['original_assessor'] == q.fixed(D/'boundary_feasibility_assess.py', HELPER_SHA)
    assert protocol['original_assessment'] == q.fixed(D/'boundary_assessment.json', FAILURE_SHA)
    assert protocol['original_assess_guard'] == q.fixed(D/'boundary_assess_guard.json', FAILURE_GUARD_SHA)
    q.guard(protocol['original_assess_guard'],
            [str(q.LIGHT),'-B',str(D/'boundary_feasibility_assess.py')], 120)
    old = json.loads(q.checked(protocol['original_assessment']))
    assert old['passed'] is False and old['receipt_checks_passed'] is True
    assert old['candidate'] == candidate and old['parent'] == parent
    assert old['assessor'] == protocol['original_assessor']
    assert old['gates'] == dict(setup=True,residual_projection=False,runtime_projection=True,cpu_agreement=True)
    assert math.isclose(old['residual']['projected_s'], .492650751, rel_tol=0, abs_tol=1e-12)
    for record in old['evidence']:
        assert Path(record['path']).suffix not in ('.npy','.npz')
        q.checked(record)
    original_admission = json.loads(q.checked(old['admission']))
    assert original_admission['candidate'] == candidate
    assert original_admission['protocol'] == q.fixed(D/'PROTOCOL.md', q.PROTOCOL_SHA)
    assert original_admission['build'] == q.fixed(D/'build.json', q.BUILD_SHA)
    assert protocol['extension'] == old['extension_admission']
    extension = json.loads(q.checked(protocol['extension']))
    assert protocol['extension'] == q.fixed(D/'extension_admission.json',
        '3a3121075e58185c5c3711c6ee32253cb017ff2d156d6adf35a9980fb4639f7f')
    assert extension['candidate'] == candidate and extension['outcome_informed_selection'] is True
    assert extension['early_boundary_required_before_valid40'] is True
    assert extension['original_component_admitted'] is False
    assert protocol['native8'] == extension['training'] == old['native8']
    native8 = json.loads(q.checked(protocol['native8']))
    assert native8['candidate'] == candidate and native8['passed_integrity'] is True
    assert native8['admitted'] is False and native8['n'] == 8
    assert native8['gates'] == dict(adjusted_gain=True,wins=True,ci_upper_negative=False)
    assert native8['assessor'] == q.fixed(D/'assess.py', q.NATIVE_ASSESSOR_SHA)
    s32 = native8['versus_s32']
    assert s32['adjusted_pct'] <= -1 and s32['adjusted_wins'] == 8 and s32['adjusted_ci95_pct'][1] < 0
    q.guard(extension['training_guard'], [str(q.NATIVE),'-B',str(D/'assess.py'),'8'], 120)
    native, native2_ratio, native8_ratio = q.native_panels(native8, candidate)

    assert [job['role'] for job in protocol['jobs']] == ['s32','candidate']
    rows = {}; jobs = []
    for job in protocol['jobs']:
        role = job['role']; source = parent if role == 's32' else candidate
        tag = 'hill_climb_precision_g4_symdiag_v1_boundary_fresh_'+role+'_v1'
        assert job['tag'] == tag and job['source'] == source
        expected_command = [str(q.LIGHT),'-B',str(q.H2/'run_boundary_census.py'),source['path'],'35,100',tag]
        assert job['command'] == expected_command
        assert Path(job['guard_path']) == D/('paired_'+role+'_guard.json')
        assert Path(job['log_path']) == D/('paired_'+role+'.log')
        census = pin(q.H2/(tag+'_census.json')); launcher = pin(q.H2/(tag+'_launcher.json'))
        gp = pin(job['guard_path'])
        data, g = q.boundary(census, launcher, gp, source, tag)
        setup = setup_record(q, job['log_path'], 624 if role == 's32' else 673,
                             600 if role == 's32' else 649)
        for row, nr in zip(data, native[role]):
            assert row['rank'] == nr['rank'] and row['flops'] == nr['flops']
        rows[role] = data
        jobs.append(dict(role=role, source=source, census=census, launcher=launcher,
                         guard=gp, log=pin(job['log_path']), setup=setup,
                         maximum_rss_bytes=g['max_rss_bytes'], full_native_bills_exact=True))
    comparisons = [dict(rank=c['rank'], candidate_residual_s=c['residual_s'],
        control_residual_s=b['residual_s'], residual_delta_s=c['residual_s']-b['residual_s'],
        request_delta=c['requests']-b['requests'], server_cpu_ratio=c['server_cpu_s']/b['server_cpu_s'])
        for c,b in zip(rows['candidate'],rows['s32'])]
    official_pin = q.fixed(q.H2/'eval_332738.json',
        '6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6')
    assert official_pin == old['official_evaluation']
    official = json.loads(q.checked(official_pin)); evaluation = official['evaluation']
    assert official['submission_id'] == 332738 and evaluation['submission_id'] == '332738'
    assert evaluation['progress']['mlps_completed'] == 100 and evaluation['progress']['mlps_failed'] == 0
    telemetry = [r['telemetry'] for r in evaluation['results']['per_mlp']]
    assert len(telemetry) == 100
    assert all(math.isfinite(t[k]) and t[k] >= 0 for t in telemetry for k in
               ('participant_predict_wall_s','flopscope_server_kernel_wall_s','residual_wall_time_s'))
    max_residual = max(t['residual_wall_time_s'] for t in telemetry)
    assert math.isclose(max_residual,.285450751,rel_tol=0,abs_tol=1e-12)
    assert math.isclose(max(t['participant_predict_wall_s'] for t in telemetry),82.952904334,rel_tol=0,abs_tol=1e-9)
    boundary_ratio = statistics.mean(r['server_cpu_ratio'] for r in comparisons)
    mismatch = abs(100*(boundary_ratio-native2_ratio))
    used_change = max(0.,native2_ratio-1,native8_ratio-1,boundary_ratio-1)
    request_delta = max(0.,statistics.mean(r['request_delta'] for r in comparisons))
    grid = [dict(native_fraction=fraction,request_cost_s=cost,
        projected_worst_wall_s=max(t['participant_predict_wall_s']+
            fraction*used_change*t['flopscope_server_kernel_wall_s']+request_delta*cost for t in telemetry))
        for fraction in (.60,.78,.95) for cost in (.00030,.00041,.00050)]
    projected_wall = max(x['projected_worst_wall_s'] for x in grid)
    projected_residual = max_residual+4*max(0.,*(r['residual_delta_s'] for r in comparisons))
    gates = dict(setup=all(j['setup']['setup_wall_s']<=4.1 for j in jobs),
        residual_projection=projected_residual<=.35,runtime_projection=projected_wall<=105,
        cpu_agreement=mismatch<=5)
    result = dict(passed=all(gates.values()),receipt_checks_passed=True,
        stage='FIXED_CONTEMPORARY_BOUNDARY_PAIR_RESOLUTION',candidate=candidate,parent=parent,
        assessor=pin(__file__),protocol=protocol_pin,jobs=jobs,gates=gates,comparisons=comparisons,
        original_assessment=protocol['original_assessment'],original_assessment_passed=False,
        original_projected_residual_s=old['residual']['projected_s'],
        original_failure_preserved=True,outcome_informed_resolution=True,
        reason=protocol['reason'],extension=protocol['extension'],native8=protocol['native8'],
        original_component_admitted=False,matched_control_performed=True,no_automatic_retry=True,
        official_evaluation=official_pin,
        residual=dict(official_s32_max_s=max_residual,multiplier=4,projected_s=projected_residual,gate_s=.35),
        runtime=dict(native2_cpu_ratio=native2_ratio,native8_cpu_ratio=native8_ratio,
            boundary_mean_cpu_ratio=boundary_ratio,native2_boundary_disagreement_pp=mismatch,
            disagreement_limit_pp=5,used_nonnegative_cpu_change=used_change,
            nonnegative_mean_request_delta=request_delta,projection_grid=grid,
            projected_worst_wall_s=projected_wall,gate_s=105),
        maximum_boundary_rss_bytes=max(j['maximum_rss_bytes'] for j in jobs),
        evidence=list(q.EVIDENCE.values()),output_parity_claim=False,worker20_qualified=False,
        setup_stress_qualified=False,release_qualified=False,submitted=False,arrays_read=False,
        model_executed=False,
        interpretation='Separate outcome-informed resolution using the one fixed contemporary pair. '
            'The original failure remains false. A passing result is early feasibility only, '
            'not measured official candidate runtime, broad validation, or full serving qualification.')
    for record in list(q.EVIDENCE.values()):
        q.checked(record)
    return result


def main():
    assert len(sys.argv) == 1
    output = D/'paired_boundary_assessment.json'
    assert not output.exists()
    try:
        result = assess()
    except Exception as exc:
        result = dict(passed=False,receipt_checks_passed=False,
            stage='FIXED_CONTEMPORARY_BOUNDARY_PAIR_RESOLUTION',
            error=dict(type=type(exc).__name__,message=str(exc)),assessor=pin(__file__),
            original_failure_preserved=True,release_qualified=False,submitted=False,
            arrays_read=False,model_executed=False)
        with output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        raise
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','receipt_checks_passed','gates')}))


if __name__ == '__main__':
    main()
