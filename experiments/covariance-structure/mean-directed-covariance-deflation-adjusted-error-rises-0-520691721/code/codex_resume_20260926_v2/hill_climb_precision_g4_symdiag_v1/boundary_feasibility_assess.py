"""Offline early boundary feasibility only; stdlib JSON/source, no model/arrays/API.

Run LIGHT -B boundary_feasibility_assess.py under the root's assessment guard.
One exclusive receipt; needs_control blocks passage rather than guessing a resolution.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

D = Path(__file__).resolve().parent
H2, P2 = D.parent, D.parent.parent
ROOT = P2.parents[3]
LIGHT = ROOT/'.venv/bin/python'
NATIVE = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
TAG = 'hill_climb_precision_g4_symdiag_v1_boundary_feasibility_v1'
CANDIDATE_SHA = '09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
BUILD_SHA = 'b14b262bb0c35cfb615bcee0be04a8e52a8ac739acd93064bb3060a6bc824315'
PROTOCOL_SHA = '04645d4f06564dd645ea13d00d653288bb80622bd92a3bc3fb1f0eb9f60e64dc'
NATIVE_ASSESSOR_SHA = '21b40f45c8c8fa8b0a136de70bd4249464c2c1337b59315b0cd406bf88e1fbb5'
DRIVER_SHA = '5dac43b95a0c96c62501cdf7322113f4baaa7f9c1529fb5a29a54134a0778dd4'
HARNESS_SHA = 'ee9218fac30b82b4796bc3bb7ecc0c8421b732f4af9f9b9023034f6ca7777281'
LIMITS = dict(seconds=240, rss_bytes=7500000000, added_swap_bytes=0,
              setup_s_max=4.1, bound=673, fused=649, residual_s_max=.35,
              projected_wall_s_max=105, cpu_disagreement_pp=5)
EVIDENCE = {}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    assert pin(record['path']) == record, record['path']
    EVIDENCE[record['path']] = record
    return Path(record['path']).read_bytes()


def load(path):
    return json.loads(checked(pin(path)))


def fixed(path, sha):
    record = pin(path)
    assert record['sha256'] == sha, str(path)
    checked(record)
    return record


def guard(record, command, seconds):
    g = json.loads(checked(record))
    assert g['command'] == command
    assert g['completed'] and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows']
    assert g['rss_trigger_bytes'] == 7500000000 and 0 < g['max_rss_bytes'] < 7500000000
    assert g['max_swap_growth_bytes'] == 0 and 0 < g['elapsed_s'] <= seconds
    return g


def boundary(census, launcher, gp, source, tag):
    assert census == pin(H2/(tag+'_census.json')) and launcher == pin(H2/(tag+'_launcher.json'))
    data = json.loads(checked(census)); launch = json.loads(checked(launcher))
    assert [r['rank'] for r in data] == [35, 100]
    assert launch['candidate'] == source['path'] and launch['candidate_sha256'] == source['sha256']
    assert launch['client_exit'] == 0 and launch['server_retired'] is True
    assert type(launch['server_exit']) is int  # Expected SIGTERM is normal after the client succeeds.
    assert Path(launch['source']) == P2/'fable_porter_v1/census_boundary_setup_v2.py'
    assert launch['census_sha256'] == HARNESS_SHA
    g = guard(gp, [str(LIGHT), '-B', str(H2/'run_boundary_census.py'), source['path'], '35,100', tag], 240)
    for row in data:
        assert row['n_out'] == 16384 and row['nonfinite'] == 0
        assert type(row['flops']) is int and row['flops'] == row['flops_ctx'] and 0 < row['flops'] <= 2**41
        assert type(row['requests']) is int and row['requests'] > 0
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ('server_cpu_s', 'wall_s'))
        assert math.isfinite(row['residual_s']) and row['residual_s'] >= 0
    return data, g


def setup_record(path):
    text = checked(pin(path)).decode(); decoder = json.JSONDecoder(); records = []
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
    assert [r[1] for r in value['setup_log'] if r[0] == 'bound'] == [673]
    assert [r[1] for r in value['setup_log'] if r[0] == 'lvfuse'] == [649]
    return value


def native_panels(receipt, candidate):
    inventory = {r['path']: r for r in receipt['evidence']}
    def prior(path):
        return checked(inventory[str(path)])
    all_rows = {'candidate': [], 's32': []}
    two = {}
    for size, ranks in ((2, [35,100]), (6, [36,45,52,58,64,70])):
        for role in all_rows:
            tag = (f'hill_climb_precision_g4_symdiag_train{size}_v1' if role == 'candidate'
                   else f'hill_climb_star_projection32_train{size}_v1')
            folder = H2/tag
            m = json.loads(prior(folder/'manifest.json'))
            rr = [json.loads(line) for line in prior(folder/'rows.jsonl').decode().splitlines()]
            expected = candidate['sha256'] if role == 'candidate' else S32_SHA
            assert m['sha256'] == expected and m['ranks'] == ranks == [r['rank'] for r in rr]
            assert m['runtime'] == str(NATIVE) and m['numpy'] == '2.4.6'
            assert m['thread_env'] == dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1')
            assert pin(m['candidate'])['sha256'] == expected
            gp = D/f'native{size}_guard.json' if role == 'candidate' else H2/'hill_climb_star_projection32_v1'/f'train{size}_guard.json'
            prior(gp)
            command = json.loads(Path(gp).read_text())['command']
            if role == 'candidate':
                assert command == [str(NATIVE),'-B',str(H2/'hill_climb_uvp_base_bcache_probe_v1/native_probe.py'),candidate['path'],','.join(map(str,ranks)),tag]
            guard(inventory[str(gp)], command, 150 if size == 2 else 300)
            for r in rr:
                assert math.isfinite(r['cpu_s']) and r['cpu_s'] > 0
                assert type(r['flops']) is int and 0 < r['flops'] <= 2**41
            all_rows[role].extend(rr)
            if size == 2:
                two[role] = rr
    ratio = lambda data: math.fsum(r['cpu_s'] for r in data['candidate'])/math.fsum(r['cpu_s'] for r in data['s32'])
    return two, ratio(two), ratio(all_rows)


def assess():
    admission_pin = pin(D/'boundary_admission.json'); admission = load(D/'boundary_admission.json')
    assert admission['root_admitted'] is True and admission['source_changes'] is False
    assert admission['serving_qualified'] is False and admission['tag'] == TAG and admission['limits'] == LIMITS
    assert admission['assessor'] == pin(Path(__file__))
    for name in ('candidate','protocol','build','native8_assessment','native8_assess_guard','assessor','boundary_driver','boundary_harness','extension_admission','run_protocol'):
        checked(admission[name])
    candidate = fixed(D/'candidate.py', CANDIDATE_SHA)
    assert admission['candidate'] == candidate
    assert admission['protocol'] == fixed(D/'PROTOCOL.md', PROTOCOL_SHA)
    assert admission['build'] == fixed(D/'build.json', BUILD_SHA)
    assert admission['boundary_driver'] == fixed(H2/'run_boundary_census.py', DRIVER_SHA)
    assert admission['boundary_harness'] == fixed(H2/'census_boundary_setup_snapshot_v1.py', HARNESS_SHA)
    fixed(P2/'fable_porter_v1/census_boundary_setup_v2.py', HARNESS_SHA)
    run_protocol_pin = fixed(D/'boundary_run_protocol.json','f51f25d56b74b835540c8ba6f5016be95f9e727418538f79641d20421451d621')
    assert admission['run_protocol'] == run_protocol_pin
    run_protocol = json.loads(checked(run_protocol_pin))
    assert run_protocol['outcomes_read'] is False and run_protocol['ranks'] == [35,100]
    for key in ('root_admitted','candidate','protocol','build','native8_assessment','native8_assess_guard',
                'extension_admission','boundary_driver','boundary_harness','tag','limits','source_changes','serving_qualified'):
        assert admission[key] == run_protocol[key]
    assert admission['native8_assessment'] == pin(D/'native8_assessment.json')
    extension_pin = fixed(D/'extension_admission.json', '3a3121075e58185c5c3711c6ee32253cb017ff2d156d6adf35a9980fb4639f7f')
    assert admission['extension_admission'] == extension_pin
    extension = json.loads(checked(extension_pin))
    assert extension['candidate'] == candidate and extension['training'] == admission['native8_assessment']
    assert extension['training_guard'] == admission['native8_assess_guard']
    assert extension['protocol'] == admission['protocol'] and extension['build'] == admission['build']
    assert extension['outcome_informed_selection'] is True and extension['original_component_admitted'] is False
    assert extension['early_boundary_required_before_valid40'] is True and extension['serving_admitted'] is False
    assert extension['score_gates'] == dict(adjusted_pct_at_most=-1,wins_min=28,ci95_upper_below=0,max_per_network_regression_pct=10)
    assert extension['policy'] == fixed(H2/'hill_climb_policy_20260928_v2.json','26e99f9a54d97a4d6fca11ac477056219f7acf351d5f7327df9b6ff7cf898fa6')
    receipt = json.loads(checked(admission['native8_assessment']))
    assert receipt['passed_integrity'] and receipt['admitted'] is False and receipt['n'] == 8 and receipt['candidate'] == candidate
    assert receipt['protocol'] == admission['protocol'] and receipt['assessor'] == fixed(D/'assess.py', NATIVE_ASSESSOR_SHA)
    assert receipt['gates'] == dict(adjusted_gain=True,wins=True,ci_upper_negative=False)
    assert receipt['adjusted_pct'] <= -.05 and receipt['adjusted_wins'] >= 6 and receipt['adjusted_ci95_pct'][1] >= 0
    versus_s32 = receipt['versus_s32']
    assert versus_s32['adjusted_pct'] <= -1 and versus_s32['adjusted_wins'] == 8 and versus_s32['adjusted_ci95_pct'][1] < 0
    assert admission['native8_assess_guard'] == pin(D/'native8_assess_guard.json')
    guard(admission['native8_assess_guard'], [str(NATIVE),'-B',str(D/'assess.py'),'8'], 120)
    native, native2_ratio, native8_ratio = native_panels(receipt, candidate)
    parent = fixed(H2/'hill_climb_star_projection32_v1/candidate.py', S32_SHA)
    old_tag = 'hill_climb_star32_release_v1_residual_candidate'
    old_census = fixed(H2/(old_tag+'_census.json'), '3a2da86f649ca729288e0b0e7f27f4cea441ce48863a5be6273de58e68bcc430')
    old_launcher = fixed(H2/(old_tag+'_launcher.json'), 'a66dfdcfbb38f857cc9ddd45dee61d865f5be1f85bc51cb0b51cbab1815dcff8')
    old_guard = fixed(H2/'hill_climb_star32_release_v1/paired_candidate_guard.json', 'd6e51c9a97103ac0fe5a73943bee86cdd5369349f673047495bb68fd03b7af67')
    old, _ = boundary(old_census, old_launcher, old_guard, parent, old_tag)
    current_census, current_launcher, current_guard = (pin(H2/(TAG+'_census.json')), pin(H2/(TAG+'_launcher.json')), pin(D/'boundary_guard.json'))
    current, g = boundary(current_census, current_launcher, current_guard, candidate, TAG)
    setup = setup_record(D/'boundary.log')
    comparisons = []
    for index, (c, b) in enumerate(zip(current, old)):
        assert c['flops'] == native['candidate'][index]['flops']
        assert b['flops'] == native['s32'][index]['flops']
        comparisons.append(dict(rank=c['rank'], native_flops=c['flops'], full_bill_exact=True,
            request_delta=c['requests']-b['requests'], residual_delta_s=c['residual_s']-b['residual_s'],
            server_cpu_ratio=c['server_cpu_s']/b['server_cpu_s']))
    official_pin = fixed(H2/'eval_332738.json','6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6')
    official = json.loads(checked(official_pin)); evaluation = official['evaluation']
    assert official['submission_id'] == 332738 and evaluation['submission_id'] == '332738'
    assert evaluation['progress']['mlps_completed'] == 100 and evaluation['progress']['mlps_failed'] == 0
    telemetry = [r['telemetry'] for r in evaluation['results']['per_mlp']]
    assert len(telemetry) == 100
    assert all(math.isfinite(t[k]) and t[k] >= 0 for t in telemetry for k in ('participant_predict_wall_s','flopscope_server_kernel_wall_s','residual_wall_time_s'))
    max_residual = max(t['residual_wall_time_s'] for t in telemetry)
    assert math.isclose(max_residual, .285450751, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(max(t['participant_predict_wall_s'] for t in telemetry),82.952904334,rel_tol=0,abs_tol=1e-9)
    audit = json.loads(checked(fixed(P2/'OFFICIAL_SOURCE_AUDIT.json','bce8f9b7ac64e61b7764d34aa764eaf219cff3de9fec5e4eb5fdbbd17bc84051')))
    assert audit['round']['residual_wall_time_limit_s'] == .4
    boundary_ratio = statistics.mean(r['server_cpu_ratio'] for r in comparisons)
    mismatch = abs(100*(boundary_ratio-native2_ratio))
    needs_control = mismatch > 5
    used_change = max(0.,native2_ratio-1,native8_ratio-1,boundary_ratio-1)
    request_delta = max(0.,statistics.mean(r['request_delta'] for r in comparisons))
    grid = [dict(native_fraction=fraction, request_cost_s=cost,
                 projected_worst_wall_s=max(t['participant_predict_wall_s']+fraction*used_change*t['flopscope_server_kernel_wall_s']+request_delta*cost for t in telemetry))
            for fraction in (.60,.78,.95) for cost in (.00030,.00041,.00050)]
    projected_wall = max(x['projected_worst_wall_s'] for x in grid)
    projected_residual = max_residual + 4*max(0.,*(r['residual_delta_s'] for r in comparisons))
    gates = dict(setup=setup['setup_wall_s']<=4.1, residual_projection=projected_residual<=.35,
                 runtime_projection=projected_wall<=105, cpu_agreement=not needs_control)
    result = dict(passed=all(gates.values()), receipt_checks_passed=True, needs_control=needs_control,
        stage='EARLY_BOUNDARY2_FEASIBILITY', candidate=candidate, parent=parent, admission=admission_pin,
        assessor=pin(Path(__file__)), native8=admission['native8_assessment'], gates=gates,
        extension_admission=extension_pin, run_protocol=run_protocol_pin, original_component_admitted=False,
        selection_disclosure=extension['selection_disclosure'], total_s32_screen_passed=True,
        boundary=current_census, launcher=current_launcher, guard=current_guard, setup=setup,
        comparisons=comparisons, official_evaluation=official_pin,
        residual=dict(official_s32_max_s=max_residual, multiplier=4, projected_s=projected_residual, gate_s=.35),
        runtime=dict(native2_cpu_ratio=native2_ratio,native8_cpu_ratio=native8_ratio,
            boundary_mean_cpu_ratio=boundary_ratio,native2_boundary_disagreement_pp=mismatch,
            disagreement_limit_pp=5,used_nonnegative_cpu_change=used_change,
            nonnegative_mean_request_delta=request_delta,projection_grid=grid,
            projected_worst_wall_s=projected_wall,gate_s=105),
        maximum_boundary_rss_bytes=g['max_rss_bytes'], evidence=list(EVIDENCE.values()),
        matched_control_required_if_needs_control=True, matched_control_performed=False,
        output_parity_claim=False, worker20_qualified=False, setup_stress_qualified=False,
        release_qualified=False, submitted=False, arrays_read=False, model_executed=False,
        interpretation='A passing early screen only permits considering the next admitted stage. This sensitivity projection is not measured official candidate runtime. A CPU mismatch needs a fresh source-bound S32 control and separate reviewed resolution; no historical exception is reused.')
    for record in list(EVIDENCE.values()):
        checked(record)
    return result


def main():
    assert len(sys.argv) == 1
    output = D/'boundary_assessment.json'
    assert not output.exists()
    try:
        result = assess()
    except Exception as exc:
        result = dict(passed=False,receipt_checks_passed=False,stage='EARLY_BOUNDARY2_FEASIBILITY',
            error=dict(type=type(exc).__name__,message=str(exc)),assessor=pin(Path(__file__)),
            release_qualified=False,submitted=False,arrays_read=False,model_executed=False)
        with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        raise
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','receipt_checks_passed','needs_control','gates')}))


if __name__ == '__main__':
    main()
