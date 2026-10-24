"""Stdlib plans/preflight for the fixed gn-perp candidate owned by root."""
from pathlib import Path
import hashlib, json, sys

D = Path(__file__).resolve().parent
H = D.parent
G = H/'hill_climb_rrpi_native_joint3_recovery_v1'
FIT = [35, 100, 36, 45]
CHECK = [52, 58, 64, 70, 76, 80, 84, 88]
BEFORE = '                C = C + _e2_H(_gn_dcov(l, mu, Cpre, K21, (_s19_k4u if _S19_GN_TORCH else k4), aux))'
AFTER = '                C = C + _e2_H(_gn_perp_host(l, m, _gn_dcov(l, mu, Cpre, K21, (_s19_k4u if _S19_GN_TORCH else k4), aux)))'
PARENT_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
PRIOR = {
    'fit4': ('43fc8ac2cd8a82835366b4144bdedb5c91780d7758f369a10353c7fb226bf10d', '57d52b3155bb09ab7add2952f2be234109e3e7b025b6b2616952307ed4e44338'),
    'check8': ('a00100251c915f57062a1fb84fab945d42b75440c91d2e376f2aa6cf6dc2201f', '9eeb31459174aa4ef6b5d7b8b3980f3954ac128224a3938e0c9f2619e952ad1a'),
}
GATES = dict(raw_pct_strictly_below=0., adjusted_pct_strictly_below=0.,
             all16_finite=True, layer0_exact=True, wins_and_ci_descriptive_only=True)


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def checked(record):
    assert pin(record['path']) == record
    return Path(record['path']).read_bytes()


def once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def source():
    parent = pin(H/'hill_climb_pair_rr_inverse_v1/candidate.py')
    assert parent['sha256'] == PARENT_SHA
    original = checked(parent).decode(); candidate = pin(D/'candidate.py')
    proposed = checked(candidate).decode()
    fragment = pin(D/'fragment.py')
    assert fragment['sha256'] == '7a9bb0f2f8ba3c6e9fa6c0a54f398cdd48b04511d760cf3c9e7664aa72a7345e'
    suffix = '\n\n' + checked(fragment).decode()
    assert original.count(BEFORE) == 1 and AFTER not in original
    assert proposed == original.replace(BEFORE, AFTER, 1) + suffix
    assert proposed.endswith(suffix) and proposed[:-len(suffix)].replace(AFTER, BEFORE, 1) == original
    compile(proposed, candidate['path'], 'exec')
    return parent, candidate, dict(before=BEFORE, after=AFTER, occurrences=1, appended_fragment=fragment, whole_source_inverse_exact=True)


def common():
    controls = {}; data = {}; prior = {}; plans = {}
    for stage, hashes in PRIOR.items():
        pp = pin(G/('native_'+stage+'_plan.json')); ap = pin(G/(stage+'_assessment.json'))
        assert pp['sha256'] == hashes[0] and ap['sha256'] == hashes[1]
        old = json.loads(checked(pp)); receipt = json.loads(checked(ap))
        assert receipt['passed_integrity'] and old['arms']['base']['source']['sha256'] == PARENT_SHA
        controls[stage] = dict(arm=old['arms']['base'], artifacts=receipt['artifacts']['base'])
        data[stage] = old['data']; prior[stage] = dict(plan=pp, assessment=ap); plans[stage] = old
    a, b = plans['fit4'], plans['check8']
    for key in ('runtime', 'light', 'driver', 'guard_source', 'setup_counts'):
        assert a[key] == b[key]
    helper = pin(H/'hill_climb_uniform_gnn_gain_v1/fit.py')
    engine = pin(H/'native_screen_engine_v1/screen.py')
    assert helper['sha256'] == '60f91d378fe4c7c1b12b12aa3a90b41acd73cf0f3714aba91eec85e13e08fd36'
    assert engine['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    for key in ('driver', 'guard_source'): checked(a[key])
    return dict(controls=controls, data=data, prior=prior, helper=helper, engine=engine,
                **{k:a[k] for k in ('runtime', 'light', 'driver', 'guard_source', 'setup_counts')})


def fit4_prerequisite(c):
    fp = pin(D/'fit4_assessment.json'); f = json.loads(checked(fp))
    assert f['passed_integrity'] and f['admitted'] and f['check8_permitted']
    assert f['candidate'] == c['candidate'] and f['parent'] == c['parent']
    assert f['plan'] == c['fit4_plan'] and f['assessor'] == c['assessor']
    assert f['metrics']['raw_pct'] < 0 and f['metrics']['adjusted_pct'] < 0
    g = json.loads((D/'fit4_assess_guard.json').read_text())
    assert g['command'] == [c['runtime'], '-B', c['assessor']['path'], f['plan']['path'], f['plan']['sha256']]
    assert g['completed'] and g['exit_code'] == 0 and g['error'] is None and g['whole_group_retired']
    assert not g['remaining_rows'] and g['max_swap_growth_bytes'] == 0
    assert g['rss_trigger_bytes'] == 7500000000 and 0 < g['max_rss_bytes'] < 7500000000
    assert 0 < g['elapsed_s'] <= 120 and g['samples'] > 0
    return [fp, pin(D/'fit4_assess_guard.json')]


def configuration(path, digest):
    pp = pin(path); assert pp['sha256'] == digest
    c = json.loads(checked(pp)); parent, candidate, inverse = source()
    assert c['schema'] == 'fixed-gn-perp-native-v1' and c['stage'] in ('fit4', 'check8')
    assert c['ranks'] == (FIT if c['stage'] == 'fit4' else CHECK) and c['gates'] == GATES
    assert c['parent'] == parent and c['candidate'] == candidate and c['source_inverse'] == inverse
    assert c['builder'] == pin(__file__) and c['assessor'] == pin(D/'assess.py')
    assert c['one_fixed_change_only'] and not c['fit_or_sweep'] and c['no_baseline_rerun']
    assert c['bank_gain_pct_min'] == 1 and c['full40_own_resources_serving_contract_required']
    for key, value in common().items(): assert c[key] == value
    assert c['arm']['source'] == candidate and c['arm']['ranks'] == c['ranks']
    prerequisites = fit4_prerequisite(c) if c['stage'] == 'check8' else []
    return pp, c, prerequisites


def main():
    assert len(sys.argv) == 1
    parent, candidate, inverse = source(); shared = common()
    for stage, ranks, seconds in [('fit4', FIT, 150), ('check8', CHECK, 300)]:
        tag = D.name+'_'+stage+'_v1'; record = str(D/('native_'+stage+'_guard.json'))
        command = [shared['runtime'], '-B', shared['driver']['path'], candidate['path'], ','.join(map(str, ranks)), tag]
        arm = dict(source=candidate, tag=tag, output=str(H/tag), ranks=ranks, driver=shared['driver'], guard=record,
            limits=dict(seconds=seconds, rss_bytes=7500000000, added_swap_bytes=0), command=command,
            guard_command=[shared['light'], '-B', shared['guard_source']['path'], '--record', record,
                           '--seconds', str(seconds), '--rss-limit', '7500000000', *command], log=str(D/('native_'+stage+'.log')))
        c = dict(schema='fixed-gn-perp-native-v1', stage=stage, ranks=ranks, gates=GATES, parent=parent,
            candidate=candidate, source_inverse=inverse, builder=pin(__file__), assessor=pin(D/'assess.py'), arm=arm,
            one_fixed_change_only=True, fit_or_sweep=False, no_baseline_rerun=True, adaptive_train_not_pristine=True,
            bank_gain_pct_min=1., full40_own_resources_serving_contract_required=True, **shared)
        if stage == 'check8': c['fit4_plan'] = pin(D/'native_fit4_plan.json')
        path = D/('native_'+stage+'_plan.json'); once(path, c)
        assessment = [c['runtime'], '-B', c['assessor']['path'], str(path), pin(path)['sha256']]
        once(D/('commands_'+stage+'.json'), dict(plan=pin(path), model=arm['guard_command'], model_log=arm['log'],
            assessment=[c['light'], '-B', c['guard_source']['path'], '--record', str(D/(stage+'_assess_guard.json')),
                        '--seconds', '120', '--rss-limit', '7500000000', *assessment], assessment_log=str(D/(stage+'_assess.log')),
            thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'), '1'),
            launch_prerequisite='Root calls prepare.configuration(plan_path, plan_sha) before model launch; CHECK8 requires actual FIT4 gain.'))
    print(json.dumps({n:pin(D/n) for n in ('native_fit4_plan.json','native_check8_plan.json','commands_fit4.json','commands_check8.json')}))


if __name__ == '__main__': main()
