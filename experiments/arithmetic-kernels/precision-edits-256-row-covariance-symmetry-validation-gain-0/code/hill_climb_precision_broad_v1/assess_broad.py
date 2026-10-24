"""Root-only VALID40 receipt/array assessment; no estimator or truth execution.

Run NATIVE -B assess_broad.py under BLAS1, 120s, RSS<7.5GB and zero added swap.
Raw error is reported without a separate veto under the pinned new policy.
"""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import math
import os
import sys

D = Path(__file__).resolve().parent
H2 = D.parent
PARENT_STUDY = H2/'hill_climb_precision_symmetry_bundle_v1'
ADMISSION_SHA = 'a6e223a7cff1cd8acbc562d86380ddf44427690919afb33c7feae5d3b393b2b9'
HELPER_SHA = '530a750f819b6e49e0c4e6901ab56b7d1af7b6cce666d07a602926aa292bd4a5'
POLICY_SHA = '26e99f9a54d97a4d6fca11ac477056219f7acf351d5f7327df9b6ff7cf898fa6'
CANDIDATE_SHA = '1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
GATES = dict(adjusted_pct_at_most=-1, wins_min=28, ci95_upper_below=0,
             max_per_network_regression_pct=10)


def initialize():
    helper = PARENT_STUDY/'assess.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_precision_helpers', helper)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)  # Only pinned helpers; its main is not invoked.
    assert Path(sys.executable).absolute() == q.NATIVE and q.np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in q.THREADS)
    return q


def pinned_json(q, record):
    return json.loads(q.checked(record))


def block(q, b, child, candidate):
    i, ranks = b['block'], b['ranks']
    assert type(i) is int and i in (1, 2, 3, 4)
    assert len(ranks) == len(set(ranks)) == 10
    assert b['guard_seconds'] == 420 and b['rss_bytes'] == 7500000000 and b['added_swap_bytes'] == 0
    if child:
        tag = f'hill_climb_precision_broad_valid40_b{i}_v1'
        assert b['candidate_tag'] == tag
        folder = H2/tag
        manifest = q.load(folder/'manifest.json')
        source = candidate
        guard_path = D/f'broad_b{i}_guard.json'
        command = [str(q.NATIVE), '-B', str(q.DRIVER), source['path'], ','.join(map(str, ranks)), tag]
        seconds = 420
    else:
        tag = f'hill_climb_star32_broad_v1_bank_valid40_b{i}_v1'
        folder = H2/tag
        assert b['baseline_manifest'] == q.pin(folder/'manifest.json')
        assert b['baseline_rows'] == q.pin(folder/'rows.jsonl')
        manifest = pinned_json(q, b['baseline_manifest'])
        source = q.pin(H2/'hill_climb_star_projection32_v1/candidate.py')
        assert source['sha256'] == S32_SHA
        guard_path = Path(b['baseline_guard']['path'])
        q.checked(b['baseline_guard'])
        command = [str(q.NATIVE), str(H2/'native_eval.py'), source['path'], ','.join(map(str, ranks)), tag]
        seconds = 360  # Original pinned S32 block guard; candidate allowance is 420s.
    assert manifest['candidate'] == source['path'] and manifest['sha256'] == source['sha256']
    q.checked(source)
    assert manifest['ranks'] == ranks and manifest['runtime'] == str(q.NATIVE)
    assert manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == dict.fromkeys(q.THREADS, '1')
    guard = q.guard(guard_path, seconds, command)
    setup = manifest['setup']
    assert math.isfinite(setup['setup_s']) and 0 <= setup['setup_s'] <= 4.1
    bound, fused = (660, 636) if child else (624, 600)
    assert [r[1] for r in setup['pb_log'] if r[0] == 'bound'] == [bound]
    assert [r[1] for r in setup['pb_log'] if r[0] == 'lvfuse'] == [fused]
    rows = [json.loads(line) for line in q.checked(q.pin(folder/'rows.jsonl')).decode().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    predictions = []
    for row in rows:
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ('mse', 'adjusted', 'util', 'cpu_s', 'wall_s'))
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert row['util'] == row['flops']/2**41
        assert math.isclose(row['adjusted'], row['mse']*max(.1, row['util']), rel_tol=1e-14, abs_tol=0)
        prediction = q.pin(folder/f'pred_{row["rank"]}.npy')
        array = q.np.load(io.BytesIO(q.checked(prediction)), allow_pickle=False)
        assert array.shape == (16, 1024) and array.dtype == q.np.dtype('float64')
        assert q.np.isfinite(array).all()
        del array
        predictions.append(prediction)
        if child:
            ubc = row['ubc']
            assert ubc['retired'] is True and ubc['retained_cache_bytes'] == ubc['live_cache_bytes'] == 0
            assert 0 <= ubc['max_resident_bytes'] <= 512000000
    evidence = dict(manifest=q.pin(folder/'manifest.json'), rows=q.pin(folder/'rows.jsonl'),
                    guard=q.pin(guard_path), predictions=predictions)
    return manifest, rows, evidence, guard['max_rss_bytes']


def assess(q):
    admission_pin = q.pin(D/'admission.json')
    assert admission_pin['sha256'] == ADMISSION_SHA
    admission = pinned_json(q, admission_pin)
    assert admission['score_gates'] == GATES and admission['no_raw_error_veto'] is True
    assert admission['no_candidate_broad_outcomes_seen'] is True and admission['serving_admitted'] is False
    assert admission['policy']['sha256'] == POLICY_SHA
    policy = pinned_json(q, admission['policy'])
    assert policy['bank_gates'] == dict(adjusted_pct_at_most=-1, ci95_upper_below=0,
        wins_min_of40=28, max_per_network_regression_pct=10, serving_required=True)
    assert policy['resource_policy'] == dict(serial_root_numerics=True, rss_bytes=7500000000,
                                            added_swap_bytes=0, blas_threads=1)
    assert admission['candidate']['sha256'] == CANDIDATE_SHA
    candidate = q.build_check()
    assert candidate == admission['candidate']
    assert admission['driver'] == q.pin(q.DRIVER)
    q.checked(admission['driver'])
    training = pinned_json(q, admission['training'])
    assert training['passed_integrity'] is True and training['admitted'] is False and training['n'] == 8
    assert training['candidate'] == candidate
    assert training['gates'] == dict(adjusted_gain=False, wins=True, raw_regression=True, ci_upper_negative=True)
    for record in training['evidence']:
        q.checked(record)
    q.checked(training['assessor'])
    training_guard = PARENT_STUDY/'native8_extension_assess_guard.json'
    q.guard(training_guard, 120, [str(q.NATIVE), '-B', training['assessor']['path'], '8'])
    base_rows, new_rows, base_manifests, new_manifests, evidence, peaks = [], [], [], [], [], []
    assert len(admission['blocks']) == 4
    for i, b in enumerate(admission['blocks'], 1):
        assert b['block'] == i
        bm, br, be, bp = block(q, b, False, candidate)
        cm, cr, ce, cp = block(q, b, True, candidate)
        base_rows.extend(br); new_rows.extend(cr)
        base_manifests.append(bm); new_manifests.append(cm)
        evidence.append(dict(baseline=be, candidate=ce)); peaks.extend((bp, cp))
    base = {r['rank']: r for r in base_rows}; new = {r['rank']: r for r in new_rows}
    ranks = sorted(base)
    assert len(base_rows) == len(new_rows) == len(ranks) == 40 and ranks == sorted(new)
    assert not set(ranks).intersection(training['ranks'])
    keys = ('mse', 'adjusted', 'util', 'cpu_s')
    bb = q.np.array([[base[r][k] for k in keys] for r in ranks])
    cc = q.np.array([[new[r][k] for k in keys] for r in ranks])
    indices = q.np.random.default_rng(20260926).integers(0, 40, size=(50000, 40))
    ratios = cc[indices, :2].mean(1)/bb[indices, :2].mean(1)-1
    ci = q.np.quantile(ratios, [.025, .975], axis=0)*100
    per_network = [dict(rank=r, raw_pct=100*(new[r]['mse']/base[r]['mse']-1),
        adjusted_pct=100*(new[r]['adjusted']/base[r]['adjusted']-1),
        util_pct=100*(new[r]['util']/base[r]['util']-1)) for r in ranks]
    metrics = dict(n=40, ranks=ranks,
        raw_pct=float(100*(cc[:, 0].mean()/bb[:, 0].mean()-1)), raw_ci95_pct=ci[:, 0].tolist(),
        adjusted_pct=float(100*(cc[:, 1].mean()/bb[:, 1].mean()-1)), adjusted_ci95_pct=ci[:, 1].tolist(),
        raw_wins=int((cc[:, 0] < bb[:, 0]).sum()), adjusted_wins=int((cc[:, 1] < bb[:, 1]).sum()),
        base_mean_adjusted=float(bb[:, 1].mean()), candidate_mean_adjusted=float(cc[:, 1].mean()),
        mean_util_pct=float(100*(cc[:, 2].mean()/bb[:, 2].mean()-1)),
        bill_pct=100*(sum(new[r]['flops'] for r in ranks)/sum(base[r]['flops'] for r in ranks)-1),
        mean_cpu_pct=float(100*(cc[:, 3].mean()/bb[:, 3].mean()-1)),
        base_sources=base_manifests, candidate_sources=new_manifests, per_network=per_network)
    maximum = max(r['adjusted_pct'] for r in per_network)
    gates = dict(adjusted=metrics['adjusted_pct'] <= -1, wins=metrics['adjusted_wins'] >= 28,
                 ci=metrics['adjusted_ci95_pct'][1] < 0, max_regression=maximum <= 10)
    own_pin = q.pin(Path(__file__))
    q.checked(own_pin)
    result = dict(passed=all(gates.values()), receipt_checks_passed=True, validation_score_passed=all(gates.values()),
        stage='VALID40', candidate=candidate, parent=q.pin(H2/'hill_climb_star_projection32_v1/candidate.py'),
        baseline_submission=332738, admission=admission_pin, policy=admission['policy'],
        training=admission['training'], training_guard=q.pin(training_guard), original_training_admitted=False,
        selection_disclosure=admission['selection_disclosure'], assessor=own_pin,
        gates=gates, metrics=metrics, bootstrap=dict(resamples=50000, seed=20260926),
        max_per_network_adjusted_regression_pct=maximum, maximum_guard_rss_bytes=max(peaks),
        validation_guards=[e['candidate']['guard'] for e in evidence], blocks=evidence,
        evidence=list(q.EVIDENCE.values()), reused_validation_not_pristine=True,
        raw_error_veto=False, prefix_parity_required=False,
        targets_read=False, model_executed=False, worker_qualified=False, release_qualified=False, submitted=False,
        next_step='Root may admit full source-bound serving only if score gates pass; no automatic release.')
    for record in list(q.EVIDENCE.values()):
        q.checked(record)
    return result


def main():
    assert len(sys.argv) == 1
    target = D/'broad_assessment.json'
    assert not target.exists()
    q = initialize()
    try:
        result = assess(q)
    except Exception as exc:
        result = dict(passed=False, receipt_checks_passed=False, stage='VALID40',
            error=dict(type=type(exc).__name__, message=str(exc)), assessor=q.pin(Path(__file__)),
            worker_qualified=False, release_qualified=False, submitted=False)
        with target.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        raise
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'receipt_checks_passed', 'gates',
                     'max_per_network_adjusted_regression_pct')}))


if __name__ == '__main__':
    main()
