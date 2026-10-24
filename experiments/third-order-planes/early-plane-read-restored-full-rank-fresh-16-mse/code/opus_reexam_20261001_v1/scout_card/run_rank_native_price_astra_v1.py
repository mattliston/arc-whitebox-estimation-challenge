"""HELD finite native price controller; eight fixed evaluations, one granted slot.

Never reads fresh/OOS scientific results, changes sources, promotes, or submits.
Root must register PREREG_v1 and supply a source-pinned review plus slot grant.
Writes a PRICE_FOR_ROOT_FREEZE proposal; only root writes NATIVE_PRICE_v1.json.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

S = Path(__file__).resolve().parent
O = S.parent
R = O / 'lead/runs/astra_rank_redistribution_v1'
H = S / 'rank_native_price_held_astra_v1'
MANIFEST = H / 'native_price_manifest_astra_v2.json'
MANIFEST_SHA = 'a863b4454e448319f4c6b9b1bd32995772bea359abd8dd5507d74b0b783541d9'
RUN = S / 'rank_native_price_run_astra_v1'
PREREG = R / 'PREREG_v1.json'
GUARD = O / 'release/guard_footprint_v3.py'
CERT = O / 'scout_native/run_cert_v2.py'
GUARD_PY = '/Users/matt/anaconda3/bin/python'
NATIVE_PY = str(O.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
STATS = ('_ASTRA_COMPUTE_STATS', '_UBC_STATS', '_UBX_STATS', '_UBX_RC_STATS', '_UBX_STATE')
SEEDS = (7100136, 7100137)
ARMS = ('zero', 'fund64', 'transfer64_off', 'transfer64')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def evidence(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def admission():
    memory = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', memory).group(1))
    processes = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=10)
    heavy = sum(int(parts[0]) >= 1_048_576 and 'python' in parts[1].lower()
                for line in processes.splitlines() if len(parts := line.split(None, 1)) == 2)
    assert free >= 50 and heavy + 1 + 1 <= 10, (free, heavy, 'truth1+new1')
    return dict(utc=stamp(), free_pct=free, heavy_python_ge1GiB=heavy,
                truth_reserve=1, new_native_workers=1, cap=10)


def configuration(review_path):
    assert sha(MANIFEST) == MANIFEST_SHA
    manifest, prereg, review = read(MANIFEST), read(PREREG), read(review_path)
    assert prereg['status'] == 'root_registered_before_fresh_evaluations'
    assert prereg['joint_allocation_registered'] is True
    assert prereg['native_price_seeds'] == list(SEEDS)
    assert prereg['native_price_manifest_sha256'] == MANIFEST_SHA
    assert prereg['native_price_aggregation'] == 'max_pair_log_ratio'
    assert review['approved'] is True and review['reviewed_by'] == 'root'
    assert review['prereg_sha256'] == sha(PREREG)
    assert review['native_price_manifest_sha256'] == MANIFEST_SHA
    assert review['launcher_sha256'] == sha(__file__)
    assert review['baseline_reuse_approved'] is True
    assert review['guard_sha256'] == sha(GUARD) and review['cert_runner_sha256'] == sha(CERT)
    assert manifest['fixed_price_plan']['seeds'] == list(SEEDS)
    assert manifest['fixed_price_plan']['sequential_arms_per_seed'] == list(ARMS)
    assert sha(manifest['parent']['path']) == manifest['parent']['sha256']
    for spec in manifest['arms'].values():
        assert sha(spec['candidate']) == spec['candidate_sha256']
        assert sha(spec['fragment']) == spec['fragment_sha256']
    for item in manifest['evidence']:
        assert sha(item['path']) == item['sha256'], item['path']
    for row in manifest['reused_price_baselines']:
        for item in row['evidence']:
            assert sha(item['path']) == item['sha256'], item['path']
    return manifest


def result(cert_path, guard_path, spec, seed, reused=False):
    # Small local certificate arrays only; never estimator/model imports here.
    import numpy as np
    cert, guard = read(cert_path), read(guard_path)
    assert cert['sha256'] == spec['sha256'] == sha(spec['path'])
    assert Path(cert['candidate']).resolve() == Path(spec['path']).resolve()
    assert cert['seeds'] == [seed] and len(cert['rows']) == 1
    assert guard['completed'] and guard['whole_group_retired'] and guard['exit_code'] == 0
    assert guard['error'] is None and guard['remaining_rows'] == []
    assert 0 < guard['max_rss_bytes'] <= 7_500_000_000
    assert 0 <= guard['max_footprint_bytes'] <= 7_500_000_000
    assert str(Path(spec['path']).resolve()) in guard['command']
    if not reused:
        assert 'PB_ALL=1' in guard['command']
        assert all(name + '=1' in guard['command'] for name in THREADS)
    pred_path = Path(cert_path).with_name(Path(cert_path).stem + '_pred.npz')
    with np.load(pred_path, allow_pickle=False) as saved:
        assert saved.files == [str(seed)]
        prediction = saved[str(seed)].copy()
    row = cert['rows'][0]
    assert prediction.shape == (16, 1024) and prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    assert hashlib.sha256(prediction.tobytes()).hexdigest()[:16] == row['pred_sha']
    assert len(row['mse_layers']) == 16 and all(math.isfinite(x) and x > 0 for x in row['mse_layers'])
    assert row['mse_final'] == row['mse_layers'][15] and row['seed'] == seed
    assert isinstance(row['flops'], int) and row['flops'] > 0 and row['ops'] > 0
    assert row['stats']['_ASTRA_COMPUTE_STATS']['calls'] == 14
    assert row['stats']['_ASTRA_COMPUTE_STATS']['saved_copy_elements'] > 0
    if not reused:
        assert set(STATS) <= set(row['stats'])
        ubc, ubx = row['stats']['_UBC_STATS'], row['stats']['_UBX_STATE']
        assert ubc['retired'] is True and ubc['live_cache_bytes'] == ubc['retained_cache_bytes'] == 0
        assert 0 <= ubc['max_resident_bytes'] <= 512_000_000
        assert ubx['bytes'] == 0 and ubx['cache'] == {} and ubx['run'] is None
        for name in ('_UBX_STATS', '_UBX_RC_STATS'):
            assert all(isinstance(x, int) and x >= 0 for x in row['stats'][name].values())
    return dict(row=row, prediction=prediction, cert=Path(cert_path),
                evidence=[evidence(p) for p in (cert_path, guard_path, pred_path)])


def gates(arm, seed, results):
    current, baseline = results[arm], results['base']
    a, b = current['row'], baseline['row']
    if arm == 'zero':
        assert current['prediction'].tobytes() == baseline['prediction'].tobytes()
        assert a['mse_layers'] == b['mse_layers'] and a['mse_final'] == b['mse_final']
        assert (a['flops'], a['ops']) == (b['flops'], b['ops'])
        return dict(seed=seed, arm=arm, all16_bytes_mse_bill_requests_exact=True)
    assert current['prediction'][:7].tobytes() == baseline['prediction'][:7].tobytes()
    assert a['mse_layers'][:7] == b['mse_layers'][:7]
    assert current['prediction'][15].tobytes() != baseline['prediction'][15].tobytes()
    check = dict(seed=seed, arm=arm, unchanged_layers_0_to_6=True, final_changed=True,
                 copy_hook_calls=a['stats']['_ASTRA_COMPUTE_STATS']['calls'])
    if arm == 'transfer64':
        off = results['transfer64_off']
        assert current['prediction'].tobytes() == off['prediction'].tobytes()
        assert a['mse_layers'] == off['row']['mse_layers']
        assert a['flops'] < off['row']['flops']
        assert a['stats']['_UBC_STATS']['hits'] > 0 and a['stats']['_UBX_STATS']['hits'] > 0
        assert a['stats']['_ASTRA_COMPUTE_STATS'] == off['row']['stats']['_ASTRA_COMPUTE_STATS']
        check.update(cache_all16_bytes_mse_exact=True, cache_flop_saving=off['row']['flops']-a['flops'])
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-granted', action='store_true')
    parser.add_argument('--review', required=True, type=Path)
    args = parser.parse_args()
    assert args.root_granted, 'Root must separately grant one native slot.'
    assert args.review.resolve().parent == O / 'lead'
    # The metadata process and outer guard also get the explicit BLAS1 profile.
    for name in THREADS:
        os.environ[name] = '1'
    manifest = configuration(args.review)
    initial_gate = admission()
    RUN.mkdir(exist_ok=False)
    plan = [(arm, seed) for seed in SEEDS for arm in ARMS]
    write(RUN / 'launch_receipt_astra_v1.json', dict(utc=stamp(), plan=plan,
          manifest=evidence(MANIFEST), prereg=evidence(PREREG), review=evidence(args.review),
          launcher=evidence(__file__), guard=evidence(GUARD), cert_runner=evidence(CERT),
          initial_gate=initial_gate, slots=1, reads_no_fresh_or_oos_results=True))
    all_results, checks = {}, []
    try:
        for seed in SEEDS:
            base = next(x for x in manifest['reused_price_baselines'] if x['seed'] == seed)
            base_cert, base_guard = [Path(x['path']) for x in base['evidence'][:2]]
            results = {'base': result(base_cert, base_guard, manifest['parent'], seed, reused=True)}
            all_results[seed] = results
            for arm in ARMS:
                configuration(args.review)  # no silent source/protocol drift between evaluations
                gate = admission()
                spec = manifest['arms'][arm]
                cert = RUN / f'{arm}_{seed}_astra_v1.json'
                guard = RUN / f'{arm}_{seed}_astra_v1_guard.json'
                log = RUN / f'{arm}_{seed}_astra_v1.log'
                envargs = ['PB_ALL=1', 'STATS='+','.join(STATS)] + [name+'=1' for name in THREADS]
                command = ['/usr/bin/nice', '-n', '10', GUARD_PY, '-B', str(GUARD),
                           '--record', str(guard), '--seconds', '900', '--rss-backstop', '7500000000',
                           '/usr/bin/env', *envargs, NATIVE_PY, '-B', str(CERT),
                           spec['candidate'], str(seed), str(cert)]
                write(RUN / f'{arm}_{seed}_admission_astra_v1.json', dict(gate=gate, command=command))
                print(json.dumps(dict(event='native_price_start', utc=stamp(), arm=arm, seed=seed)), flush=True)
                with log.open('xb') as stream:
                    subprocess.run(command, cwd=O/'scout_native', env=os.environ.copy(),
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                results[arm] = result(cert, guard, dict(path=spec['candidate'], sha256=spec['candidate_sha256']), seed)
                check = gates(arm, seed, results)
                write(RUN / f'{arm}_{seed}_checks_astra_v1.json', check)
                checks.append(check)
                print(json.dumps(dict(event='native_price_done', utc=stamp(), arm=arm, seed=seed,
                                      flops=results[arm]['row']['flops'], ops=results[arm]['row']['ops'])), flush=True)
        candidate_sha = {'base':manifest['parent']['sha256']}
        candidate_sha.update({arm:manifest['arms'][arm]['candidate_sha256'] for arm in ('fund64','transfer64')})
        pairs, factors = {}, {}
        for arm in ('fund64', 'transfer64'):
            pairs[arm] = [dict(seed=seed, base_path=str(all_results[seed]['base']['cert']),
                              base_sha256=sha(all_results[seed]['base']['cert']),
                              arm_path=str(all_results[seed][arm]['cert']),
                              arm_sha256=sha(all_results[seed][arm]['cert'])) for seed in SEEDS]
            factor = max(all_results[seed][arm]['row']['flops']/all_results[seed]['base']['row']['flops'] for seed in SEEDS)
            factors[arm] = dict(max_pair_ratio=factor, max_pair_log_ratio=math.log(factor), bill_pct=100*(factor-1))
        write(RUN / 'PRICE_FOR_ROOT_FREEZE_astra_v1.json', dict(
            status='ready_for_root_freeze_before_fresh_summary', utc=stamp(),
            prereg_sha256=sha(PREREG), native_price_manifest_sha256=MANIFEST_SHA,
            aggregation='max_pair_log_ratio', candidate_source_sha256=candidate_sha, arms=pairs,
            measured_factors=factors, checks=checks,
            evidence=[item for seed in SEEDS for itemset in all_results[seed].values() for item in itemset['evidence']],
            no_fresh_or_oos_content_read=True, all_guards_retired=True, evaluations=8))
        print(json.dumps(dict(event='native_price_complete_slot_released', utc=stamp(), measured_factors=factors)), flush=True)
    except BaseException as error:
        write(RUN / 'STOPPED_astra_v1.json', dict(utc=stamp(), error=repr(error), completed_checks=checks,
                                                no_retry_or_next_evaluation_authorized=True))
        raise


if __name__ == '__main__':
    main()
