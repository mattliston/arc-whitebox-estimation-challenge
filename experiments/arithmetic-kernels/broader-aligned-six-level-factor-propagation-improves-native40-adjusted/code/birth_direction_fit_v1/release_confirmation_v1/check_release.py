"""Exact-source confirmation-bank preparation/runtime qualification.

Default/help is inert. --prepare creates only a sealed launch contract;
--qualify checks an already-completed persistent NumPy-free series, then
reads four existing independent4M truths for numerical fidelity. No model
import, estimator run, fit, package or upload occurs in this checker.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BD, NEW = HERE.parent, HERE.parent.parent
P2 = NEW.parent
ENV = P2 / 'data_gate/.venv'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SOURCE = BD / 'build_terminal_v1/estimator.py'
CANDIDATE = 'birth_direction_fit_v1/build_terminal_v1/estimator.py'
SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
SCORE_SHA = '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b'
NATIVE_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
TAG = 'birth_direction_terminal_confirmation_openblas_series_v1'
RANKS = (35, 100, 500, 3016500, 35)
INIT, BUDGET = 1383913786, 2**41
LAUNCH_ENV = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
    MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
PINS = {
    SOURCE: SHA,
    BD / 'build_terminal_v1/build.json': '4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    BD / 'training_v1/terminal_theta.json': 'de0eb2c2b3dbda9cef1e6d31146755f18f400aaa47f793460017fd9040753337',
    BD / 'panel_v2.py': 'e5f16497f4c8c825f9a203dac3e8c0c3ceac218731605c7c818ae5e8a237268e',
    BD / 'panel_v2/score/result.json': SCORE_SHA,
    BD / 'panel_v2/terminal/result.json': NATIVE_SHA,
    NEW / 'run_remote_series_v1.py': '00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1',
    NEW / 'check_remote_series_v1.py': 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return sha(path)


def unchanged(pins):
    assert all(sha(p) == s for p, s in pins.items()), 'Changed qualification artifact'


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    assert expected is None or value == expected, str(path)
    assert str(path) not in pins or pins[str(path)] == value
    pins[str(path)] = value
    return value


def command():
    return [str(ENV / 'bin/python'), '-u', str(NEW / 'run_remote_series_v1.py'),
            '--candidate', CANDIDATE, '--tag', TAG]


def admission():
    pins = {}
    for path, digest in PINS.items():
        pin(pins, path, digest)
    for path in (Path(__file__).resolve(), HERE / 'PROTOCOL.md'):
        pin(pins, path)
    score, native = read(BD / 'panel_v2/score/result.json'), read(BD / 'panel_v2/terminal/result.json')
    for role, record, digest in (('score', score, SCORE_SHA), ('terminal', native, NATIVE_SHA)):
        cp = BD / 'panel_v2' / role / 'completion.json'
        pin(pins, cp)
        complete = read(cp)
        assert record['passed'] and record['completed'] and record['hashes_unchanged']
        assert complete['passed'] and complete['result_sha256'] == digest
        assert record['schema'] == 'birth_direction_terminal_independent14_v2'
        assert record['candidate_sha256'] == SHA
    assert score['qualification_passed'] and score['retain_local_component'] and score['no_refit']
    assert score['fresh_official_input_complete_bills_only'] and not score['archive_bills_used_for_scoring']
    assert score['acquisition_sha256']['terminal'] == NATIVE_SHA
    assert score['group_strict_gain_gate']['public8'] and score['group_strict_gain_gate']['virgin6']
    assert score['summaries']['candidate']['combined14']['mean_adjusted_noise_corrected'] == 4.4837991190464774e-9
    assert native['initialization_flops'] == INIT and native['initialization_once'] and native['repeat_excluded']
    assert not native['truth_files_accessed'] and not native['instrumented']
    assert native['input_representation'] == 'owned FNP float32 copies, official loader convention'
    assert not native['input_preparation_in_participant_bill'] and not native['candidate_state_reset']
    # The entire successful panel qualification stays a frozen trusted receipt.
    # Recheck the actual15-call output/bill artifacts, not rerun training/targets.
    for path, digest in native['artifacts'].items():
        pin(pins, path, digest)
    assert len(native['trials']) == 15 and len(native['artifacts']) == 75
    assert native['trials'][0]['output_bytes_sha256'] == native['trials'][-1]['output_bytes_sha256']
    assert native['trials'][0]['flops'] - native['trials'][-1]['flops'] == INIT
    man = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    fixtures = {}
    for rank in set(RANKS):
        if rank < 1000:
            assert man[rank]['role_rank'] == rank
            fixture = STORE / Path(man[rank]['path']).parent / 'weights.npy'
        else:
            fixture = P2 / 'public_g19_attribution_v1' / f'fresh_{rank}_weights.npy'
        expected = native['fixture_bindings'][f'weights_{rank}']
        assert Path(expected['path']).resolve() == fixture.resolve()
        pin(pins, fixture, expected['sha256'])
        fixtures[str(rank)] = dict(path=str(fixture.resolve()), sha256=expected['sha256'])
    return pins, fixtures, native, score


def prepare():
    pins, fixtures, _, _ = admission()
    paths = [NEW / (TAG + suffix) for suffix in ('.json', '_resources.json', '_server.log')]
    paths += [NEW / f'{TAG}_{i}_r{rank}.f32' for i, rank in enumerate(RANKS)]
    assert not any(p.exists() for p in paths) and not (HERE / 'launch.json').exists()
    unchanged(pins)
    launch = dict(passed=True, schema='birth_direction_confirmation_launch_v1', utc=utc(),
        source_bindings=pins, fixture_bindings=fixtures, candidate_sha256=SHA,
        candidate=CANDIDATE, tag=TAG, ranks=list(RANKS), env=LAUNCH_ENV,
        command=command(), cwd=str(NEW), native_panel_score_sha256=SCORE_SHA,
        native_panel_result_sha256=NATIVE_SHA, user_explicit_confirmation_bank_authorized=True,
        estimator_executed=False, truth_accessed=False, uploaded=False)
    digest = write(HERE / 'launch.json', launch)
    write(HERE / 'prepare_completion.json', dict(passed=True, launch_sha256=digest, utc=utc()))
    print(json.dumps(dict(prepared=True, launch_sha256=digest, command=command(), cwd=str(NEW))), flush=True)


def qualify(launch_digest):
    assert sha(HERE / 'launch.json') == launch_digest
    launch = read(HERE / 'launch.json')
    assert read(HERE / 'prepare_completion.json')['launch_sha256'] == launch_digest
    pins, fixtures, native, panel = admission()
    assert launch['source_bindings'] == pins and launch['fixture_bindings'] == fixtures
    assert launch['command'] == command() and launch['env'] == LAUNCH_ENV and launch['ranks'] == list(RANKS)
    assert launch['candidate_sha256'] == SHA and launch['candidate'] == CANDIDATE and launch['tag'] == TAG
    pin(pins, HERE / 'launch.json', launch_digest)
    pin(pins, HERE / 'prepare_completion.json')
    result = dict(passed=False, completed=False, candidate_sha256=SHA,
        native_panel_score_sha256=SCORE_SHA, native_panel_result_sha256=NATIVE_SHA,
        source_bindings=pins, artifacts={}, checks=[], truth_bindings={},
        truth_accessed=False, official_score=False, uploaded=False, started_utc=utc())
    write(HERE / 'runtime_started.json', result)
    start = time.monotonic()
    try:
        series_path, resource_path = NEW / (TAG + '.json'), NEW / (TAG + '_resources.json')
        for path in (series_path, resource_path, NEW / (TAG + '_server.log')):
            pin(result['artifacts'], path)
            assert path.stat().st_mtime_ns >= (HERE / 'launch.json').stat().st_mtime_ns
        series, usage = read(series_path), read(resource_path)
        assert series['status'] == 'completed' and series['source_sha256'] == SHA
        assert series['candidate'] == usage['candidate'] == CANDIDATE
        assert series['repeated_output_byte_identical'] and len(series['records']) == 5
        samples = usage['samples']
        assert samples and all(len(s) == 2 and s[0] > 0 and type(s[1]) is int and s[1] >= 0 for s in samples)
        assert 0 < usage['peak_combined_rss_bytes'] == max(s[1] for s in samples) < 8e9
        import numpy as np
        remote_arrays, native_arrays = {}, {}
        for i, (rank, row) in enumerate(zip(RANKS, series['records'])):
            assert row['index'] == i and row['rank'] == rank
            assert row['output_shape'] == [16, 1024] and row['output_dtype'] == 'float32'
            f, timing = row['flops'], row['timing']
            assert type(f) is int and 0 < f <= BUDGET and row['util'] == f / BUDGET
            assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in timing.values())
            assert 0 < timing['wall_time_s'] < 60 and timing['residual_wall_time_s'] < .4
            path = NEW / f'{TAG}_{i}_r{rank}.f32'
            pin(result['artifacts'], path, row['output_sha256'])
            assert path.stat().st_mtime_ns >= (HERE / 'launch.json').stat().st_mtime_ns
            data = path.read_bytes()
            assert len(data) == 16 * 1024 * 4
            arr = np.frombuffer(data, dtype=np.float32).reshape(16, 1024)
            assert np.isfinite(arr).all() and (arr >= 0).all()
            trial = next(t for t in native['trials'] if t['rank'] == rank)
            prior = np.load(trial['output'], allow_pickle=False)
            assert prior.shape == arr.shape and prior.dtype == arr.dtype
            assert hashlib.sha256(prior.tobytes()).hexdigest() == trial['output_bytes_sha256']
            difference = float(np.max(np.abs(arr.astype(float) - prior.astype(float))))
            assert difference <= 1e-5
            if i < 4:
                remote_arrays[rank], native_arrays[rank] = arr, prior
            else:
                assert data == remote_arrays[35].tobytes()
            result['checks'].append(dict(call=i, rank=rank, flops=f, multiplier=max(.1, f / BUDGET),
                wall_s=timing['wall_time_s'], residual_s=timing['residual_wall_time_s'],
                output_sha256=row['output_sha256'], native_max_abs=difference))
        assert series['records'][0]['flops'] - series['records'][-1]['flops'] == INIT
        unchanged(pins); unchanged(result['artifacts'])
        seal = dict(utc=utc(), source_bindings=pins, artifacts=result['artifacts'], checks=result['checks'],
                    truth_accessed=False, all5_complete=True, repeat_excluded=True)
        result['outputs_and_bills_sealed_before_truth_sha256'] = write(HERE / 'outputs_and_bills_sealed_before_truth.json', seal)
        pin(result['artifacts'], HERE / 'outputs_and_bills_sealed_before_truth.json')
        result['per_net'] = []
        for i, rank in enumerate(RANKS[:4]):
            binding = panel['truth_bindings'][str(rank)]
            pin(result['truth_bindings'], binding['path'], binding['sha256'])
            result['truth_accessed'] = True
            with np.load(binding['path'], allow_pickle=False) as data:
                mean, se = data['mc_post'][15].astype(float), data['mc_se'][15].astype(float)
            assert mean.shape == se.shape == (1024,) and np.isfinite(mean).all() and np.isfinite(se).all() and (se >= 0).all()
            noise = float(np.mean(se * se))
            nraw = float(np.mean((native_arrays[rank][15].astype(float) - mean)**2))
            raw = float(np.mean((remote_arrays[rank][15].astype(float) - mean)**2))
            saved = next(t for t in panel['per_role']['candidate'] if t['rank'] == rank)
            assert nraw == saved['raw_observed'] and noise == saved['mc_se_squared_mean']
            assert abs(raw / nraw - 1) <= .01
            f = series['records'][i]['flops']
            result['per_net'].append(dict(rank=rank, raw_observed=raw, raw_noise_corrected=raw-noise,
                native_raw_observed=nraw, remote_native_mse_ratio=raw/nraw, mc_se_squared_mean=noise,
                flops=f, multiplier=max(.1, f/BUDGET), adjusted_noise_corrected=(raw-noise)*max(.1, f/BUDGET)))
        unchanged(pins); unchanged(result['artifacts']); unchanged(result['truth_bindings'])
        result.update(passed=True, completed=True, hashes_unchanged=True, runtime_gates_passed=True,
            native_fidelity_passed=True, all5_complete=True, initialization_flops=INIT,
            peak_combined_rss_bytes=usage['peak_combined_rss_bytes'], repeat_excluded=True,
            four_net_remote_diagnostic_mean_adjusted=sum(r['adjusted_noise_corrected'] for r in result['per_net'])/4,
            local_panel_nc_adjusted=4.4837991190464774e-9,
            scope='macOS persistent NumPy-free client/OpenBLAS diagnostic; not Linux cgroup or official grade')
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    result.update(wall_seconds=time.monotonic()-start, completed_utc=utc(),
                  checker_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    digest = write(HERE / 'runtime_result.json', result)
    write(HERE / 'runtime_completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    print(json.dumps({k:result.get(k) for k in ('passed','runtime_gates_passed','native_fidelity_passed','peak_combined_rss_bytes','error')}), flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--qualify', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--launch-sha256')
    args = parser.parse_args()
    if not (args.prepare or args.qualify):
        parser.print_help(); return 0
    assert sha(__file__) == args.source_sha256
    assert Path(sys.prefix).resolve() == ENV.resolve() and os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == v for k, v in LAUNCH_ENV.items())
    assert {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench')} == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0')
    def timeout(_signum, _frame):
        raise TimeoutError('Release metadata/fidelity check300s watchdog')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(300)
    try:
        return prepare() if args.prepare else qualify(args.launch_sha256)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
