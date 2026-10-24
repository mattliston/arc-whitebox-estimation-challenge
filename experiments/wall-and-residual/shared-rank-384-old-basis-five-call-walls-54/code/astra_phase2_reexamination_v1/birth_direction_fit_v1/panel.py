"""Inert terminal-birth-direction independent-truth panel: --acquire / --score.

Both modes require the same --terminal-score-sha256. Only a completed,
integrity-passing, advancing terminal40 judge admits this fixed source.
Acquire one fresh unwrapped estimator on public8 + virgin6 + repeat35;
score separately against the archived current409f outputs and own bills.
The reused independent4M truths are not previously unseen model holdouts:
35/100/200 were fit nets and150 was VALID. No fitting or source selection.
Retain only strict mean per-net NC-adjusted gains on BOTH groups. Native
residual is report-only; local milestones are NOT official/release claims.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
ROOT = HERE / 'panel_v1'
CANDIDATE = HERE / 'build_terminal_v1/estimator.py'
JUDGE = HERE / 'terminal_score_v1'
ARCHIVE = NEW / 'unused_materializations_v1/native'
BASE_SCORE = NEW / 'unused_materializations_v1/score/result.json'
TRUTH_INDEX = NEW / 'response_rows_panel_v1/score/result.json'
BANK = NEW / 'unused_materializations_v1/estimator.py'
BANK_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
BASE_ACQ_SHA = 'a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951'
BASE_SCORE_SHA = '0798a699e54f985ac6dff2f278860a2110aea5047ea3c3920294a768b3542d4d'
TRUTH_INDEX_SHA = 'a31981681305de1828c48d8b6e8830c87ed979789c7d600b949217814f479cdd'
HELPER_SHA = '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304'
PANEL_SHA = '6fcf38e55173a6053c51d536dc9791a0c925a84cd4164d39d159120c0d33acc6'
FIXED = {
    HERE / 'judge_terminal.py': '4f4ccb45d75f52c36530cc1266847334c84014ea60eb25d933d03aa933420c0b',
    HERE / 'native.py': '7718591dd4324956fdcad52721009e7a9f7fb01c6969710b622376633ffbe075',
    HERE / 'PROTOCOL.md': '9638ccee8e75e830f205b5cc8304d7333c7b202ae5441db3a71f610645f94212',
    HERE / 'NATIVE_SCOPE_CLARIFICATION.md': 'b43aa00adcc30f434f5e301abcd490deb220020096ea47daebcf908f420bd333',
    NEW / 'check_response_rows_adjusted_v1.py': HELPER_SHA,
    NEW / 'check_response_rows_panel_v1.py': PANEL_SHA,
    BANK: BANK_SHA,
    ARCHIVE / 'result.json': BASE_ACQ_SHA,
    BASE_SCORE: BASE_SCORE_SHA,
    TRUTH_INDEX: TRUTH_INDEX_SHA,
}
VERSIONS = dict(flopscope='0.12.0', whestbench='0.16.0', numpy='2.4.6', threadpoolctl='3.6.0')
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
TIMINGS = ('context_wall_s', 'meter_wall_s', 'backend_s', 'overhead_s', 'native_residual_s')
SCHEMA = 'birth_direction_terminal_independent14_v1'
RSS_LIMIT = 8_000_000_000


def load(path, name, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen module changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    h = load(NEW / 'check_response_rows_adjusted_v1.py', 'bd_panel_utilities', HELPER_SHA)
    p = load(NEW / 'check_response_rows_panel_v1.py', 'bd_panel_fixtures', PANEL_SHA)
    return h, p


def runtime(h):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve() and
              'anaconda3' not in sys.executable and 'anaconda3' not in sys.prefix, 'OFFPY only')
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and
              all(os.environ.get(k) == '1' for k in BLAS) and
              os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'nice10 / BLAS1 / no bytecode')
    h.require({n: importlib.metadata.version(n) for n in VERSIONS} == VERSIONS, 'Native versions')


def prerequisites(h, p, terminal_sha):
    """Only source/receipt/weight/output hashes; never any truth-array bytes."""
    pins = {str(path): h.binding(path) for path in (*FIXED, Path(__file__))}
    h.require(all(pins[str(path)]['sha256'] == value for path, value in FIXED.items()), 'Fixed source/receipt pins')
    forbidden = {str((p.G19 / (f'mps_truth_rank{r:04d}_indep4M.npz' if r in p.PUBLIC
                              else f'fresh_{r}_truth4M.npz')).resolve()) for r in p.RANKS}

    def bind(mapping):
        for key, item in mapping.items():
            path, expected = (item['path'], item['sha256']) if isinstance(item, dict) else (key, item)
            path = str(Path(path).resolve())
            h.require(path not in forbidden, 'Independent truth bytes in pre-acquisition provenance')
            if Path(path).parent == P2 / 'public_g22_operator_features_v1/truthpost':
                # Old readiness includes TRAIN35/100 truth hashes. The pinned
                # judge/result authenticates those declarations; do NOT reopen
                # or rehash numerical training truths for this panel.
                continue
            h.require(h.digest(path) == expected, 'Source/artifact changed: ' + path)
            pins[path] = dict(path=path, sha256=expected)

    for name in ('result.json', 'completion.json', 'predictions_and_bills_sealed.json'):
        pins[str(JUDGE / name)] = h.binding(JUDGE / name)
    h.require(h.digest(JUDGE / 'result.json') == terminal_sha, 'Requested terminal judge seal')
    j, end = h.load_json(JUDGE / 'result.json'), h.load_json(JUDGE / 'completion.json')
    h.require(end['passed'] and end['completed'] and end['result_sha256'] == terminal_sha and
              j['passed'] and j['completed'] and j['integrity_passed'] and
              j['advances_to_independent14'] and j['source_and_artifact_hashes_unchanged'] and
              j['all_predictions_verified_before_truth'], 'Terminal40 did not qualify/advance')
    h.require(set(j['comparisons']) == {'bank', 'zero'} and all(
        v['passed'] and v['objective_improved'] and v['mean_adjusted_ratio'] < 1 and
        v['wins_gate'] and v['adjusted_wins'] >= 24 for v in j['comparisons'].values()), 'Terminal objective gate')
    h.require(h.digest(JUDGE / 'predictions_and_bills_sealed.json') ==
              j['predictions_and_bills_seal_sha256'], 'Terminal prediction/bill seal')
    seal = h.load_json(JUDGE / 'predictions_and_bills_sealed.json')
    h.require(seal['source_bindings'] == j['source_bindings'] and not seal['truth_accessed'], 'Terminal seal ownership')
    bind(j['source_bindings'])
    h.require(j['source_bindings'].get(str(HERE / 'judge_terminal.py')) == FIXED[HERE / 'judge_terminal.py'],
              'Actual terminal judge source')
    candidate_sha = j['source_bindings'].get(str(CANDIDATE))
    h.require(candidate_sha == h.digest(CANDIDATE), 'Exact terminal candidate source')
    build_path = CANDIDATE.with_name('build.json')
    h.require(j['source_bindings'].get(str(build_path)) == h.digest(build_path), 'Terminal build ownership')
    build = h.load_json(build_path)
    h.require(build['passed'] and build['inverse_source_byte_exact'] and
              build['all_other_parent_source_byte_exact'] and build['imports_and_learned_P_payloads_unchanged'] and
              build['parent_sha256'] == BANK_SHA and build['candidate_sha256'] == candidate_sha, 'Frozen build proof')
    bind(build['source_bindings'])
    for name in ('completion.json', 'started.json', 'predictions_and_bills_sealed.json'):
        pins[str(ARCHIVE / name)] = h.binding(ARCHIVE / name)
    base = h.load_json(ARCHIVE / 'result.json')
    complete = h.load_json(ARCHIVE / 'completion.json')
    h.require(complete['passed'] and complete['result_sha256'] == BASE_ACQ_SHA and base['passed'] and
              base['hashes_unchanged'] and not base['truth_files_accessed'] and base['candidate_sha256'] == BANK_SHA and
              base['versions'] == VERSIONS, 'Qualified current409f acquisition')
    h.require(h.digest(ARCHIVE / 'started.json') == base['started_sha256'] and
              h.load_json(ARCHIVE / 'started.json') == base['started'] and
              h.digest(ARCHIVE / 'predictions_and_bills_sealed.json') == base['outputs_seal_sha256'], 'Parent seals')
    saved = h.load_json(ARCHIVE / 'predictions_and_bills_sealed.json')
    h.require(saved['trials'] == base['trials'] and saved['artifacts'] == base['artifacts'] and
              saved['fixture_bindings'] == base['fixture_bindings'] and not saved['truth_files_accessed'], 'Parent output ownership')
    bind(base['source_bindings'])
    bind(base['artifacts'])
    fixtures, rows = p.fixtures(h)
    h.require(fixtures == base['fixture_bindings'] and
              [t['rank'] for t in base['trials']] == list(p.RANKS) + [35], 'Exact parent fixture sequence')
    h.require(base['trials'][0]['flops'] - base['trials'][-1]['flops'] == 1383913666 and
              base['trials'][0]['output_bytes_sha256'] == base['trials'][-1]['output_bytes_sha256'], 'Parent initialization/repeat')
    old = h.load_json(BASE_SCORE)
    pins[str(BASE_SCORE.with_name('completion.json'))] = h.binding(BASE_SCORE.with_name('completion.json'))
    h.require(h.load_json(BASE_SCORE.with_name('completion.json'))['passed'] and
              h.load_json(BASE_SCORE.with_name('completion.json'))['result_sha256'] == BASE_SCORE_SHA and
              old['passed'] and old['candidate_sha256'] == BANK_SHA and old['acquisition_sha256'] == BASE_ACQ_SHA and
              old['fixture_bindings'] == fixtures and [r['rank'] for r in old['per_net']] == list(p.RANKS), 'Same-frame parent score')
    for call, trial in enumerate(base['trials']):
        h.require(trial['call'] == call and trial['includes_initialization'] is (call == 0) and
                  h.load_json(ARCHIVE / f'call_{call:02d}_bill.json') == trial, 'Parent per-call bill/init')
        h.require(Path(base['artifacts'][trial['output_artifact']]['path']).resolve() == ARCHIVE / trial['output_artifact'],
                  'Parent output path')
        if call < 14:
            h.require(trial['flops'] == old['per_net'][call]['flops'], 'Parent scored bill identity')
    h.check_bindings(pins)
    return pins, fixtures, rows, base, old, candidate_sha


def output(h, np, path, bytes_sha):
    value = np.load(path, allow_pickle=False)
    h.require(type(value) is np.ndarray, 'Concrete saved output')
    h.assert_output(np, value)
    h.require(hashlib.sha256(value.tobytes()).hexdigest() == bytes_sha, 'Saved output bytes')
    return value


def verify_acquisition(h, p, terminal_sha, expected_sha=None):
    pins, fixtures, rows, base, old, candidate_sha = prerequisites(h, p, terminal_sha)
    directory = ROOT / 'candidate'
    result_sha = h.digest(directory / 'result.json')
    r, end = h.load_json(directory / 'result.json'), h.load_json(directory / 'completion.json')
    h.require(expected_sha is None or expected_sha == result_sha, 'Requested panel result seal')
    h.require(end['passed'] and end['completed'] and end['result_sha256'] == result_sha and
              r['passed'] and r['completed'] and r['hashes_unchanged'] and r['schema'] == SCHEMA and
              not r['truth_files_accessed'] and r['candidate_sha256'] == candidate_sha and
              r['terminal_score_sha256'] == terminal_sha and r['source_bindings'] == pins and
              r['fixture_bindings'] == fixtures and r['versions'] == VERSIONS, 'Passing fixed panel acquisition')
    h.require(h.digest(directory / 'started.json') == r['started_sha256'] and
              h.load_json(directory / 'started.json') == r['started'] and
              r['started']['initialization_once'] and r['started']['no_warmup'] and
              not r['started']['instrumented'] and r['started']['nice'] >= 10 and
              r['started']['blas_env'] == dict.fromkeys(BLAS, '1'), 'Acquisition start/runtime seal')
    h.require(h.digest(directory / 'predictions_and_bills_sealed.json') == r['outputs_seal_sha256'], 'Panel output seal')
    seal = h.load_json(directory / 'predictions_and_bills_sealed.json')
    h.require(all(seal[k] == r[k] for k in ('source_bindings', 'fixture_bindings', 'artifacts', 'trials', 'candidate_sha256'))
              and not seal['truth_files_accessed'], 'Panel seal ownership')
    h.require([t['rank'] for t in r['trials']] == list(p.RANKS) + [35], 'Panel14 plus repeat35')
    expected = set()
    for call, t in enumerate(r['trials']):
        name = f"call_{call:02d}_rank{t['rank']}_output.npy"
        bill = f'call_{call:02d}_bill.json'
        expected.update((name, bill))
        h.require(t['call'] == call and t['output_artifact'] == name and
                  t['includes_initialization'] is (call == 0) and t['layer0_byte_exact'] and
                  t['served_materialized_byte_exact'] and type(t['flops']) is int and 0 < t['flops'] <= h.BUDGET and
                  t['parent_flops'] == base['trials'][call]['flops'] and
                  all(math.isfinite(t[k]) and t[k] >= 0 for k in TIMINGS) and
                  0 < t['context_wall_s'] < 120 and t['meter_wall_s'] < 120 and
                  0 < t['rss_bytes'] < RSS_LIMIT, 'Panel call resource/output/bill gate')
        h.require(h.load_json(directory / bill) == t, 'Panel bill/result identity')
    h.require(set(r['artifacts']) == expected and all(
        Path(v['path']).resolve() == directory / k for k, v in r['artifacts'].items()), 'Exact panel artifacts')
    h.require(r['trials'][0]['output_bytes_sha256'] == r['trials'][-1]['output_bytes_sha256'] and
              r['initialization_flops'] == r['trials'][0]['flops'] - r['trials'][-1]['flops'] >= 0, 'Panel repeat/initialization')
    h.check_bindings(r['artifacts'])
    return r, (pins, fixtures, rows, base, old, candidate_sha)


def acquire(h, p, args, directory, result):
    pins, fixtures, rows, base, old, candidate_sha = prerequisites(h, p, args.terminal_score_sha256)
    started = dict(schema=SCHEMA, utc=h.utc(), pid=os.getpid(), executable=sys.executable,
                   nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, blas_env=dict.fromkeys(BLAS, '1'),
                   source_bindings=pins, fixture_bindings=fixtures, sequence=list(p.RANKS) + [35],
                   candidate_sha256=candidate_sha, terminal_score_sha256=args.terminal_score_sha256,
                   initialization_once=True, no_warmup=True, instrumented=False, truth_files_accessed=False,
                   archived_parent_runtime_not_contemporaneous=True,
                   prior_readiness_truth_declarations_sealed_but_arrays_not_reopened=True,
                   output_boundary='Concrete served FP32 checked and materialized inside BudgetContext; '
                   'finiteness/byte-view serialization checked outside; no cast or repair/deferred work')
    result.update(started=started, started_sha256=h.write_json(directory / 'started.json', started),
                  source_bindings=pins, fixture_bindings=fixtures, candidate_sha256=candidate_sha,
                  parent_sha256=BANK_SHA, terminal_score_sha256=args.terminal_score_sha256, versions=VERSIONS,
                  trials=[], artifacts={}, truth_files_accessed=False, instrumented=False)
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    flops.configure(symmetry_warnings=False)
    module = load(CANDIDATE, 'bd_panel_unwrapped_terminal', candidate_sha)
    estimator, first_bytes = None, None
    with threadpool_limits(limits=1):
        result['threadpools'] = threadpool_info()
        h.require(all(x['num_threads'] == 1 for x in result['threadpools'] if x.get('user_api') == 'blas'), 'BLAS thread count')
        for call, row in enumerate(rows + [rows[0]]):
            weights = np.load(row['weights_path'], mmap_mode='r', allow_pickle=False)
            h.require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32 and
                      bool(np.isfinite(weights).all()), 'Canonical weights')
            mlp = MLP(width=1024, depth=16, weights=list(weights),
                      seed=int(derive_estimator_seed(row['mlp_seed'])), name=f"birth-panel-{row['rank']}")
            began = h.utc(); before = time.perf_counter()
            signal.alarm(125)
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                if estimator is None:
                    estimator = module.Estimator()
                served = estimator.predict(mlp, h.BUDGET)
                h.require(isinstance(served, np.ndarray) and served.dtype == np.float32 and
                          served.shape == (16, 1024), 'Concrete served FP32(16,1024)')
                materialized = materialise_predictions(served, depth=16, width=1024)
            elapsed = time.perf_counter() - before
            signal.alarm(0)
            h.assert_output(np, np.asarray(served)); h.assert_output(np, np.asarray(materialized))
            raw_bytes = materialized.tobytes()
            h.require(served.tobytes() == raw_bytes, 'Materializer changed served bytes')
            values = np.frombuffer(raw_bytes, dtype=np.float32).reshape(16, 1024)
            prior = base['trials'][call]
            reference = output(h, np, ARCHIVE / prior['output_artifact'], prior['output_bytes_sha256'])
            name = f"call_{call:02d}_rank{row['rank']}_output.npy"
            result['artifacts'][name] = h.save_npy(np, directory / name, values)
            trial = dict(call=call, rank=row['rank'], began_utc=began, ended_utc=h.utc(),
                         flops=int(ctx.flops_used), parent_flops=prior['flops'], context_wall_s=elapsed,
                         meter_wall_s=float(ctx.wall_time_s), backend_s=float(ctx.flopscope_backend_time_s),
                         overhead_s=float(ctx.flopscope_overhead_time_s), native_residual_s=float(ctx.residual_wall_time_s),
                         rss_bytes=h.peak_rss_bytes(), output_artifact=name,
                         output_bytes_sha256=hashlib.sha256(raw_bytes).hexdigest(), includes_initialization=call == 0,
                         layer0_byte_exact=values[0].tobytes() == reference[0].tobytes(),
                         served_materialized_byte_exact=True)
            h.require(all(math.isfinite(trial[k]) and trial[k] >= 0 for k in TIMINGS),
                      'Finite timing fields before JSON serialization')
            result['trials'].append(trial)
            bill = f'call_{call:02d}_bill.json'
            h.write_json(directory / bill, trial); result['artifacts'][bill] = h.binding(directory / bill)
            print(json.dumps(trial, allow_nan=False), flush=True)
            h.require(trial['layer0_byte_exact'] and 0 < trial['flops'] <= h.BUDGET and
                      all(math.isfinite(trial[k]) and trial[k] >= 0 for k in TIMINGS) and
                      0 < elapsed < 120 and trial['meter_wall_s'] < 120 and trial['rss_bytes'] < RSS_LIMIT,
                      'Native panel output/resource gate')
            if call == 0:
                first_bytes = raw_bytes
            if call == 14:
                h.require(raw_bytes == first_bytes, 'Actual repeated rank35 bytes')
            del weights, mlp, served, materialized, values, reference, raw_bytes
            gc.collect()
    initialization = result['trials'][0]['flops'] - result['trials'][-1]['flops']
    h.require(initialization >= 0 and h.peak_rss_bytes() < RSS_LIMIT, 'Final initialization/resource gate')
    for items in (pins, fixtures, base['artifacts'], result['artifacts']):
        h.check_bindings(items)
    sealed = h.write_json(directory / 'predictions_and_bills_sealed.json', dict(
        utc=h.utc(), candidate_sha256=candidate_sha, source_bindings=pins, fixture_bindings=fixtures,
        trials=result['trials'], artifacts=result['artifacts'], truth_files_accessed=False))
    result.update(passed=True, completed=True, hashes_unchanged=True,
                  initialization_flops=initialization, outputs_seal_sha256=sealed)


def score(h, p, args, directory, result):
    r, (pins, fixtures, rows, base, old, candidate_sha) = verify_acquisition(h, p, args.terminal_score_sha256)
    import numpy as np  # No estimator or FlopScope import in score mode.
    acq = ROOT / 'candidate'
    seals = {str(acq / name): h.binding(acq / name) for name in
             ('started.json', 'result.json', 'completion.json', 'predictions_and_bills_sealed.json')}
    outputs = {'candidate': [], 'parent': []}
    for call in range(15):
        for role, record, folder in (('candidate', r, acq), ('parent', base, ARCHIVE)):
            trial = record['trials'][call]
            outputs[role].append(output(h, np, folder / trial['output_artifact'], trial['output_bytes_sha256']))
        h.require(outputs['candidate'][-1][0].tobytes() == outputs['parent'][-1][0].tobytes(), 'Loaded layer0 parity')
    h.require(all(v[0].tobytes() == v[-1].tobytes() for v in outputs.values()), 'Loaded repeat identity')
    for items in (pins, fixtures, seals, r['artifacts'], base['artifacts']):
        h.check_bindings(items)
    before_sha = h.write_json(directory / 'both_roles_verified_before_truth.json', dict(
        utc=h.utc(), role_seals=seals, source_bindings=pins, candidate_sha256=candidate_sha,
        parent_acquisition_sha256=BASE_ACQ_SHA, parent_score_sha256=BASE_SCORE_SHA,
        truth_files_accessed=False, ranks=list(p.RANKS), initialization_once=True, repeat_excluded=True))
    result.update(source_bindings=pins, fixture_bindings=fixtures, role_seals=seals,
                  candidate_sha256=candidate_sha, parent_sha256=BANK_SHA,
                  terminal_score_sha256=args.terminal_score_sha256,
                  both_roles_verified_before_truth_sha256=before_sha,
                  per_role={'candidate': [], 'parent': []}, truth_bindings={})
    truth_index = h.load_json(TRUTH_INDEX)
    h.require(truth_index['qualification_passed'] and truth_index['hashes_unchanged'] and
              set(truth_index['truth_bindings']) == {str(x) for x in p.RANKS}, 'Original independent truth index')
    result['truth_files_accessed'] = True
    h.write_json(directory / 'truth_access_started.json', dict(utc=h.utc(), prior_seal_sha256=before_sha))
    # Authenticate every fixed truth before opening any numerical truth array.
    for rank in p.RANKS:
        path = p.G19 / (f'mps_truth_rank{rank:04d}_indep4M.npz' if rank in p.PUBLIC else f'fresh_{rank}_truth4M.npz')
        expected = truth_index['truth_bindings'][str(rank)]
        h.require(Path(expected['path']).resolve() == path and h.digest(path) == expected['sha256'], 'Original4M truth binding')
        result['truth_bindings'][str(rank)] = h.binding(path)
    for call, rank in enumerate(p.RANKS):
        with np.load(result['truth_bindings'][str(rank)]['path'], allow_pickle=False) as src:
            mean, se = src['mc_post'][15].astype(np.float64), src['mc_se'][15].astype(np.float64)
        h.require(mean.shape == se.shape == (1024,) and bool(np.isfinite(mean).all()) and
                  bool(np.isfinite(se).all()) and bool((se >= 0).all()), 'Independent4M final truth contract')
        noise = float(np.mean(se * se))
        for role, record in (('candidate', r), ('parent', base)):
            trial = record['trials'][call]
            raw = float(np.mean((outputs[role][call][15].astype(np.float64) - mean) ** 2))
            nc = raw - noise  # Deliberately UNCLIPPED; same established estimator.
            factor = max(.1, trial['flops'] / h.BUDGET)
            h.require(math.isfinite(raw) and raw >= 0 and math.isfinite(nc), 'Finite score')
            row = dict(rank=rank, flops=trial['flops'], output_bytes_sha256=trial['output_bytes_sha256'],
                       raw_observed=raw, raw_noise_corrected=nc, mc_se_squared_mean=noise, multiplier=factor,
                       adjusted_observed=raw * factor, adjusted_noise_corrected=nc * factor)
            if role == 'parent':
                archived = old['per_net'][call]
                h.require(all(row[k] == archived[k] for k in
                          ('rank', 'flops', 'raw_observed', 'raw_noise_corrected', 'multiplier', 'adjusted_noise_corrected')),
                          'Archived parent own-bill/raw frame must reproduce EXACTLY')
            result['per_role'][role].append(row)
        print(json.dumps({role: result['per_role'][role][-1] for role in outputs}, allow_nan=False), flush=True)
    summaries = {}
    for role, scored in result['per_role'].items():
        summaries[role] = {}
        for group, ranks in (('public8', p.PUBLIC), ('virgin6', p.VIRGIN), ('combined14', p.RANKS)):
            subset = [t for t in scored if t['rank'] in ranks]
            fields = ('raw_observed', 'raw_noise_corrected', 'mc_se_squared_mean', 'flops',
                      'adjusted_observed', 'adjusted_noise_corrected')
            summaries[role][group] = dict(n=len(subset), **{
                'mean_' + key: sum(t[key] for t in subset) / len(subset) for key in fields})
    ratios, wins = {}, {}
    for group in ('public8', 'virgin6', 'combined14'):
        b = summaries['parent'][group]['mean_adjusted_noise_corrected']
        c = summaries['candidate'][group]['mean_adjusted_noise_corrected']
        h.require(b > 0, 'Positive group comparator denominator')
        ratios[group], wins[group] = c / b, c < b
    for items in (pins, fixtures, seals, r['artifacts'], base['artifacts'], result['truth_bindings']):
        h.check_bindings(items)
    h.require(h.digest(directory / 'both_roles_verified_before_truth.json') == before_sha and
              h.peak_rss_bytes() < RSS_LIMIT, 'Score end seal/resources')
    result.update(completed=True, passed=True, qualification_passed=True, hashes_unchanged=True,
                  baseline_recomputed_exact=True, summaries=summaries, adjusted_candidate_over_parent=ratios,
                  group_strict_gain_gate=wins, retain_local_component=wins['public8'] and wins['virgin6'],
                  acquisition_sha256=seals[str(acq / 'result.json')]['sha256'], parent_score_sha256=BASE_SCORE_SHA,
                  initialization_flops=r['initialization_flops'], repeat_excluded=True,
                  aggregation='Mean per-net (observed final MSE - mean(mc_se15**2)) * max(.1, own complete F/2**41); '
                  'unclipped; initialization once; repeated call14 excluded',
                  frame='Reused independent4M public8/virgin6. Public35/100/200 were TRAIN and150 VALID; '
                  'independent MC truth does not make these unseen model holdouts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--acquire', action='store_true')
    mode.add_argument('--score', action='store_true')
    parser.add_argument('--terminal-score-sha256')
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    value = args.terminal_score_sha256
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        parser.error('Both modes require the fixed --terminal-score-sha256')
    h, p = helpers()
    runtime(h)
    if args.acquire:
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), 'Acquire into exclusive fixed panel root first')
    directory = ROOT / ('candidate' if args.acquire else 'score')
    directory.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA, mode='acquire' if args.acquire else 'score', passed=False, completed=False,
                  started_utc=h.utc(), truth_files_accessed=False, production_residual_validated=False,
                  official_score=False, milestone_achieved=False, goal_achieved=False, no_refit=True)
    began = time.monotonic()

    def alarm(_signum, _frame):
        raise TimeoutError('Fixed panel hard watchdog')

    signal.signal(signal.SIGALRM, alarm)
    if args.score:
        signal.alarm(300)
    try:
        if args.acquire:
            acquire(h, p, args, directory, result)
        else:
            score(h, p, args, directory, result)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        result.update(completed_utc=h.utc(), wall_seconds=time.monotonic() - began, peak_rss_bytes=h.peak_rss_bytes())
        digest = h.write_json(directory / 'result.json', result)
        h.write_json(directory / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
                     result_sha256=digest, retain_local_component=result.get('retain_local_component', False)))
    print(json.dumps({k: result.get(k) for k in
          ('passed', 'completed', 'retain_local_component', 'adjusted_candidate_over_parent', 'error')}, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
