"""Inert CD production preparation and saved-output fidelity qualification.

No estimator, remote stage, upload or numerical library is run at import.
--prepare is stdlib-only. --qualify imports NumPy only after the complete
five-call production watch and all target-free prerequisite seals pass.
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
import textwrap
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
ENV = P2 / 'data_gate/.venv'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
PREP, QUAL = HERE / 'production_v1/prepare', HERE / 'production_v1/qualify'
OLD = NEW / 'birth_direction_fit_v1/release_confirmation_v1/check_release.py'
OLD_SHA = '3aac716546920c844d60e6d51fd8f5673c3ae055b2e10fae0456709fe69d68ee'
SOURCE = HERE / 'build_v1/estimator.py'
CANDIDATE = 'current_comp_partial_d6_v1/build_v1/estimator.py'
SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
TAG = 'current_comp_partial_d6_production_v1'
RANKS = (35, 100, 500, 3016500, 35)
INIT, BUDGET = 1944970298, 2**41
PANEL_NC = 4.2691389272565025e-9
LAUNCH_ENV = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
    MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
FROZEN = {
    OLD: OLD_SHA,
    SOURCE: SHA,
    NEW / 'run_remote_series_v1.py': '00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1',
    NEW / 'check_remote_series_v1.py': 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}
# These are actual completed current-CD receipts, not prospective hashes.
RECEIPTS = {
    'build': ('build_v1', 'bf17b06afb5e8b7e5b526768c550a7c59eb6ec9808d060cb79b8e1dd083a18af', 'current_comp_partial_d6_build_v1'),
    'build_watch': ('build_watch_v1', '95bae47d29f9767de13eb63b1b88e76ddab11e81ee977af38e4404886b08c53f', 'current_comp_partial_d6_build_watch_v1'),
    'native40': ('native40_terminal_v1', 'b9434921a668538f2cf434a6df773bca7e42fab94c9acc5f85f1213592f6fc2b', 'current_comp_partial_d6_native40_v1'),
    'native40_watch': ('native40_terminal_watch_v1', '29a6df3cd06b7c0c5e315315c1a85a240dfe0df592ebb26df9408de9fca42e87', 'current_comp_partial_d6_native40_watch_v1'),
    'score40': ('score40_v1', '939f4d1d18172fa732d04f21998a69b498aeb9421db43bb40c8b69c1ca125e81', 'current_comp_partial_d6_score40_v1'),
    'score40_watch': ('score40_watch_v1', '9ded35a26256179bca426344f9c0752fcfef7534b8ebbf58041be9ed7e39c3a6', 'current_comp_partial_d6_score40_watch_v1'),
    'panel': ('panel_v2/terminal', '3aaf29635c3822067f37a9cc934e1f8ae2692beef48be4538c6caa43189cc7f1', 'current_comp_partial_d6_independent14_native_v2'),
    'panel_watch': ('panel_v2/terminal_watch', '376323c876acc78594f7f23caa67c2a0388ea32374e540c75a8bec8273ec5f94', 'current_comp_partial_d6_independent14_watch_v2'),
    'panel_score': ('panel_v2/score', 'aab95242f83cc9a0a1937a2e6b195830e1ef8d1bc8dc4366101d732426998481', 'current_comp_partial_d6_independent14_score_v1'),
    'panel_score_watch': ('panel_v2/score_watch', '653ace788f0474335cd2c20d04b385069e521f2f625e584c2a2a73f19c787a2c', 'current_comp_partial_d6_independent14_score_watch_v1'),
}
STARTED = None
TICK = None


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def guard():
    if STARTED is not None:
        require(time.monotonic() - STARTED < 300, 'Checker whole-stage 300s')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000, 'Checker 2GB')
    if TICK is not None:
        TICK()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            guard()
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size < 64 * 1024**2, 'Bounded metadata JSON')
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return sha(path)


def declare(pins, path, expected):
    key = str(Path(path).resolve())
    require(isinstance(expected, str) and len(expected) == 64 and
            all(c in '0123456789abcdef' for c in expected), 'SHA256 declaration')
    require(key not in pins or pins[key] == expected, 'Conflicting immutable binding: ' + key)
    pins[key] = expected
    return expected


def pin(pins, path, expected=None):
    value = sha(path)
    require(expected is None or value == expected, 'Changed artifact: ' + str(path))
    return declare(pins, path, value)


def unchanged(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, 'Changed qualification artifact: ' + path)


def three(folder, expected, pins, legacy_watch=False):
    folder = Path(folder)
    paths = [folder / name for name in ('result.json', 'completion.json', 'finalization.json')]
    digests = [pin(pins, paths[0], expected), pin(pins, paths[1]), pin(pins, paths[2])]
    r, c, f = [read(p) for p in paths]
    require(r['passed'] is True and c['passed'] is True and f['passed'] is True, 'Three PASS seals')
    require(c['result_sha256'] == digests[0] and f['completion_sha256'] == digests[1], 'Actual three-seal chain')
    if legacy_watch:
        # Original native40/score40 watchers deliberately have this older schema.
        require(f['elapsed_through_completion_io_s'] >= c['elapsed_through_result_hash_s'], 'Legacy final publication clock')
        require(set(f) == {'passed', 'completion_sha256', 'elapsed_through_completion_io_s'}, 'Exact old finalization schema')
    else:
        require(all(x['completed'] is True for x in (r, c, f)), 'Three completed seals')
        require(f['result_sha256'] == digests[0] and math.isfinite(f['elapsed_s']), 'Final result ownership')
    return r, dict(zip(('result_sha256', 'completion_sha256', 'finalization_sha256'), digests))


def source_helpers():
    """Extract only the old numerical qualifier body; never import its module."""
    require(sha(OLD) == OLD_SHA, 'Frozen d50 qualifier owner')
    source = OLD.read_text()
    start = source.index('        series_path, resource_path = NEW / (TAG')
    end = source.index('    except BaseException:', start)
    original = source[start:end]
    edits = [
        ("HERE / 'launch.json'", "PREP / 'launch.json'"),
        ("HERE / 'outputs_and_bills_sealed_before_truth.json'", "QUAL / 'outputs_and_bills_sealed_before_truth.json'"),
        ("panel['per_role']['candidate']", "panel['per_role']['terminal']"),
        ('local_panel_nc_adjusted=4.4837991190464774e-9', 'local_panel_nc_adjusted=4.2691389272565025e-9'),
    ]
    derived = original
    for old, new in edits:
        require(old in derived and new not in derived, 'Unique vocabulary source adapter')
        derived = derived.replace(old, new)
    inverse = derived
    for old, new in reversed(edits):
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Byte-exact original numerical-body inverse')
    body = textwrap.dedent(derived)
    code = 'def qualified_numeric(pins, native, panel, result):\n' + textwrap.indent(body, '    ') + '    return result\n'
    proof = dict(owner_sha256=OLD_SHA, original_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_body_sha256=hashlib.sha256(derived.encode()).hexdigest(), source_inverse_verified=True,
        numerical_operations_and_thresholds_unchanged=True,
        changes=['exclusive CD paths', 'panel role terminal', 'actual CD independent14 aggregate'],
        native_vector_max_abs=1e-5, native_remote_raw_relative_limit=.01)
    return code, proof


def command():
    return [str(ENV / 'bin/python'), '-B', '-u', str(NEW / 'run_remote_series_v1.py'),
            '--candidate', CANDIDATE, '--tag', TAG]


def admission(args, tick=None):
    global TICK
    TICK = tick
    require(not any(n.split('.')[0] in {'numpy', 'torch', 'flopscope', 'flopscope_server'} for n in sys.modules), 'No numerical imports in admission')
    pins = {}
    for p, s in FROZEN.items():
        pin(pins, p, s)
    for p, s in ((Path(__file__), args.source_sha256), (HERE / 'PRODUCTION_PROTOCOL.md', args.protocol_sha256),
                 (HERE / 'production_watch.py', args.watcher_sha256), (HERE / 'PRODUCTION_WATCH_SPEC.md', args.watch_spec_sha256)):
        pin(pins, p, s)
    records, seals = {}, {}
    for name, (folder, digest, schema) in RECEIPTS.items():
        records[name], seals[name] = three(HERE / folder, digest, pins, name in ('native40_watch', 'score40_watch'))
        require(records[name]['schema'] == schema, 'Exact historical schema: ' + name)
    # Recorded source/input maps are authenticated receipt boundaries, not module imports.
    excluded_truths = {str(Path(v['path']).resolve()) for name in ('score40', 'panel_score')
                       for v in records[name]['truth_bindings'].values()}
    for r in records.values():
        for field in ('source_bindings', 'input_bindings', 'artifacts'):
            for p, s in r.get(field, {}).items():
                require(str(Path(p).resolve()) not in excluded_truths, 'Target-free historical inventory')
                declare(pins, p, s)
    for child, watcher in (('build', 'build_watch'), ('native40', 'native40_watch'),
                           ('score40', 'score40_watch'), ('panel', 'panel_watch'), ('panel_score', 'panel_score_watch')):
        w = records[watcher]
        require(w['sources_unchanged'] and w['whole_owned_tree_retired'], 'Original watcher integrity/retirement')
        for key, digest in seals[child].items():
            require(w['child_' + key] == digest, 'Actual child three-seal ownership')
    for name in ('build', 'native40', 'score40', 'panel', 'panel_score'):
        require(records[name]['hashes_unchanged'] and records[name]['sources_unchanged'], 'Historical hash/source PASS')
    b, n40, s40, native, panel = [records[k] for k in ('build', 'native40', 'score40', 'panel', 'panel_score')]
    require(b['candidate_sha256'] == native['candidate_sha256'] == n40['candidate_sha256'] == panel['candidate_sha256'] == SHA, 'Single unchanged candidate')
    require(b['candidates']['terminal']['sha256'] == SHA, 'Built terminal literal')
    require(b['proof']['parent_inverse_sha256'] == '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a', 'Exact COMP inverse')
    for k in ('whole_source_inverse_verified', 'new_partial_inverse_to_COMP_verified', 'P_A_cstar_literal_bytes_unchanged', 'all_unrelated_source_bytes_unchanged'):
        require(b['proof'][k] is True, 'Original build proof: ' + k)
    require(b['proof']['additional_both_binding_flops'] == 561056000 and not b['proof']['second_backing_arena'], 'D6 binding ownership')
    require(s40['advances_to_independent14'] and s40['comparisons']['terminal_vs_COMP']['gate_passed'], 'Actual native40 advancement')
    require(s40['comparisons']['terminal_vs_COMP']['at_least_24_wins'] and len(s40['records']) == 40, 'Forty complete comparisons')
    for key, name in (('native', 'native40'), ('watch', 'native40_watch'), ('build', 'build'), ('build_watch', 'build_watch')):
        require(s40['lineage']['terminal'][key] == seals[name], 'Native40 lineage join')
    require(native['native40_lineage']['score'] == seals['score40'] and native['native40_lineage']['score_watch'] == seals['score40_watch'], 'Panel follows exact native40 decision')
    require(panel['acquisition_sha256'] == seals['panel']['result_sha256'] and panel['acquisition_watch_sha256'] == seals['panel_watch']['result_sha256'], 'Panel acquisition joins')
    require(panel['acquisition_finalization_sha256'] == seals['panel']['finalization_sha256'], 'Panel finalization join')
    require(panel['qualification_passed'] and panel['retain_local_component'] and panel['no_refit'], 'Completed panel qualification')
    require(all(panel['group_strict_gain_gate'][g] is True for g in ('public8', 'virgin6')), 'Strict BOTH independent groups')
    require(panel['summaries']['terminal']['combined14']['mean_adjusted_noise_corrected'] == PANEL_NC, 'Exact retained panel aggregate')
    for r, count in ((n40, 41), (native, 15)):
        require(r['full_model_calls'] == r['retired_call_count'] == count, 'Complete native roster')
        for k in ('own_complete_bills', 'all16_domains_qualified', 'all47_selectors_qualified', 'initialization_ownership_passed',
                  'repeat_output_domain_byte_exact', 'additional_bindings_paid_once', 'same_existing_arena_qualified', 'dispatch_tariffs_qualified'):
            require(r[k] is True, 'Native acquisition gate: ' + k)
        require(not r['truth_accessed'], 'Native is target-free')
    require(native['initialization_flops'] == INIT and native['initialization_once'] and native['repeat_excluded'], 'Original panel startup')
    require(native['all90_artifacts_qualified'] and len(native['artifacts']) == 90 and len(native['trials']) == 15, 'All90 panel artifacts')
    require(not native['candidate_state_reset'] and not native['input_preparation_in_participant_bill'], 'Unchanged lifecycle/input ownership')
    require(native['input_representation'] == 'owned FNP float32 copies, official loader convention', 'Original input representation')
    require(native['trials'][0]['output_bytes_sha256'] == native['trials'][-1]['output_bytes_sha256'], 'Original repeat bytes')
    require(native['trials'][0]['flops'] - native['trials'][-1]['flops'] == INIT, 'Original own startup bill')
    fixtures = {}
    man = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    for rank in sorted(set(RANKS)):
        if rank < 1000:
            require(man[rank]['role_rank'] == rank, 'Canonical manifest rank')
            fixture = STORE / Path(man[rank]['path']).parent / 'weights.npy'
        else:
            fixture = P2 / 'public_g19_attribution_v1' / f'fresh_{rank}_weights.npy'
        entry = native['fixture_bindings'][f'weights_{rank}']
        require(Path(entry['path']).resolve() == fixture.resolve(), 'Exact production/native fixture')
        declare(pins, fixture, entry['sha256'])
        fixtures[str(rank)] = dict(path=str(fixture.resolve()), sha256=entry['sha256'])
    for entry in native['fixture_bindings'].values():
        declare(pins, entry['path'], entry['sha256'])
    code, proof = source_helpers()
    unchanged(pins)
    sources = {p: s for p, s in pins.items() if Path(p).suffix in ('.py', '.md', '.so', '.pyi')}
    inputs = {p: s for p, s in pins.items() if p not in sources}
    return dict(pins=pins, source_bindings=sources, input_bindings=inputs, fixture_bindings=fixtures,
                native=native, panel=panel, prerequisites=seals, source_proof=proof, numerical_source=code)


def series_paths():
    return [NEW / (TAG + s) for s in ('.json', '_resources.json', '_server.log')] + [NEW / f'{TAG}_{i}_r{r}.f32' for i, r in enumerate(RANKS)]


def preparation(admitted):
    require(not any(p.exists() for p in series_paths()), 'Exclusive future production files')
    launch = dict(schema='current_comp_partial_d6_production_launch_v1', passed=True, completed=True,
        candidate_sha256=SHA, candidate=CANDIDATE, tag=TAG, ranks=list(RANKS), env=LAUNCH_ENV,
        command=command(), cwd=str(NEW), initialization_flops=INIT, utc=utc(),
        source_bindings=admitted['source_bindings'], input_bindings=admitted['input_bindings'],
        fixture_bindings=admitted['fixture_bindings'], prerequisites=admitted['prerequisites'],
        source_inverse_proof=admitted['source_proof'], uploaded=False, estimator_executed=False,
        truth_accessed=False, submission_authorized=False,
        user_explicit_confirmation_bank_authorized=False,
        production_qualified=False, automatic_next_stage=False)
    return write(PREP / 'launch.json', launch)


def production_watch(args, admitted):
    extra = {}
    preparation_record, preparation_seals = three(PREP, args.preparation_sha256, extra)
    pin(extra, PREP / 'launch.json', args.launch_sha256)
    launch = read(PREP / 'launch.json')
    require(preparation_record['schema'] == 'current_comp_partial_d6_production_check_v1'
            and preparation_record['launch_sha256'] == args.launch_sha256
            and preparation_record['stage'] == 'prepare', 'Prepared launch ownership')
    for field in ('source_bindings', 'input_bindings', 'fixture_bindings', 'prerequisites'):
        require(launch[field] == admitted[field], 'Unchanged admitted launch: ' + field)
    require(launch['candidate_sha256'] == SHA and launch['candidate'] == CANDIDATE and launch['tag'] == TAG, 'Launch candidate')
    require(launch['ranks'] == list(RANKS) and launch['command'] == command() and launch['env'] == LAUNCH_ENV and launch['cwd'] == str(NEW), 'Exact launch command/environment')
    require(launch['initialization_flops'] == INIT and not launch['uploaded'] and not launch['truth_accessed'], 'Launch scope')
    watch, watch_seals = three(HERE / 'production_watch_v1', args.production_watch_sha256, extra)
    require(watch['schema'] == 'current_comp_partial_d6_production_watch_v1', 'New production watcher')
    require(watch['launch_sha256'] == args.launch_sha256 and watch['preparation_sha256'] == args.preparation_sha256 and watch['candidate_sha256'] == SHA, 'Exact supervised production launch')
    for flag in ('hashes_unchanged', 'sources_unchanged', 'cleanup_passed', 'whole_owned_tree_retired', 'observational_groups_retired', 'actual_completed_samples'):
        require(watch[flag] is True, 'Production watch gate: ' + flag)
    require(watch['full_model_calls'] == 5 and len(watch['observed_records']) == 5, 'All five production calls')
    require(0 < watch['maximum_combined_rss_bytes'] < 8e9 and 0 < watch['maximum_interval_s'] <= 1 and watch['maximum_swap_growth_bytes'] == 0, 'Actual production resources')
    intervals = watch['completed_sample_intervals_s']
    require(intervals and all(math.isfinite(x) and 0 < x <= 1 for x in intervals) and max(intervals) == watch['maximum_interval_s'], 'Every completed sample interval')
    require(watch['elapsed_s'] < 600 and not watch['truth_arrays_accessed'] and watch['numerical_imports'] is False, 'Whole production scope')
    for p, digest in ((HERE / 'production_watch.py', args.watcher_sha256), (Path(__file__), args.source_sha256),
                      (HERE / 'PRODUCTION_PROTOCOL.md', args.protocol_sha256), (HERE / 'PRODUCTION_WATCH_SPEC.md', args.watch_spec_sha256)):
        require(watch['source_bindings'][str(p.resolve())] == digest, 'Production reviewed source owner')
    for field in ('source_bindings', 'input_bindings', 'artifacts'):
        for p, s in watch[field].items():
            declare(extra, p, s)
    # The watcher must own the series, raw outputs, resource record and server log.
    all_watch = {**watch['input_bindings'], **watch['artifacts']}
    for p in series_paths():
        require(str(p.resolve()) in all_watch, 'Complete pretruth production artifact')
        pin(extra, p, all_watch[str(p.resolve())])
    require(read(NEW / (TAG + '.json'))['records'] == watch['observed_records'], 'Independent actual-five record replay')
    unchanged(extra)
    return extra, dict(preparation=preparation_seals, production_watch=watch_seals)


def main():
    global STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--qualify', action='store_true')
    for name in ('source', 'protocol', 'watcher', 'watch-spec', 'launch', 'preparation', 'production-watch'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if not (args.prepare or args.qualify):
        parser.print_help()
        return 0
    STARTED = time.monotonic()
    required = ('source', 'protocol', 'watcher', 'watch_spec') + (('launch', 'preparation', 'production_watch') if args.qualify else ())
    for name in required:
        value = getattr(args, name + '_sha256')
        require(isinstance(value, str) and len(value) == 64 and
                all(c in '0123456789abcdef' for c in value), 'Explicit reviewed/actual SHA: ' + name)
    require(Path(sys.prefix).resolve() == ENV.resolve() and Path(sys.executable).parent.resolve() == (ENV / 'bin').resolve(), 'Pinned OFFPY environment')
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Assertions active, bytecode disabled and nice10')
    require(all(os.environ.get(k) == v for k, v in LAUNCH_ENV.items()), 'Frozen launch environment')
    require({k: importlib.metadata.version(k) for k in ('numpy', 'flopscope', 'whestbench')} == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0'), 'Original runtime packages')
    def timeout(_signum, _frame):
        raise TimeoutError('Production checker whole-stage 300s')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    folder = PREP if args.prepare else QUAL
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema='current_comp_partial_d6_production_check_v1', stage='prepare' if args.prepare else 'qualify',
        passed=False, completed=False, candidate_sha256=SHA, source_bindings={}, input_bindings={},
        fixture_bindings={}, artifacts={}, checks=[], truth_bindings={}, truth_accessed=False,
        estimator_executed=False, uploaded=False, official_score=False, automatic_next_stage=False,
        started_utc=utc())
    try:
        admitted = admission(args)
        result.update({k: admitted[k] for k in ('source_bindings', 'input_bindings', 'fixture_bindings', 'prerequisites')})
        result['source_inverse_proof'] = admitted['source_proof']
        if args.prepare:
            result['launch_sha256'] = preparation(admitted)
            declare(result['artifacts'], PREP / 'launch.json', result['launch_sha256'])
            result.update(passed=True, completed=True, production_qualified=False, numerical_imports=False)
        else:
            extra, extra_seals = production_watch(args, admitted)
            for p, s in extra.items():
                declare(result['input_bindings'], p, s)
                declare(admitted['pins'], p, s)
            result['production_prerequisites'] = extra_seals
            env = dict(globals())
            exec(compile(admitted['numerical_source'], str(OLD) + ':CD-exact-fidelity', 'exec'), env)
            env['qualified_numeric'](admitted['pins'], admitted['native'], admitted['panel'], result)
            result.update(production_qualified=True, numerical_imports=['numpy'], launch_sha256=args.launch_sha256,
                preparation_sha256=args.preparation_sha256, production_watch_sha256=args.production_watch_sha256)
        for field in ('source_bindings', 'input_bindings', 'artifacts', 'truth_bindings'):
            unchanged(result[field])
        guard()
        result.update(hashes_unchanged=True, sources_unchanged=True)
    except BaseException:
        result.update(passed=False, completed=False, production_qualified=False, error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic() - STARTED, finished_utc=utc(), peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    digest = write(folder / 'result.json', result)
    completion = write(folder / 'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    elapsed = time.monotonic() - STARTED
    final_pass = result['passed'] and elapsed < 300 and result['peak_rss_bytes'] < 2_000_000_000
    final = write(folder / 'finalization.json', dict(passed=final_pass, completed=final_pass,
        result_sha256=digest, completion_sha256=completion, elapsed_s=elapsed))
    guard()
    signal.alarm(0)
    print(json.dumps(dict(passed=final_pass, stage=result['stage'], result_sha256=digest,
        completion_sha256=completion, finalization_sha256=final, error=result.get('error'))), flush=True)
    return 0 if final_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
