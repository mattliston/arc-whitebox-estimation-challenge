"""APUSH fixed CHECK4 saved-output score; inert until explicit --run."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CD = NEW/'current_comp_partial_d6_v1'
JOINT = NEW/'current_comp_joint_p_node_v1/score.py'
COMP = NEW/'current_p1_node_composition_v1'
RC = NEW/'current_d50_response10_candidate_v1/score.py'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
INDEX = NEW/'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
RANKS, ARMS, BUDGET = (37, 39, 42, 48), ('terminal',), 2**41
SCHEMA = 'current_comp_aligned_push_d6_score_v1'
INITIALIZATION = 1944970298
CD_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
BASE_SHA = 'b9434921a668538f2cf434a6df773bca7e42fab94c9acc5f85f1213592f6fc2b'
SCORE_SHA = '939f4d1d18172fa732d04f21998a69b498aeb9421db43bb40c8b69c1ca125e81'
PINS = {
    JOINT: '7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e',
    RC: '3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',
    COMP/'native.py': 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    CENSUS: '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    INDEX: '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c',
    HERE/'DECISION.md': 'ff026a7659875606a06206a89f047331a77d86af7da030266be74130bbf784f0',
    HERE/'build.py': 'ef008756a35e3e15e3f0286b6c846ea8d47c63a838d575bb01e54661cbe60f2b',
    CD/'build_v1/estimator.py': CD_SHA,
}
# Actual completed CD boundaries, not a reconstruction of historical eligibility.
BASE_CHAINS = {
    'native40_terminal_v1': (BASE_SHA,
        '4397428b9cbbe76d1f96a64897ca6ae1dcd65faaf442f8a451ac74b86dd02f05',
        'a93c6670e67bd96ae0f19edf946288adb1ef210d45eb31347b9a51e419febbea'),
    'native40_terminal_watch_v1': ('29a6df3cd06b7c0c5e315315c1a85a240dfe0df592ebb26df9408de9fca42e87',
        '3f9f2bf54f585674c3bac14eb99b5561634a9c28f80bcfa6d707a6aa28d46f8a',
        'dcf7a4ea46735f0ec1aacf34a107fc740f6f58dbd291b512bc10ce93d4566493'),
    'score40_v1': (SCORE_SHA,
        'c4641e744bdccccacc6107059e99d1438b5237dd34efd0aa3749719851b31d4e',
        '32560f73b5e132fb16956f49d4241aa8831dac1c3efd872ec124f308498f5e4c'),
    'score40_watch_v1': ('9ded35a26256179bca426344f9c0752fcfef7534b8ebbf58041be9ed7e39c3a6',
        '75868e9deeb58f8ac6fc194689c66d2d4bdd3daaca9c2c5c3fa942e3ec5f6d2a',
        '9adf3144fa0924982044bfc28c8ea1aaf06ff15e19ba74d28b63b5c790ab1b0b'),
}
BASE_SAMPLES = {
    'native40_terminal_watch_v1': '7de02bf73eb14caf81ad8fe83b2242f27408ff0e813f5db2ea79f9be86c8dbd4',
    'score40_watch_v1': 'a1e4613401dfd211b72c99823a992d014d5d00b7ed389ed54c74b05ee76817fd',
}
SHA_NAMES = ('source', 'protocol', 'watcher', 'watch-spec', 'native-source', 'native-protocol',
    'native-watch-source', 'native-watch-spec', 'native', 'native-watch', 'admission', 'build', 'runtime', 'candidate')
GUARD = lambda: None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block); GUARD()
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def extract(path, name):
    source = Path(path).read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Unique frozen function '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def inverse_adapter(source, changes):
    result = source
    for before, after, count in changes:
        require(result.count(before) == count, 'Exact source span '+before[:90])
        result = result.replace(before, after)
    inverse = result
    for before, after, count in reversed(changes):
        require(inverse.count(after) == count, 'Exact inverse span '+after[:90])
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Whole extracted source inverse')
    ast.parse(result)
    return result


class Admission:
    def __init__(self, result, guard):
        self.result, self.guard, self.pins = result, guard, result['input_bindings']

    def pin(self, path, digest):
        path = Path(path).resolve(); key = str(path)
        require(path.is_relative_to(P2) and 'truthpost' not in path.parts
            and not (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower()), 'Target-free admission path')
        require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'Exact SHA256 binding')
        require(self.pins.get(key, digest) == digest, 'Consistent immutable binding')
        if key not in self.pins:
            require(sha(path) == digest, 'Immutable source/input '+key); self.pins[key] = digest
        if path.suffix in ('.py', '.md'):
            self.result['source_bindings'][key] = digest
        self.guard(); return digest

    def bind(self, mapping, ignored=None):
        for key, value in mapping.items():
            path, digest = (value['path'], value['sha256']) if isinstance(value, dict) else (key, value)
            self.pin(path, digest)

    def inventory(self, record):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'fixture_bindings', 'artifacts'):
            self.bind(record.get(group, {}))  # Never target_bindings/truth_bindings.

    def chain(self, folder, digest, *, limit=None, legacy_watch=False, inventory=False, expected=None):
        paths = [folder/name for name in ('result.json', 'completion.json', 'finalization.json')]
        hashes = [digest, sha(paths[1]), sha(paths[2])]
        require(expected is None or tuple(hashes) == expected, 'Frozen historical three actual seals')
        for path, value in zip(paths, hashes):
            self.pin(path, value)
        r, e, f = map(read, paths)
        require(all(v.get('passed') is True for v in (r, e, f)) and e['result_sha256'] == digest
            and f['completion_sha256'] == hashes[1], 'Successful actual three-seal chain '+str(folder))
        if legacy_watch:
            require(e['elapsed_through_result_hash_s'] < limit and f['elapsed_through_completion_io_s'] < limit,
                'Original watch final-publication bound; no invented completed fields')
        else:
            require(all(v.get('completed') is True for v in (r, e, f)) and f['result_sha256'] == digest,
                'Complete new three-seal chain')
            require(limit is None or 0 <= f['elapsed_s'] < limit, 'Complete worker publication deadline')
        if inventory:
            self.inventory(r)
        return r, e, dict(zip(('result_sha256', 'completion_sha256', 'finalization_sha256'), hashes))


def helpers(admit):
    """Only exact saved-output/domain/lifecycle functions, no legacy module imports."""
    life = extract(CENSUS, 'lifecycle'); npy = extract(COMP/'native.py', 'npy_bytes')
    original = extract(RC, 'trial_output')
    trial = inverse_adapter(original, [('value.dtype == np.float64', 'value.dtype == np.float32', 1)])
    control_original = extract(JOINT, 'original_control_checks')
    control = inverse_adapter(control_original, [('Original own COMP', 'Original own CD', 1),
        ('Archived own COMP', 'Archived own CD', 1)])
    env = dict(Path=Path, ast=ast, hashlib=hashlib, math=math, require=require, sha=sha, read=read,
        bind=admit.bind, BUDGET=BUDGET)
    exec(compile(life+'\n'+npy+'\n'+trial+'\n'+control, str(JOINT)+':saved-functions-only', 'exec'), env)
    proof = dict(lifecycle_exact_source_sha256=hashlib.sha256(life.encode()).hexdigest(),
        npy_exact_source_sha256=hashlib.sha256(npy.encode()).hexdigest(),
        saved_output_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        saved_FP32_output_sha256=hashlib.sha256(trial.encode()).hexdigest(), saved_dtype_inverse_verified=True,
        control_inverse_verified=True, imported_legacy_modules=False)
    return env, proof


def baseline(admit, env):
    records, seals = {}, {}
    for name, expected in BASE_CHAINS.items():
        watch = 'watch' in name
        records[name], _, seals[name] = admit.chain(CD/name, expected[0],
            limit=1900 if name.startswith('native') else 120, legacy_watch=watch, expected=expected)
    native, scored = records['native40_terminal_v1'], records['score40_v1']
    require(native['schema'] == 'current_comp_partial_d6_native40_v1' and native['candidate_sha256'] == CD_SHA
        and native['full_model_calls'] == native['retired_call_count'] == len(native['trials']) == 41
        and native['initialization_flops'] == INITIALIZATION and native['ranks'][:4] == list(RANKS)
        and [t['rank'] for t in native['trials'][:4]] == list(RANKS) and native['trials'][-1]['rank'] == 37
        and native['truth_accessed'] is False and all(native.get(k) is True for k in
            ('sources_unchanged', 'hashes_unchanged', 'own_complete_bills', 'all16_domains_qualified',
             'all47_selectors_qualified', 'repeat_output_domain_byte_exact', 'initialization_once',
             'additional_bindings_paid_once', 'same_existing_arena_qualified', 'dispatch_tariffs_qualified')),
        'Qualified completed CD native40 boundary, original first-four histories')
    require(scored['schema'] == 'current_comp_partial_d6_score40_v1' and len(scored['records']) == 40
        and all(scored.get(k) is True for k in ('integrity_passed', 'hashes_unchanged', 'sources_unchanged',
            'all_predictions_verified_before_truth', 'exact_CHECK4_first4_score_replay', 'advances_to_independent14'))
        and scored['requested_seals']['terminal_native_sha256'] == BASE_SHA
        and scored['requested_seals']['terminal_candidate_sha256'] == CD_SHA,
        'Frozen successful CD original-precision score, joined to actual native40')
    for name, child_name in (('native40_terminal_watch_v1', 'native40_terminal_v1'), ('score40_watch_v1', 'score40_v1')):
        watch = records[name]; cap = 1900 if name.startswith('native') else 120
        require(all(watch.get(k) is True for k in ('sources_unchanged', 'whole_owned_tree_retired', 'cleanup_passed'))
            and watch['elapsed_s'] < cap and watch['maximum_interval_s'] <= 1
            and watch['maximum_combined_rss_bytes'] < (8_000_000_000 if cap == 1900 else 1_000_000_000)
            and all(watch['child_'+key] == value for key, value in seals[child_name].items()),
            'Actual CD independent-watch resource/child seals')
        path = CD/name/'samples.jsonl'; admit.pin(path, BASE_SAMPLES[name])
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(samples) == watch['observation_samples'] and samples and not samples[-1]['child_pids']
            and all(s['swap_bytes'] <= watch['initial_swap_bytes'] for s in samples)
            and max(s['combined_rss_bytes'] for s in samples) == watch['maximum_combined_rss_bytes'],
            'Original saved zero-swap observations and retired final tree')
    rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']; refs = {}
    for call, rank in enumerate(RANKS):
        found = [row for row in rows if row['role_rank'] == rank]
        require(len(found) == 1, 'One canonical weight fixture per CHECK rank')
        rp = (STORE/found[0]['path']).resolve(); require(rp.is_relative_to(STORE), 'Canonical STORE path')
        admit.pin(rp, found[0]['sha256']); fixture = read(rp); wp = rp.parent/'weights.npy'
        wh = fixture['files']['weights.npy']['sha256']; admit.pin(wp, wh)
        prepared = native['input_preparation'][call]
        require(fixture['role_rank'] == rank and fixture['role'] == 'observer_fit'
            and prepared['rank'] == rank and prepared['call'] == call and prepared['weight_file_sha256'] == wh,
            'CD native first-four exact canonical weight SHA')
        refs[rank] = dict(trial=native['trials'][call], weight_file_sha256=wh, weight_file=str(wp))
    folder = CD/'native40_terminal_v1'
    env['original_control_checks'](admit, env, folder, native, refs)
    admit.result['baseline_seals'] = seals
    return folder, native, scored, refs


def acquisition_admission(args, out, result, guard):
    admit = Admission(result, guard)
    sources = {**PINS, Path(__file__).resolve(): args.source_sha256,
        HERE/'SCORE_PROTOCOL.md': args.protocol_sha256, HERE/'watch_score.py': args.watcher_sha256,
        HERE/'SCORE_WATCH_SPEC.md': args.watch_spec_sha256, HERE/'native.py': args.native_source_sha256,
        HERE/'NATIVE_PROTOCOL.md': args.native_protocol_sha256, HERE/'watch_native.py': args.native_watch_source_sha256,
        HERE/'NATIVE_WATCH_SPEC.md': args.native_watch_spec_sha256}
    for path, digest in sources.items():
        admit.pin(path, digest)
    env, proof = helpers(admit); result['helper_source_proof'] = proof
    bf, base, oldscore, refs = baseline(admit, env)
    build, _, bs = admit.chain(HERE/'build_v1', args.build_sha256, inventory=True)
    runtime, _, rs = admit.chain(HERE/'runtime_v1', args.runtime_sha256, limit=300, inventory=True)
    require(build['schema'] == 'current_comp_aligned_push_d6_build_v1' and build['candidate_sha256'] == args.candidate_sha256
        and build['proof']['parent_sha256'] == CD_SHA and build['proof']['existing_complete_initialization_flops'] == INITIALIZATION
        and build['proof']['additional_initialization_over_CD'] == 0 and all(build['proof'][k] is True for k in
            ('whole_source_inverse_verified', 'only_final_dispatch_predicate_changed', 'heads_and_physics_byte_unchanged',
             'same_original_backing_arena', 'no_read_or_square_change', 'no_padding', 'no_rank_or_policy_tuning'))
        and build['estimator_imported'] is False and build['truth_accessed'] is False, 'Exact fixed source-only policy build')
    require(runtime['schema'] == 'current_comp_aligned_push_d6_runtime_v1' and runtime['candidate_sha256'] == args.candidate_sha256
        and runtime['build_seals'] == bs and runtime['ranks'] == [35, 35] and runtime['full_model_calls'] == 2
        and runtime['initialization_flops'] == INITIALIZATION and runtime['truth_arrays_accessed'] is False
        and runtime['elapsed_s'] < 300 and runtime['maximum_combined_rss_bytes'] < 8_000_000_000
        and runtime['maximum_interval_s'] <= 1 and runtime['maximum_swap_growth_bytes'] == 0
        and all(runtime.get(k) is True for k in ('runtime_falsifier_passed', 'hashes_unchanged', 'sources_unchanged',
            'whole_owned_tree_retired', 'observational_groups_retired', 'cleanup_passed', 'actual_completed_samples',
            'repeat_byte_exact')), 'Actual successful unchanged-candidate runtime2 prerequisite')
    nf, af, wf = (HERE/name for name in ('native_terminal_v1', 'native_terminal_admission_v1', 'native_watch_v1'))
    native, end, ns = admit.chain(nf, args.native_sha256, limit=600, inventory=True)
    admission, _, ads = admit.chain(af, args.admission_sha256, limit=60, inventory=True)
    watch, _, ws = admit.chain(wf, args.native_watch_sha256, limit=600, inventory=True)
    context_sha = native['admission_context_sha256']; cp = af/'context.json'
    require(cp.stat().st_size < 4_000_000, 'Bounded retired-admission context')
    admit.pin(cp, context_sha); context = read(cp); admit.inventory(context)
    require(admission['schema'] == 'current_comp_aligned_push_d6_native_admission_v1'
        and admission['context_sha256'] == context_sha and admission['elapsed_s'] < 60
        and admission['peak_rss_bytes'] < 2_000_000_000 and admission['admission_pid'] == context['admission_pid']
        and admission['sources_unchanged'] is True and admission['hashes_unchanged'] is True
        and context['schema'] == 'current_comp_aligned_push_d6_native_context_v1'
        and context['role'] == 'terminal' and context['ranks'] == list(RANKS)
        and context['candidate_sha256'] == native['candidate_sha256'] == args.candidate_sha256
        and context['build_sha256'] == args.build_sha256 and context['runtime_sha256'] == args.runtime_sha256
        and context['builder_sha256'] == PINS[HERE/'build.py'] and context['source_admission_complete'] is True
        and context['runtime_prerequisite_qualified'] is True and context['truth_accessed'] is False
        and context['numerical_imports'] is False, 'Separate actual retired stdlib admission and fixed source lineage')
    require(native['schema'] == 'current_comp_aligned_push_d6_native_v1' and native['stage'] == 'native'
        and native['role'] == 'terminal' and native['ranks'] == list(RANKS)
        and native['full_model_calls'] == native['retired_call_count'] == len(native['trials']) == 5
        and native['initialization_flops'] == INITIALIZATION and native['truth_accessed'] is False
        and native['score_measured'] is False and all(native.get(k) is True for k in
            ('own_complete_bills', 'all16_domains_qualified', 'all47_selectors_qualified', 'initialization_once',
             'repeat_output_domain_byte_exact', 'additional_bindings_paid_once', 'same_existing_arena_qualified',
             'dispatch_tariffs_qualified', 'all_eligible_push_dispatches_qualified', 'runtime_prerequisite_qualified',
             'sources_unchanged', 'hashes_unchanged', 'all_outputs_and_bills_sealed_before_scoring')),
        'Complete APUSH five-call own bills, all16/47, initialization, ordered generic dispatch qualification')
    require(watch['schema'] == 'current_comp_aligned_push_d6_native_watch_v1'
        and all(watch.get(k) is True for k in ('hashes_unchanged', 'sources_unchanged', 'whole_owned_tree_retired',
            'observational_groups_retired', 'cleanup_passed', 'actual_completed_samples'))
        and watch['full_model_calls'] == watch['retired_call_count'] == 5
        and watch['retired_calls'] == [dict(call=i, rank=r) for i, r in enumerate((*RANKS, 37))]
        and watch['elapsed_s'] < 600 and watch['maximum_admission_rss_bytes'] < 2_000_000_000
        and watch['maximum_worker_rss_bytes'] < 8_000_000_000
        and watch['maximum_combined_rss_bytes'] < 8_000_000_000 and watch['maximum_interval_s'] <= 1
        and watch['maximum_swap_growth_bytes'] == 0 and all(watch['child_'+k] == v for k, v in ns.items())
        and watch['context_sha256'] == context_sha and watch['admission_seals'] == ads
        and watch['admission_producer_reaped'] is True and watch['all5_complete'] is True
        and watch['candidate_sha256'] == args.candidate_sha256 and watch['truth_arrays_accessed'] is False
        and [p['phase'] for p in watch['phases']] == ['admission', 'worker']
        and all(p['exited'] is True for p in watch['phases'])
        and watch['phases'][0]['elapsed_through_verification_s'] < 60,
        'Actual external native-watch success, full child/admission seals and complete resource bounds')
    reviewed = {str(path): digest for path, digest in sources.items() if path.parent == HERE and path.name in
        ('native.py', 'NATIVE_PROTOCOL.md', 'watch_native.py', 'NATIVE_WATCH_SPEC.md')}
    require(all(context['source_bindings'].get(p) == h and watch['source_bindings'].get(p) == h
        for p, h in reviewed.items()), 'Same four reviewed native owners in actual context/watch')
    for rank, reference in refs.items():
        actual = context['reference_by_rank'][str(rank)]
        require(actual['trial'] == reference['trial'] and actual['weight_file_sha256'] == reference['weight_file_sha256']
            and actual['diagnostics'] == base['diagnostic_records'][RANKS.index(rank)],
            'APUSH was admitted against the exact retained CD first-four packets')
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'No numerical admission imports')
    spec = importlib.util.spec_from_file_location('apush_saved_metadata_verifier', HERE/'native.py')
    owner = importlib.util.module_from_spec(spec); sys.modules[spec.name] = owner; spec.loader.exec_module(owner)
    na = SimpleNamespace(role='terminal', source_sha256=args.native_source_sha256,
        protocol_sha256=args.native_protocol_sha256, watcher_sha256=args.native_watch_source_sha256,
        watch_spec_sha256=args.native_watch_spec_sha256, context_sha256=context_sha,
        candidate_sha256=args.candidate_sha256, build_sha256=args.build_sha256, runtime_sha256=args.runtime_sha256,
        builder_sha256=PINS[HERE/'build.py'])
    context['reference_by_rank'] = {int(k): v for k, v in context['reference_by_rank'].items()}
    require(owner.read_context(na, context_sha, guard) == context, 'Exact lightweight native context revalidation')
    owner.verify_acquisition(native, end, args.native_sha256, na, context, guard)
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'Native verifier is stdlib only')
    result.update(complete_APUSH_acquisition_reverified=True, all_five_new_outputs_verified=True,
        all_five_domains_and_selectors_reverified=True, own_complete_bills=True,
        runtime_prerequisite_qualified=True, frozen_geometry_price_used_for_scoring=False)
    lineage = dict(terminal=dict(build=bs, runtime=rs, admission=ads, native=ns, watch=ws, context_sha256=context_sha))
    return admit, env, bf, base, oldscore, refs, {'terminal': (nf, native)}, lineage


def arithmetic_source():
    """Exact JOINT/CD saved-score arithmetic; only admission, labels and one arm differ."""
    require(sha(JOINT) == PINS[JOINT], 'Frozen arithmetic owner')
    original = extract(JOINT, 'score')
    cut = original.index('    require(not any(k in sys.modules')
    prefix = original[:cut]
    replacement = ('def score(args, out, result, guard):\n'
        '    admit, env, bf, base, oldscore, refs, acquisitions, lineage = acquisition_admission(args, out, result, guard)\n')
    changes = [(prefix, replacement, 1), ("'COMP'", "'CD'", original.count("'COMP'")),
        ('all_ten_new_outputs_verified=True', 'all_five_new_outputs_verified=True', 1),
        ("oldscore['records'][call]['per_role']['composition']", "oldscore['records'][call]['per_role']['terminal']", 1),
        ("(('joint', 'CD'), ('p_only', 'CD'), ('joint', 'p_only'))", "(('terminal', 'CD'),)", 2),
        ('Same original1e9 truth bytes as retained COMP scores', 'Same original1e9 truth bytes as retained CD scores', 1),
        ('Exact four-row original COMP score replay', 'Exact four-row original CD native40 score replay', 1),
        ("        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],\n"
         "        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],\n"
         "        exact_COMP_score_replay=True, no_native_gradient_claim=True,",
         "        aligned_push_component=comparisons['terminal_vs_CD']['gate_passed'],\n"
         "        advances_to_native40=comparisons['terminal_vs_CD']['gate_passed'],\n"
         "        exact_CD_score_replay=True, no_native_gradient_claim=True,", 1)]
    derived = inverse_adapter(original, changes)
    proof = dict(owner=str(JOINT), owner_sha256=PINS[JOINT], original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_function_sha256=hashlib.sha256(derived.encode()).hexdigest(), full_function_inverse_verified=True,
        original_FP64_per_network_arithmetic_unchanged=True, original_mean_product_aggregation_unchanged=True,
        original_strict_adjusted_plus_three_wins_unchanged=True, direct_sealed_baseline_not_ancestor_reconstruction=True)
    return derived, proof


def main():
    global GUARD
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    for name in SHA_NAMES:
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}', getattr(args, name.replace('-', '_')+'_sha256') or '') for name in SHA_NAMES),
        'All fourteen reviewed source/future-receipt SHA arguments are required')
    require(sys.prefix == str(P2/'data_gate/.venv') and sys.dont_write_bytecode and __debug__, 'Official offline interpreter, -B, assertions')
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == '1' for k in
        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')),
        'Nice10 and six serial BLAS controls')
    out = HERE/'score_v1'; require(not out.exists(), 'Exclusive never-executed score output'); out.mkdir()
    began = time.monotonic()
    result = dict(schema=SCHEMA, stage='score', passed=False, completed=False, integrity_passed=False,
        source_bindings={}, input_bindings={}, artifacts={}, truth_bindings={}, requested_seals=vars(args),
        candidate_sha256=args.candidate_sha256, full_model_calls=0, parameter_updates=0, no_model_imported=True,
        training_occurred=False, truth_accessed=False, repeat_excluded=True, submission_authorized=False,
        uploaded=False, user_explicit_confirmation_bank_authorized=False, automatic_next_stage=False,
        production_five_qualified=False, new_independent14_qualified=False)
    write(out/'started.json', dict(schema=SCHEMA, pid=os.getpid(), requested_seals=vars(args)))
    def guard(*ignored):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(time.monotonic()-began < 120 and rss < 1_000_000_000, 'Whole120s/1GB including admission, IO and sealing')
        return rss
    GUARD = guard; signal.signal(signal.SIGALRM, guard); signal.setitimer(signal.ITIMER_REAL, .2, .2)
    try:
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'No pre-admission numerical imports')
        body, proof = arithmetic_source(); result['scoring_source_proof'] = proof
        env = dict(globals()); exec(compile(body, str(JOINT)+':fixed-APUSH-CHECK4', 'exec'), env)
        env['score'](args, out, result, guard); result['peak_rss_bytes'] = guard()
    except BaseException:
        result.update(passed=False, completed=False, integrity_passed=False, error=traceback.format_exc())
        # Failure publication has no positive authority and cannot be blocked by the expired success guard.
        signal.setitimer(signal.ITIMER_REAL, 0); GUARD = lambda: None
    result['elapsed_s'] = time.monotonic()-began
    write(out/'result.json', result); rh = sha(out/'result.json')
    ok = result['passed'] and time.monotonic()-began < 120
    write(out/'completion.json', dict(passed=ok, completed=ok, result_sha256=rh)); eh = sha(out/'completion.json')
    elapsed = time.monotonic()-began; ok = ok and elapsed < 120
    write(out/'finalization.json', dict(passed=ok, completed=ok, result_sha256=rh, completion_sha256=eh, elapsed_s=elapsed))
    signal.setitimer(signal.ITIMER_REAL, 0); GUARD = lambda: None
    print(json.dumps({k: result.get(k) for k in ('passed', 'summary', 'comparisons', 'advances_to_native40', 'error')}, allow_nan=False), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
