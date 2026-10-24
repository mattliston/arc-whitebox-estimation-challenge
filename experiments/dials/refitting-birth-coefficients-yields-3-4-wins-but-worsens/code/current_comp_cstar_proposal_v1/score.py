"""One sealed coefficient-only CHECK4; exact original saved-output arithmetic."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_joint_p_node_v1/score.py'
OWNER_SHA = '7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e'
MODEL_OWNER_SHA = 'f5c26f04750fa147ef897484ddc3a152ad744e6170b7542e8046af3193a79656'
PROPOSAL_PROTOCOL_SHA = '0a1db21c3a9f6f0a268ae8c47c1f4ad1c6b16ff6a0438aaeb2278dfaa6e97359'


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _function(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    _require(len(nodes) == 1, 'Unique original scorer function '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def _cstar_admission(g, args, arm, admit, env, refs, builder):
    """Actual CPROP seals/metadata only; no estimator/model/target decoding."""
    require, read, sha = g['require'], g['read'], g['sha']
    here, comp, ranks, ns = g['HERE'], g['COMP'], g['RANKS'], g['SimpleNamespace']
    require(arm == 'terminal', 'Only the fixed sole coefficients endpoint')
    get = lambda name: getattr(args, 'terminal_'+name+'_sha256')
    bf = here/'build_v1'
    build, _, build_seals = admit.chain(bf, get('build'), limit=120)
    built = builder.verify_build(get('build'), get('candidate'), args.builder_sha256,
                                args.build_protocol_sha256, tick=admit.guard)
    require(built['build'] == build, 'Same fully revalidated sealed build and prerequisite lineage')
    admit.bind(built['source_bindings']); admit.bind(built['input_bindings'])
    candidate = dict(path=str(bf/'estimator.py'), sha256=get('candidate'))
    parent = dict(path=str(comp/'build_v1/terminal_estimator.py'), sha256=g['PARENT_SHA'])
    require(build['schema'] == 'current_comp_cstar_proposal_build_v1' and build['role'] == 'terminal'
        and build['candidate'] == candidate and build['parent'] == parent
        and built['candidates'] == dict(parent=parent, terminal=candidate)
        and build['hashes_unchanged'] and build['sources_unchanged']
        and build['build_watch_required_before_native'] and build['parameter_updates'] == 0
        and build['estimator_executed'] is False and build['truth_arrays_accessed'] is False
        and build['refit_performed'] is False and build['native_gradient_qualified'] is False
        and build['CS_failure_preserved'] and 0 < build['peak_rss_bytes'] < 2_000_000_000
        and all(build['proof'].get(k) is True for k in ('literal_inverse_byte_exact',
            'literal_roundtrip_byte_exact', 'full_shape_FP32_finite', 'all_non_cstar_source_byte_exact',
            'P_A_source_byte_exact', 'frozen_nonP_and_inactive66_byte_exact', 'gaussian_row0_byte_exact',
            'original_runtime_getter_and_physics_unchanged'))
        and build['proof']['changed_value_names'] == ['_BD_COEFFICIENT_LITERAL']
        and build['proof']['new_inference_operations_at_fixed_geometry'] == 0
        and build['proof']['actual_candidate_masks_and_complete_cost_still_required'],
        'Only the actual sealed FP32 cstar literal, exact original COMP P/A/physics')

    nf, wf, ap = (here/name for name in
        ('native_terminal_v1', 'native_terminal_watch_v1', 'native_terminal_admission_v1'))
    native, end, native_seals = admit.chain(nf, get('native'))
    # New watch has actual elapsed_s finalization; never apply an old watch schema.
    watch, _, watch_seals = admit.chain(wf, get('watch'))
    context_sha = native['admission_context_sha256']; admit.pin(ap/'context.json', context_sha)
    require((ap/'context.json').stat().st_size < 2*1024**2, 'Exact bounded retired metadata context')
    context = read(ap/'context.json')
    ar_sha = native['source_bindings'].get(str(ap/'result.json'), native.get('input_bindings', {}).get(str(ap/'result.json')))
    admission, _, admission_seals = admit.chain(ap, ar_sha, limit=60)
    require(admission['schema'] == context['schema'] == 'current_comp_cstar_proposal_native_admission_v1'
        and admission['role'] == context['role'] == native['role'] == native['arm'] == 'terminal'
        and admission['context_sha256'] == context_sha and admission['sources_unchanged']
        and admission['hashes_unchanged'] and admission['numerical_imports'] is False
        and admission['truth_accessed'] is False and 0 <= admission['elapsed_s'] < 60
        and 0 < admission['peak_rss_bytes'] < 2_000_000_000
        and type(admission['admission_pid']) is int and admission['admission_pid'] == context['admission_pid']
        and context['ranks'] == list(ranks) and context['source_admission_complete']
        and context['numerical_imports'] is False and context['truth_accessed'] is False
        and context['candidate'] == candidate['path'] and context['candidate_sha256'] == candidate['sha256']
        and context['build_sha256'] == native['build_sha256'] == get('build')
        and context['builder_sha256'] == native['builder_sha256'] == args.builder_sha256
        and context['build_watch_sha256'] == native['build_watch_sha256'] == args.build_watch_sha256
        and native['full_admission_in_retired_process'] and native['no_fit'] and not native['score_measured']
        and native['full_model_calls'] == native['retired_call_count'] == 5
        and native['proof'] == context['proof'], 'Same source-admission/candidate and actual five-call lifetime')
    admit.inventory(context)
    owned = {str(here/'native.py'): args.native_source_sha256,
        str(here/'watch_native.py'): args.native_watch_source_sha256,
        str(here/'NATIVE_PROTOCOL.md'): args.native_protocol_sha256,
        str(here/'NATIVE_WATCH_SPEC.md'): args.native_watch_spec_sha256,
        str(here/'build.py'): args.builder_sha256, str(here/'BUILD_PROTOCOL.md'): args.build_protocol_sha256,
        str(here/'watch_build.py'): args.build_watch_source_sha256,
        str(here/'BUILD_WATCH_SPEC.md'): args.build_watch_spec_sha256,
        str(here/'model.py'): MODEL_OWNER_SHA}
    for record in (native, watch, context):
        require(all(record['source_bindings'].get(p) == h for p, h in owned.items()),
            'Every current native/build/protocol/supervisor/model owner is explicitly bound')
    require(watch['schema'] == 'current_comp_cstar_proposal_native_watch_v1'
        and watch['role'] == 'terminal' and watch['candidate_sha256'] == get('candidate')
        and watch['build_sha256'] == get('build') and watch['build_watch_sha256'] == args.build_watch_sha256
        and watch['admission_context_sha256'] == context_sha and watch['admission_result_sha256'] == ar_sha
        and watch['input_bindings'][str(ap/'context.json')] == context_sha
        and watch['full_model_calls'] == watch['retired_calls'] == watch['completed_bill_count'] == 5
        and all(watch.get(k) is True for k in ('hashes_unchanged', 'sources_unchanged',
            'child_execution_completed', 'whole_owned_tree_retired', 'cleanup_passed',
            'observational_groups_retired', 'actual_completed_samples', 'foreign_numerical_workers_checked',
            'all30_artifacts_rehashed', 'all30_artifacts_semantically_verified'))
        and watch['numerical_imports'] is False and watch['truth_arrays_accessed'] is False
        and watch['score_measured'] is False and watch['whole_cap_s'] == 600
        and watch['admission_cap_s'] == 60 and watch['per_call_cap_s'] == watch['progress_cap_s'] == 120
        and all(watch['child_'+k] == v for k, v in native_seals.items())
        and 0 <= watch['elapsed_s'] < 600 and 0 < watch['maximum_interval_s'] <= 1
        and 0 < watch['maximum_combined_rss_bytes'] < 8_000_000_000
        and watch['maximum_swap_growth_bytes'] == 0, 'Actual complete new600s/8GB/zero-swap independent watch')
    phases = watch['phase_records']
    require(len(phases) == 2 and [p['phase'] for p in phases] == ['admission', 'native']
        and phases[0]['pid'] == admission['admission_pid'] and phases[0]['pid'] != phases[1]['pid']
        and all(p['whole_owned_tree_retired'] and p['exit_code'] == 0 for p in phases)
        and 0 <= phases[0]['elapsed_s'] < 60 and 0 <= phases[1]['elapsed_s'] < 600,
        'Separately reaped source child before the one numerical child')
    for phase in phases:
        try:
            g['os'].kill(phase['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Scoring requires every recorded producer PID gone')
    intervals = watch['completed_sample_intervals_s']
    require(intervals and len(intervals) == watch['observation_samples']
        and max(intervals) == watch['maximum_interval_s']
        and all(g['math'].isfinite(x) and 0 <= x <= 1 for x in intervals), 'Actual completed-sample history')
    for name in ('samples.jsonl', 'admission.log', 'native.log', 'started.json'):
        admit.pin(wf/name, watch['artifacts'][str(wf/name)])
    samples = [json.loads(line) for line in (wf/'samples.jsonl').read_text().splitlines()]
    require(len(samples) == len(intervals) and samples[-1]['retired_calls'] == 5
        and g['math'].isfinite(watch['initial_swap_bytes']) and watch['initial_swap_bytes'] >= 0
        and max(x['combined_rss_bytes'] for x in samples) == watch['maximum_combined_rss_bytes']
        and all(x['phase'] in ('admission', 'native') and 0 <= x['swap_bytes'] <= watch['initial_swap_bytes']
            and 0 < x['combined_rss_bytes'] < (2_000_000_000 if x['phase'] == 'admission' else 8_000_000_000)
            and 0 <= x['elapsed_s'] < 600 and 0 <= x['retired_calls'] <= 5 for x in samples)
        and all(a['elapsed_s'] <= b['elapsed_s'] and a['retired_calls'] <= b['retired_calls']
            for a, b in zip(samples, samples[1:])), 'Corroborated phase-specific combined RSS and zero-added-swap history')
    require(native['versions'] == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'),
        'Original official native runtime')
    for rank in ranks:
        prior = dict(refs[rank]); prior.pop('weight_file')
        require(context['reference_by_rank'][str(rank)] == prior, 'Same exact COMP control and physical weight bytes')

    spec = g['importlib'].util.spec_from_file_location('cprop_score_native_metadata_owner', here/'native.py')
    owner = g['importlib'].util.module_from_spec(spec); spec.loader.exec_module(owner)
    for path, digest in owner.OWNERS.items():
        admit.pin(path, digest)
    authority = ns(source_sha256=args.native_source_sha256, model_sha256=MODEL_OWNER_SHA,
        protocol_sha256=args.native_protocol_sha256, watcher_sha256=args.native_watch_source_sha256,
        watch_spec_sha256=args.native_watch_spec_sha256, builder_sha256=args.builder_sha256,
        build_protocol_sha256=args.build_protocol_sha256, build_sha256=get('build'),
        build_watch_sha256=args.build_watch_sha256, build_watcher_sha256=args.build_watch_source_sha256,
        build_watch_spec_sha256=args.build_watch_spec_sha256, candidate_sha256=get('candidate'))
    build_watch_inputs = owner.build_watch_admission(authority, built); admit.bind(build_watch_inputs)
    require(all(context['input_bindings'].get(p) == h for p, h in build_watch_inputs.items()),
        'Exact build-watch3 chain was already admitted before the native worker')
    control_inputs = {}
    owner.control_watch_admission(read(comp/'native_terminal_v2/result.json'),
        read(comp/'native_terminal_v2/completion.json'), control_inputs)
    admit.bind(control_inputs)
    n = owner.module(comp/'native.py', 'cprop_score_original_observer_source')
    _, _, _, paid_proof = owner.prepare(n)
    require(paid_proof == context['proof'] == native['proof']
        and context['native_adapter_source_inverse'] == native['native_adapter_source_inverse'] == owner.NATIVE_ADAPTER_PROOF,
        'Same exact original COMP observer/paid-loop inverses, no participant arithmetic additions')
    verifier_proof = owner.verify_native(native, end, get('native'), authority, context, admit.guard)
    require(verifier_proof == watch['complete_artifact_verifier_proof']
        and native['all_outputs_and_bills_sealed_before_scoring'] and native['cstar_materialization_owned']
        and native['no_D6_composition'] and native['native_gradient_claim'] is False,
        'All30 own files semantically reverified, with both cstar getters paid inside the original prefix')
    admit.result.update(complete_cstar_acquisition_reverified=True, source_build_watch_qualified=True,
        all_five_domains_and_selectors_reverified=True, no_model_imported=True, CS_failure_preserved=True,
        candidate_sha256=get('candidate'), build_sha256=get('build'), build_watch_sha256=args.build_watch_sha256,
        native_result_sha256=get('native'), native_watch_result_sha256=get('watch'))
    bw = here/'build_watch_v1'
    build_watch_seals = {name.removesuffix('.json')+'_sha256': build_watch_inputs[str(bw/name)]
        for name in ('result.json', 'completion.json', 'finalization.json')}
    return nf, native, dict(build=build_seals, build_watch=build_watch_seals,
        admission=admission_seals, native=native_seals, watch=watch_seals, context_sha256=context_sha)


def derived_source():
    _require(_sha(OWNER) == OWNER_SHA, 'Exact frozen successful CHECK4 arithmetic owner')
    original = OWNER.read_text()
    old_admission = _function(original, 'arm_admission')
    old_local_pins = """    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '4ef8567620f6b2efb2890fda32eb029082fc1a07e084ab2f37ec35b4d29e4118',
    HERE/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',"""
    new_local_pins = """    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': '"""+PROPOSAL_PROTOCOL_SHA+"""',
    NEW/'current_comp_joint_p_node_v1/score.py': '"""+OWNER_SHA+"""',"""
    changes = (
        ('"""Fixed two-arm native CHECK4; no model, fitting or partial scoring."""',
         '"""Fixed sole cstar native CHECK4; no model, fitting or partial scoring."""', 1),
        ("('p_only', 'joint'), 2**41", "('terminal',), 2**41", 1),
        ("MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'",
         "MODEL_SHA = '"+MODEL_OWNER_SHA+"'", 1),
        ("SCHEMA = 'current_comp_joint_p_node_score_v1'", "SCHEMA = 'current_comp_cstar_proposal_score_v1'", 1),
        (old_local_pins, new_local_pins, 1),
        (old_admission, "def arm_admission(args, arm, admit, env, refs, builder):\n"
            "    return cstar_admission(globals(), args, arm, admit, env, refs, builder)\n", 1),
        ("HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256}",
         "HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256,\n"
         "        HERE/'NATIVE_WATCH_SPEC.md': args.native_watch_spec_sha256,\n"
         "        HERE/'watch_build.py': args.build_watch_source_sha256,\n"
         "        HERE/'BUILD_WATCH_SPEC.md': args.build_watch_spec_sha256,\n"
         "        HERE/'watch_score.py': args.watcher_sha256, HERE/'SCORE_WATCH_SPEC.md': args.watch_spec_sha256}", 1),
        ("env, proof = helpers(admit); result['helper_source_proof'] = proof",
         "env, proof = helpers(admit); result['helper_source_proof'] = proof\n"
         "    result['endpoint_source_inverse_proof'] = ENDPOINT_SOURCE_PROOF", 1),
        ('all_ten_new_outputs_verified', 'all_five_new_outputs_verified', 1),
        ("(('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only'))", "(('terminal', 'COMP'),)", 2),
        ("        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],\n"
         "        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],",
         "        cstar_component=comparisons['terminal_vs_COMP']['gate_passed'],\n"
         "        advances_to_native40=comparisons['terminal_vs_COMP']['gate_passed'],", 1),
        ("'native-watch-source', 'native-protocol', 'builder', 'build-protocol'",
         "'native-watch-source', 'native-protocol', 'native-watch-spec', 'builder', 'build-protocol',\n"
         "        'build-watch', 'build-watch-source', 'build-watch-spec', 'watcher', 'watch-spec'", 1),
        ("for arm in ('p-only', 'joint') for kind", "for arm in ('terminal',) for kind", 1),
        ("'joint_learning_lead', 'p_only_component'", "'cstar_component', 'advances_to_native40'", 1),
    )
    changed = original
    for before, after, count in changes:
        _require(changed.count(before) == count, 'Exact sole-cstar metadata span: '+before[:100])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact sole-cstar inverse span: '+after[:100])
        inverse = inverse.replace(after, before)
    _require(inverse == original, 'Whole original CHECK4 scorer byte inverse')
    ast.parse(changed)
    proof = dict(owner=str(OWNER), owner_sha256=OWNER_SHA, full_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_functions=['Admission', 'baseline', 'original_control_checks', 'helpers', 'lifecycle_owner'],
        unchanged_FP64_MSE_and_own_multiplier=True, unchanged_Pareto_definition=True,
        changed_scope='one sole cstar endpoint; actual current3-seal admission; strict mean gain and3/4 wins')
    return changed, proof


cstar_admission = _cstar_admission
_source, ENDPOINT_SOURCE_PROOF = derived_source()
# Normal import/runpy exposes the resolved helpers and main, but never executes a
# stage unless the unchanged original --run entrypoint is explicitly invoked.
exec(compile(_source, str(OWNER)+':one-cstar-terminal', 'exec'), globals())
