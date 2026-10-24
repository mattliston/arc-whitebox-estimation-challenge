"""One COMP/partial-D6 CHECK4: source-inverse saved-output scoring only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_joint_p_node_v1/score.py'
OWNER_SHA = '7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e'


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


def _composition_admission(g, args, arm, admit, env, refs, builder):
    """New metadata joins; actual CD verifier and old scoring arithmetic stay exact."""
    require, read, sha = g['require'], g['read'], g['sha']
    here, comp, ranks, ns = g['HERE'], g['COMP'], g['RANKS'], g['SimpleNamespace']
    require(arm == 'terminal', 'Exactly one sealed fixed composition')
    get = lambda key: getattr(args, 'terminal_'+key+'_sha256')
    build_folder = here/'build_v1'
    build, _, bs = admit.chain(build_folder, get('build'), limit=60)
    built = builder.verify_build(get('build'), get('candidate'), args.builder_sha256, args.build_protocol_sha256)
    require(built['build'] == build, 'Same source-only build independently rederived without estimator import')
    admit.bind(built['source_bindings']); admit.bind(built['input_bindings'])
    candidate = dict(path=str(build_folder/'estimator.py'), sha256=get('candidate'))
    parent = dict(path=str(comp/'build_v1/terminal_estimator.py'), sha256=g['PARENT_SHA'])
    require(build['schema'] == 'current_comp_partial_d6_build_v1' and build['role'] == 'terminal'
        and build['candidates']['terminal'] == candidate and build['candidates']['parent'] == parent
        and build['candidates']['null'] == dict(path=str(build_folder/'null_estimator.py'), sha256=g['PARENT_SHA'])
        and build['candidate_sha256'] == get('candidate') and build['null_source_byte_exact']
        and all(build['proof'][key] for key in ('whole_source_inverse_verified', 'old_partial_inverse_to_P1_verified',
            'new_partial_inverse_to_COMP_verified', 'exact_three_span_composition', 'imports_byte_exact',
            'all_unrelated_source_bytes_unchanged', 'P_A_cstar_literal_bytes_unchanged',
            'cached_child_helpers_byte_exact', 'child_bypasses_dispatcher', 'kernel_class_product_byte_exact'))
        and build['proof']['dispatch_shape'] == [1024, 832, 8192]
        and build['proof']['additional_both_binding_flops'] == 561056000
        and build['proof']['original_arena_bytes'] == 1168900096
        and not build['proof']['runtime_layout_generation'] and not build['proof']['second_backing_arena'],
        'Exact retained three-span composition; no new parameter, shape or storage owner')

    nf, wf, ap = here/'native_terminal_v1', here/'native_terminal_watch_v1', here/'native_terminal_admission_v1'
    native, end, native_seals = admit.chain(nf, get('native'))
    watch, _, watch_seals = admit.chain(wf, get('watch'), watch=True)
    context_sha = native['admission_context_sha256']; admit.pin(ap/'context.json', context_sha)
    require((ap/'context.json').stat().st_size < 4_000_000, 'Original bounded source-admission context')
    context = read(ap/'context.json')
    ar_sha = native['source_bindings'].get(str(ap/'result.json'), native.get('input_bindings', {}).get(str(ap/'result.json')))
    admission, _, admission_seals = admit.chain(ap, ar_sha, limit=60)
    require(admission['role'] == context['role'] == native['role'] == 'terminal'
        and admission['context_sha256'] == context_sha and admission['sources_unchanged']
        and admission['numerical_imports'] is False and admission['truth_accessed'] is False
        and 0 <= admission['elapsed_s'] < 60 and 0 < admission['peak_rss_bytes'] < 2_000_000_000
        and type(admission['admission_pid']) is int and admission['admission_pid'] == context['admission_pid']
        and context['ranks'] == list(ranks) and context['source_admission_complete']
        and context['numerical_imports'] is False and context['truth_accessed'] is False
        and context['candidate'] == candidate['path'] and context['candidate_sha256'] == candidate['sha256']
        and context['build_sha256'] == native['build_sha256'] == get('build')
        and context['builder_sha256'] == native['builder_sha256'] == args.builder_sha256
        and context['build_protocol_sha256'] == args.build_protocol_sha256
        and context['build_watch_sha256'] == native['build_watch_sha256']
        and native['full_admission_in_retired_process'] and native['no_fit'] and not native['score_measured']
        and native['full_model_calls'] == native['retired_call_count'] == 5
        and native['proof'] == context['proof'], 'Same retired full admission and five-call candidate lifetime')
    admit.inventory(context)
    owned = {str(here/'native.py'): args.native_source_sha256,
        str(here/'watch_native.py'): args.native_watch_source_sha256,
        str(here/'NATIVE_PROTOCOL.md'): args.native_protocol_sha256,
        str(here/'NATIVE_WATCH_SPEC.md'): args.native_watch_spec_sha256,
        str(here/'build.py'): args.builder_sha256}
    for record in (native, watch, context):
        require(all(record['source_bindings'].get(path) == digest for path, digest in owned.items()),
            'Same explicitly reviewed native/builder/protocol/watch/spec owners')
    require(context['source_bindings'][str(here/'BUILD_PROTOCOL.md')] == args.build_protocol_sha256
        and watch['schema'] == 'current_comp_partial_d6_native_watch_v1'
        and watch['source_bindings'][str(ap/'context.json')] == context_sha
        and watch['retired_calls'] == 5 and watch['whole_cap_s'] == 600
        and watch['numerical_imports'] is False and watch['mps_slot_clear_attested']
        and all(watch['child_'+key] == value for key, value in native_seals.items())
        and all(watch['supervisor_source_proof'][key] for key in
            ('full_source_inverse_verified', 'resource_loop_and_finalization_unchanged', 'only_fixed_terminal5')),
        'Independent completed native supervisor owns every child seal and the admitted context')
    require(native['versions'] == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'),
        'Original official native runtime')
    baseline = read(comp/'native_terminal_v2/result.json')
    for call, rank in enumerate(ranks):
        prior = dict(refs[rank]); prior.pop('weight_file')
        prior['diagnostic'] = baseline['diagnostic_records'][call]
        require(context['reference_by_rank'][str(rank)] == prior, 'Same canonical COMP weight/output/domain control')

    for name in ('samples.jsonl', 'worker.log', 'started.json'):
        admit.pin(wf/name, sha(wf/name))
    samples = [json.loads(line) for line in (wf/'samples.jsonl').read_text().splitlines()]
    require(len(samples) == watch['observation_samples'] and samples and samples[-1]['retired_calls'] == 5
        and 0 < watch['maximum_interval_s'] <= 1 and 0 < watch['maximum_combined_rss_bytes'] < 8_000_000_000
        and g['math'].isfinite(watch['initial_swap_bytes']) and watch['initial_swap_bytes'] >= 0
        and max(x['combined_rss_bytes'] for x in samples) == watch['maximum_combined_rss_bytes']
        and all(0 <= x['swap_bytes'] <= watch['initial_swap_bytes'] and 0 < x['combined_rss_bytes'] < 8_000_000_000
            and 0 <= x['elapsed_s'] < 600 and 0 <= x['retired_calls'] <= 5 for x in samples)
        and all(a['elapsed_s'] <= b['elapsed_s'] and a['retired_calls'] <= b['retired_calls'] for a, b in zip(samples, samples[1:])),
        'Corroborated complete zero-added-swap native resource history')

    spec = g['importlib'].util.spec_from_file_location('comp_partial_d6_saved_metadata_owner', here/'native.py')
    owner = g['importlib'].util.module_from_spec(spec); spec.loader.exec_module(owner)
    for path, digest in owner.OWNERS.items():
        admit.pin(path, digest)
    build_watch_sources, build_watch_inputs = {}, {}
    owner.build_watch_admission(ns(builder_sha256=args.builder_sha256, build_protocol_sha256=args.build_protocol_sha256,
        build_sha256=get('build'), build_watch_sha256=context['build_watch_sha256']), build_watch_sources, build_watch_inputs)
    admit.bind(build_watch_sources); admit.bind(build_watch_inputs)
    for group, bindings in (('source_bindings', build_watch_sources), ('input_bindings', build_watch_inputs)):
        require(all(context[group].get(path) == digest for path, digest in bindings.items()),
            'Native admission already bound the full successful source-build supervision')

    # Exact current native metadata functions, with only stdlib function
    # providers in their globals. No COMP/PD legacy module is imported.
    metadata_source, metadata_proof = owner.metadata_sources()
    _, observer_proof = owner.observer_sources()
    _, lifecycle_proof = owner.lifecycle_source()
    proof = native['proof']
    require(proof['metadata_functions'] == metadata_proof and proof['outer_lifecycle'] == lifecycle_proof
        and all(proof[key] == value for key, value in observer_proof.items())
        and proof['original_paid_arithmetic_unchanged'] and proof['candidate_source_changes_owned_only_by_builder']
        and proof['paid_loop']['paid_loop_inverse_verified'] and proof['paid_loop']['observer_inverse_verified']
        and proof['paid_loop']['new_candidate_arithmetic'] is False
        and proof['paid_loop']['changed_paid_statements'] == 'one post-release stdout marker only',
        'Exact admitted CD observer/metadata/lifecycle inverses and preserved paid-loop proof')
    timings_path = g['NEW']/'birth_direction_fit_v1/native_v2.py'
    assignments = [node for node in ast.parse(timings_path.read_text()).body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TIMINGS']
    require(len(assignments) == 1, 'Unique original bill timing fields')
    timings = ast.literal_eval(assignments[0].value)
    require(timings == ('wall_s', 'context_wall_s', 'flopscope_backend_s', 'flopscope_overhead_s', 'native_residual_s'),
        'Exact source-owned complete bill timing inventory')

    def hash_tick(path, tick=None):
        digest = sha(path)
        if tick:
            tick()
        return digest

    def bind_tick(mapping, tick=None):
        admit.bind(mapping)
        if tick:
            tick()

    metadata_env = dict(env, HERE=here, OUTPUT=nf, VALID=ranks, NATIVE_SCHEMA=owner.NATIVE_SCHEMA,
        EXTRA_LAYOUTS=owner.EXTRA_LAYOUTS, EXTRA_BINDINGS=owner.EXTRA_BINDINGS,
        PRODUCT_FLOPS=owner.PRODUCT_FLOPS, SHAPE=owner.SHAPE, sha=hash_tick, bind=bind_tick,
        write=g['write'], base=lambda: ns(npy_bytes=env['npy_bytes'], legacy=lambda: ns(TIMINGS=timings)))
    exec(compile(metadata_source, str(here/'native.py')+':exact_saved_metadata', 'exec'), metadata_env)
    metadata_env['verify_acquisition'](native, end, get('native'), ns(context_sha256=context_sha), context, admit.guard)
    require(native['COMP_parameter_initialization_owned'] and native['all_outputs_and_bills_sealed_before_scoring']
        and native['initialization_accounting']['additional_binding_flops'] == 561056000
        and not native['initialization_accounting']['startup_subtracted_from_primary_bills']
        and not native['initialization_accounting']['old_initialization_value_assumed']
        and native['initialization_accounting']['nested_extra_binding_regions_not_double_counted']
        and native['initialization_accounting']['nested_prefix_materializations_not_double_counted'],
        'All own complete primary bills include actual initialization and both bindings exactly once')
    admit.result.update(complete_D6_acquisition_reverified=True, source_build_watch_qualified=True,
        all_five_domains_and_selectors_reverified=True, no_model_imported=True,
        old_max_read_timing_gate_still_failed=True, frozen_geometry_estimate_not_used_for_scoring=True)
    bw = here/'build_watch_v1'
    build_watch_seals = {name.removesuffix('.json')+'_sha256': build_watch_inputs[str(bw/name)]
        for name in ('result.json', 'completion.json', 'finalization.json')}
    return nf, native, dict(build=bs, build_watch=build_watch_seals, admission=admission_seals,
        native=native_seals, watch=watch_seals, context_sha256=context_sha)


def derived_source():
    _require(_sha(OWNER) == OWNER_SHA, 'Frozen complete CHECK4 scorer owner')
    original = OWNER.read_text()
    old_admission = _function(original, 'arm_admission')
    old_local_pins = """    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '4ef8567620f6b2efb2890fda32eb029082fc1a07e084ab2f37ec35b4d29e4118',
    HERE/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',"""
    new_local_pins = """    HERE/'DECISION.md': 'b594093e92a190014b534ed6ebc7e0688326ba7d830dac404fefd76d0c43827a',
    NEW/'current_comp_joint_p_node_v1/score.py': '"""+OWNER_SHA+"""',
    NEW/'current_comp_joint_p_node_v1/watch_native.py': 'd35ba7cbe406ae5e6200f8778b7a582bc6ea17336be8086b5e813e5a25dbe65e',
    NEW/'current_comp_receiver_birth_v1/watch_qualification.py': '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb',"""
    changes = (
        ('"""Fixed two-arm native CHECK4; no model, fitting or partial scoring."""',
         '"""Fixed one-terminal COMP/partial-D6 CHECK4; no model, fitting or partial scoring."""', 1),
        ("('p_only', 'joint'), 2**41", "('terminal',), 2**41", 1),
        ("MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'",
         'MODEL_SHA = None  # No model or trainable payload is an endpoint dependency.', 1),
        ("SCHEMA = 'current_comp_joint_p_node_score_v1'", "SCHEMA = 'current_comp_partial_d6_score_v1'", 1),
        (old_local_pins, new_local_pins, 1),
        (old_admission, "def arm_admission(args, arm, admit, env, refs, builder):\n"
            "    return composition_admission(globals(), args, arm, admit, env, refs, builder)\n", 1),
        ("HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256}",
         "HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256,\n"
         "        HERE/'NATIVE_WATCH_SPEC.md': args.native_watch_spec_sha256}", 1),
        ("env, proof = helpers(admit); result['helper_source_proof'] = proof",
         "env, proof = helpers(admit); result['helper_source_proof'] = proof\n"
         "    result['endpoint_source_inverse_proof'] = ENDPOINT_SOURCE_PROOF", 1),
        ('all_ten_new_outputs_verified', 'all_five_new_outputs_verified', 1),
        ("(('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only'))", "(('terminal', 'COMP'),)", 2),
        ("        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],\n"
         "        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],",
         "        partial_D6_component=comparisons['terminal_vs_COMP']['gate_passed'],\n"
         "        advances_to_native40=comparisons['terminal_vs_COMP']['gate_passed'],", 1),
        ("'native-watch-source', 'native-protocol', 'builder', 'build-protocol'",
         "'native-watch-source', 'native-protocol', 'builder', 'build-protocol', 'native-watch-spec'", 1),
        ("for arm in ('p-only', 'joint') for kind", "for arm in ('terminal',) for kind", 1),
        ("'joint_learning_lead', 'p_only_component'", "'partial_D6_component', 'advances_to_native40'", 1),
    )
    changed = original
    for before, after, count in changes:
        _require(changed.count(before) == count, 'Exact fixed-composition scorer source span: '+before[:100])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact fixed-composition scorer inverse span: '+after[:100])
        inverse = inverse.replace(after, before)
    _require(inverse == original, 'Whole frozen scorer byte inverse')
    ast.parse(changed)
    proof = dict(owner=str(OWNER), owner_sha256=OWNER_SHA, full_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_functions=['Admission', 'baseline', 'original_control_checks', 'lifecycle_owner', 'helpers'],
        unchanged_FP64_MSE_and_own_multiplier=True, unchanged_Pareto_definition=True,
        changed_scope='one fixed composition, exact current acquisition joins, strict gain and3/4 wins')
    return changed, proof


_source, _proof = derived_source()
exec(compile(_source, str(OWNER)+':one-comp-partial-d6-terminal', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'composition_admission': _composition_admission,
     'ENDPOINT_SOURCE_PROOF': _proof})
