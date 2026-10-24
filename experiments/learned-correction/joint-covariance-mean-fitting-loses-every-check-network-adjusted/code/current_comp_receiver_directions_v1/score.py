"""One receiver terminal CHECK4; inverse-owned saved-output arithmetic only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent/'current_comp_joint_p_node_v1'
OWNER = JOINT/'score.py'
OWNER_SHA = '7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e'
WATCH_OWNER_SHA = 'cca85002951bedef0891610c086a3984aaa6f887e4a772b8dc85c38701bfd071'
ZERO_RESULT_SHA = 'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625'
ZERO_END_SHA = '0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073'


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
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    _require(len(nodes) == 1, 'Unique original scorer function '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def _receiver_admission(g, args, arm, admit, env, refs, builder):
    """New metadata joins only; exact old acquisition/lifecycle validator remains."""
    require, read, sha = g['require'], g['read'], g['sha']
    here, comp, ranks = g['HERE'], g['COMP'], g['RANKS']
    require(arm == 'terminal', 'Exactly one sealed receiver terminal')

    def one(role, digests, expected_refs):
        build_hash, candidate_hash, native_hash, watch_hash = digests
        build_folder = here/f'build_{role}_v1'
        build, _, bs = admit.chain(build_folder, build_hash, limit=60)
        built = builder.verify_build(role, build_hash, candidate_hash, args.builder_sha256, args.build_protocol_sha256)
        require(built['build'] == build, 'Same complete receiver build independently source-reverified')
        admit.bind(built['source_bindings']); admit.bind(built['input_bindings'])
        require(build['schema'] == 'current_comp_receiver_directions_build_v1' and build['role'] == role
            and build['hashes_unchanged'] and build['sources_unchanged'] and build['peak_rss_bytes'] < 2_000_000_000
            and build['candidate'] == dict(path=str(build_folder/'estimator.py'), sha256=candidate_hash)
            and build['parent'] == dict(path=str(comp/'build_v1/terminal_estimator.py'), sha256=g['PARENT_SHA'])
            and build['source_bindings'][str(here/'build_v2.py')] == args.builder_sha256
            and build['source_bindings'][str(here/'BUILD_PROTOCOL_V2.md')] == args.build_protocol_sha256
            and all(build['proof'][key] for key in ('full_source_inverse_verified', 'physical_function_inverse_verified',
                'original_stored_pair_repair_once', 'zero_parameters_not_bypassed', 'original_P_A_cstar_literals_unchanged',
                'literal_roundtrip_byte_exact', 'zero_terminal_literal_only', 'full_shape_FP32_finite'))
            and build['receiver_literal']['dtype'] == 'float32' and build['receiver_literal']['shape'] == [64, 4],
            'Exact qualified256-value receiver build; no parent/circuit/refit substitution')
        nf, wf, ap = (here/f'native_{role}_v1', here/f'native_{role}_watch_v1', here/f'native_{role}_admission_v1')
        native, end, ns = admit.chain(nf, native_hash)
        watch, _, ws = admit.chain(wf, watch_hash, watch=True)
        context_sha = native['admission_context_sha256']; admit.pin(ap/'context.json', context_sha)
        context = read(ap/'context.json')
        ar_sha = native['source_bindings'].get(str(ap/'result.json'), native.get('input_bindings', {}).get(str(ap/'result.json')))
        ar, _, ads = admit.chain(ap, ar_sha, limit=60)
        count = len(expected_refs)+1
        require(ar['role'] == context['role'] == native['role'] == role and ar['context_sha256'] == context_sha
            and ar['sources_unchanged'] and not ar['truth_accessed'] and not ar['numerical_imports']
            and ar['elapsed_s'] < 60 and ar['peak_rss_bytes'] < 2_000_000_000
            and context['ranks'] == list(expected_refs) and context['source_admission_complete']
            and not context['truth_accessed'] and not context['numerical_imports']
            and context['candidate'] == build['candidate']['path'] and context['candidate_sha256'] == candidate_hash
            and context['build_sha256'] == native['build_sha256'] == build_hash
            and native['full_admission_in_retired_process'] and native['full_model_calls'] == native['retired_call_count'] == count
            and native['proof'] == context['proof'], 'Same retired full admission, fixed role and sole candidate')
        admit.inventory(context)
        owned = {str(here/'native.py'): args.native_source_sha256,
            str(here/'watch_native.py'): args.native_watch_source_sha256,
            str(here/'NATIVE_PROTOCOL.md'): args.native_protocol_sha256, str(here/'model.py'): g['MODEL_SHA']}
        for item in (native, watch, context):
            require(all(item['source_bindings'].get(path) == digest for path, digest in owned.items()),
                'Same reviewed native/model/protocol/watch source authority')
        require(context['source_bindings'][str(here/'build_v2.py')] == args.builder_sha256
            and context['source_bindings'][str(here/'BUILD_PROTOCOL_V2.md')] == args.build_protocol_sha256
            and watch['source_bindings'][str(ap/'context.json')] == context_sha
            and watch['retired_calls'] == count and all(watch['child_'+key] == value for key, value in ns.items()),
            'Independent watch owns every native seal and the retired source context')
        for rank, reference in expected_refs.items():
            prior = dict(reference); prior.pop('weight_file', None)
            require(context['reference_by_rank'][str(rank)] == prior, 'Same physical COMP output and diagnostic reference')
        require(native['versions'] == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'),
            'Original complete native runtime')
        for name in ('samples.jsonl', 'worker.log', 'started.json'):
            admit.pin(wf/name, sha(wf/name))
        samples = [json.loads(line) for line in (wf/'samples.jsonl').read_text().splitlines()]
        require(len(samples) == watch['observation_samples'] and samples and samples[-1]['retired_calls'] == count
            and max(item['combined_rss_bytes'] for item in samples) == watch['maximum_combined_rss_bytes']
            and all(item['swap_bytes'] <= watch['initial_swap_bytes'] and item['combined_rss_bytes'] < 8_000_000_000 for item in samples),
            'Corroborated completed native zero-swap observations')
        env.update(HERE=here, FIXED={role: tuple(expected_refs)}, CANDIDATES={role: candidate_hash},
            SCHEMA='current_comp_receiver_directions_native_v1', BUILD_SHA=build_hash,
            BUILDER_SHA=args.builder_sha256, PROTOCOL_SHA=args.native_protocol_sha256)
        env['verify_acquisition'](native, end, native_hash, role, args.native_source_sha256,
            args.native_watch_source_sha256, expected_refs)
        proof = native['proof']
        require(native['receiver_R_initialization_owned'] and all(proof[key] for key in (
            'receiver_observer_inverse_verified', 'nine_filtered_code_identities', 'original_paid_arithmetic_unchanged',
            'one_additional_getter_observed', 'getter_inside_existing_prefix_not_double_counted'))
            and proof['paid_loop']['paid_loop_inverse_verified'] and proof['paid_loop']['observer_inverse_verified']
            and proof['paid_loop']['changed_paid_statements'] == 'one post-release stdout marker only',
            'Original paid loop with only passive R ownership and post-release marker')
        for call, item in enumerate(native['diagnostic_records']):
            diagnostic = read(item['path']); startup = diagnostic['startup']; cold = call == 0
            receiver = [value for value in startup['materializations'] if value['name'] == '_bdr_parameters']
            require(len(receiver) == 1 and receiver[0]['inside_prefix'] and receiver[0]['literal_elements'] == 256
                and receiver[0]['empty_at_entry'] is cold and receiver[0]['flops'] == (512 if cold else 0)
                and startup['receiver_R_materialization_flops'] == receiver[0]['flops']
                and startup['prefix']['R_empty_at_entry'] is cold and startup['prefix']['R_ready_at_binary_entry']
                and startup['observed_startup_region_flops'] == startup['prefix']['flops']
                    +sum(value['flops'] for value in startup['bindings'])+startup['outside_prefix_materialization_flops'],
                'R512cold/0warm belongs inside prefix once, never added twice or deducted')
            reference = expected_refs[diagnostic['rank']]
            require(diagnostic['P1_reference']['path'] == reference['output']
                and diagnostic['P1_reference']['sha256'] == reference['output_sha256'], 'Historical key names actual COMP reference')
            if role == 'zero':
                dm = reference['diagnostic']; admit.pin(dm['path'], dm['sha256'])
                require(diagnostic['domains'] == read(dm['path'])['domains'], 'Null all16 domains and all47 selector bytes equal COMP')
        if role == 'zero':
            require(native['zero_all16_COMP_byte_exact'] and native['zero_all47_COMP_byte_exact']
                and build['receiver_literal']['raw_sha256'] == hashlib.sha256(bytes(1024)).hexdigest(),
                'Exact positive-zero live receiver and native COMP identity')
        return nf, native, context, build, dict(build=bs, admission=ads, native=ns, watch=ws, context_sha256=context_sha)

    baseline = read(comp/'native_terminal_v2/result.json')
    terminal_refs = {rank: dict(refs[rank], diagnostic=baseline['diagnostic_records'][call]) for call, rank in enumerate(ranks)}
    terminal_hashes = tuple(getattr(args, 'terminal_'+key+'_sha256') for key in ('build', 'candidate', 'native', 'watch'))
    nf, native, context, build, seals = one('terminal', terminal_hashes, terminal_refs)
    zero_folder = comp/'panel_v2/terminal'
    admit.pin(zero_folder/'result.json', ZERO_RESULT_SHA); admit.pin(zero_folder/'completion.json', ZERO_END_SHA)
    reference, reference_end = read(zero_folder/'result.json'), read(zero_folder/'completion.json')
    require(reference['passed'] and reference['completed'] and reference['hashes_unchanged'] and reference['sources_unchanged']
        and reference['candidate_sha256'] == g['PARENT_SHA'] and reference['own_complete_bills']
        and reference['all16_domains_qualified'] and reference['all47_selectors_qualified'] and not reference['truth_accessed']
        and reference_end['passed'] and reference_end['completed'] and reference_end['result_sha256'] == ZERO_RESULT_SHA
        and len(reference['trials']) == len(reference['input_preparation']) == len(reference['diagnostic_records']) == 15,
        'Exact completed COMP panel source for canonical zero35/100, no borrowed native bills')
    zero_refs = {}
    for call, rank in enumerate((35, 100)):
        trial, prepared, dm = reference['trials'][call], reference['input_preparation'][call], reference['diagnostic_records'][call]
        require(trial['rank'] == prepared['rank'] == rank and trial['call'] == prepared['call'] == call,
            'Exact canonical zero reference call/weight join')
        admit.pin(trial['output'], reference['artifacts'][trial['output']]); admit.pin(dm['path'], reference['artifacts'][dm['path']])
        require(trial['output_sha256'] == reference['artifacts'][trial['output']] and dm['sha256'] == reference['artifacts'][dm['path']],
            'Original COMP null output/domain bindings')
        zero_refs[rank] = dict(trial, weight_file_sha256=prepared['weight_file_sha256'], diagnostic=dm)
    dependency = context['input_bindings']
    zero_rp, zero_wp = here/'native_zero_v1/result.json', here/'native_zero_watch_v1/result.json'
    zero_digest, zero_watch_digest = dependency[str(zero_rp)], dependency[str(zero_wp)]
    admit.pin(zero_rp, zero_digest); admit.pin(zero_wp, zero_watch_digest)
    zero_record = read(zero_rp)
    zero_hashes = (zero_record['build_sha256'], zero_record['candidate_sha256'], zero_digest, zero_watch_digest)
    _, _, _, zero_build, zero_seals = one('zero', zero_hashes, zero_refs)
    for role_folder, role_seals in ((here/'native_zero_v1', zero_seals['native']),
                                   (here/'native_zero_watch_v1', zero_seals['watch'])):
        for name, digest in role_seals.items():
            path = role_folder/(name.removesuffix('_sha256')+'.json')
            require(dependency.get(str(path)) == digest, 'Terminal admission already owned every passed zero seal')
    require(build['prerequisites'] == zero_build['prerequisites']
        and build['qualification_owner_pins'] == zero_build['qualification_owner_pins']
        and build['proof']['zero_source_sha256'] == zero_hashes[1]
        and build['initial_P'] == zero_build['initial_P'] and build['initial_A'] == zero_build['initial_A']
        and build['cstar_raw_sha256'] == zero_build['cstar_raw_sha256'],
        'Same fresh qualification and frozen parent; terminal differs only in receiver literal')
    admit.result['zero_prerequisite_qualified'] = True
    return nf, native, dict(seals, zero=zero_seals)


def derived_source():
    _require(_sha(OWNER) == OWNER_SHA, 'Frozen complete CHECK4 scorer owner')
    original = OWNER.read_text()
    old_admission = _function(original, 'arm_admission')
    old_local_pins = """    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '4ef8567620f6b2efb2890fda32eb029082fc1a07e084ab2f37ec35b4d29e4118',
    HERE/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',"""
    new_local_pins = """    HERE/'MODEL_PROTOCOL.md': '22e15a727e3e71eb11a7d289ef38a6b121328c52c127a2d87968e341074eae02',
    HERE/'DECISION.md': 'ad9b923d6c322585257013b293237a28b1d77826a86fb4a412a21a9bd9118b8c',
    HERE/'TRAIN_PROTOCOL.md': '7c989ea707c9790d92793084569378983fad0861412d233153433e70ad012531',
    NEW/'current_comp_joint_p_node_v1/score.py': '"""+OWNER_SHA+"""',
    NEW/'current_comp_joint_p_node_v1/watch_score.py': '"""+WATCH_OWNER_SHA+"""',"""
    changes = (
        ('"""Fixed two-arm native CHECK4; no model, fitting or partial scoring."""',
         '"""Fixed one-terminal receiver CHECK4; no model, fitting or partial scoring."""', 1),
        ("('p_only', 'joint'), 2**41", "('terminal',), 2**41", 1),
        ("MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'",
         "MODEL_SHA = '773f6fa146ff6944fa43b125d98f6f643c71bbe2440ef9d3e669a227397afd0e'", 1),
        ("SCHEMA = 'current_comp_joint_p_node_score_v1'", "SCHEMA = 'current_comp_receiver_directions_score_v1'", 1),
        (old_local_pins, new_local_pins, 1),
        (old_admission, "def arm_admission(args, arm, admit, env, refs, builder):\n"
            "    return receiver_admission(globals(), args, arm, admit, env, refs, builder)\n", 1),
        ("HERE/'build.py'", "HERE/'build_v2.py'", 3),
        ("HERE/'BUILD_PROTOCOL.md'", "HERE/'BUILD_PROTOCOL_V2.md'", 1),
        ("env, proof = helpers(admit); result['helper_source_proof'] = proof",
         "env, proof = helpers(admit); result['helper_source_proof'] = proof\n"
         "    result['endpoint_source_inverse_proof'] = ENDPOINT_SOURCE_PROOF", 1),
        ('all_ten_new_outputs_verified', 'all_five_new_outputs_verified', 1),
        ("(('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only'))", "(('terminal', 'COMP'),)", 2),
        ("        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],\n"
         "        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],",
         "        receiver_main_lead=comparisons['terminal_vs_COMP']['mean_adjusted_ratio'] <= .95\n"
         "            and comparisons['terminal_vs_COMP']['at_least_three_wins'],\n"
         "        receiver_component=comparisons['terminal_vs_COMP']['gate_passed']\n"
         "            and comparisons['terminal_vs_COMP']['mean_adjusted_ratio'] > .95,", 1),
        ("for arm in ('p-only', 'joint') for kind", "for arm in ('terminal',) for kind", 1),
        ("'joint_learning_lead', 'p_only_component'", "'receiver_main_lead', 'receiver_component'", 1),
    )
    changed = original
    for before, after, count in changes:
        _require(changed.count(before) == count, 'Exact receiver scorer source span: '+before[:100])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact receiver scorer inverse span: '+after[:100])
        inverse = inverse.replace(after, before)
    _require(inverse == original, 'Whole frozen scorer byte inverse')
    ast.parse(changed)
    proof = dict(owner=str(OWNER), owner_sha256=OWNER_SHA, full_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_functions=['Admission', 'baseline', 'original_control_checks', 'lifecycle_owner'],
        unchanged_FP64_MSE_and_own_multiplier=True, changed_scope='one receiver role, new metadata joins and prospective5% gate')
    return changed, proof


_source, _proof = derived_source()
exec(compile(_source, str(OWNER)+':receiver-one-terminal', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'receiver_admission': _receiver_admission,
     'ENDPOINT_SOURCE_PROOF': _proof})
