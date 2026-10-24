"""Fixed P/covariance CHECK4 against COMP and the sealed matched P-only arm."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent/'current_comp_joint_p_node_v1'
RD = HERE.parent/'current_comp_receiver_directions_v1'
OWNER = JOINT/'score.py'
OWNER_SHA = '7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e'
RD_SHA = 'e6697bb3f8349078ec078f342178ad5c36136d2300c2ec41af085155efc9e42b'
MATCHED_SCORE_SHA = '38060450e770c2c66310d05919086b6fa3177d2461077587b2db61068b1e3b4b'
MATCHED_TRAIN_SHA = 'c2ce62dc3565e6d77ac14e8e6110348bd364eeec59cb0e9378500bd1348c3157'
MATCHED = dict(
    builder_sha256='11e7f7588e692587c11480f6a09bd7cfb9492d02851381755821b7dbf01a9ed4',
    build_protocol_sha256='4dbfcdb214ca59036061498e3144a123b313611ff80681fd6e9169078ad2e852',
    native_source_sha256='851813e9679cd58f1b23bbd15de36cdeb866df0e646e1e8cb9e9aa86e924b0bc',
    native_watch_source_sha256='d35ba7cbe406ae5e6200f8778b7a582bc6ea17336be8086b5e813e5a25dbe65e',
    native_protocol_sha256='51fd9d5280c523d6edcd30d650035e9be07428d1e4cbed2647837ff4993f666c',
    p_only_build_sha256='12cffa5ae166412639558438a9c751b2fa04888f3ef33951877d1d85c273cb86',
    p_only_candidate_sha256='1e1cce080a3e1599be4bbfcf47ae64048b9e355294ea958755903f5c88e45d9d',
    p_only_native_sha256='c9f6e910bbad02ec140a3db69ead30d8b1dcd6dda95cf1cdc9c793b9550e54a1',
    p_only_watch_sha256='8c5a832308d39e2ec580635039d6e7d9416a15cb6f5836a1782e44b5aa998fad')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def function(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Unique frozen source function '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def adapt(source, changes):
    result = source
    for before, after, count in changes:
        require(result.count(before) == count, 'Exact covariance scorer source span '+before[:100])
        result = result.replace(before, after)
    inverse = result
    for before, after, count in reversed(changes):
        require(inverse.count(after) == count, 'Exact covariance scorer inverse span '+after[:100])
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Full original scoring source byte inverse')
    ast.parse(result)
    return result


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def covariance_admission_source():
    require(sha(RD/'score.py') == RD_SHA, 'Frozen zero/terminal admission owner')
    original = function((RD/'score.py').read_text(), '_receiver_admission')
    changes = (
        ("'current_comp_receiver_directions_build_v1'", "'current_comp_joint_p_covariance_build_v1'", 1),
        ("'current_comp_receiver_directions_native_v1'", "'current_comp_joint_p_covariance_native_v1'", 1),
        ("here/'build_v2.py'", "here/'build.py'", 2),
        ("here/'BUILD_PROTOCOL_V2.md'", "here/'BUILD_PROTOCOL.md'", 2),
        ('original_P_A_cstar_literals_unchanged', 'original_A_cstar_literals_unchanged', 1),
        ("'literal_roundtrip_byte_exact', 'zero_terminal_literal_only', 'full_shape_FP32_finite'",
         "'literal_roundtrip_byte_exact', 'frozen_nonP_and_inactive66_byte_exact', 'full_shape_FP32_finite'", 1),
        ('receiver_literal', 'covariance_literal', 3),
        ("['shape'] == [64, 4]", "['shape'] == [14, 65]", 1),
        ('receiver_R_initialization_owned', 'covariance_Q_initialization_owned', 1),
        ('receiver_observer_inverse_verified', 'covariance_observer_inverse_verified', 1),
        ('_bdr_parameters', '_pcc_parameters', 1),
        ("['literal_elements'] == 256", "['literal_elements'] == 910", 1),
        ('(512 if cold else 0)', '(1820 if cold else 0)', 1),
        ('receiver_R_materialization_flops', 'covariance_Q_materialization_flops', 1),
        ('R_empty_at_entry', 'Q_empty_at_entry', 1),
        ('R_ready_at_binary_entry', 'Q_ready_at_binary_entry', 1),
        ('hashlib.sha256(bytes(1024))', 'hashlib.sha256(bytes(3640))', 1),
        ('Exact qualified256-value receiver build', 'Exact qualified910-value covariance/P build', 1),
        ('R512cold/0warm', 'Q1820cold/0warm', 1),
        ('terminal differs only in receiver literal', 'terminal differs only in declared P/Q surface', 1),
    )
    return adapt(original, changes)


def matched_training(admit, current):
    """Metadata identity of the completed matched control, never a new fit."""
    admit.pin(JOINT/'train_p_only_v1/result.json', MATCHED_TRAIN_SHA)
    old = json.loads((JOINT/'train_p_only_v1/result.json').read_text())
    for key in ('initial_P', 'initial_A', 'cstar_raw_sha256', 'seed', 'batch_nets',
                'train_ranks', 'validation_ranks', 'loss', 'target_original_dtypes'):
        require(current[key] == old[key], 'Exactly matched P-only training field '+key)
    cost_tables = []
    for item in (current, old):
        meta = item['cost_weights']; admit.pin(meta['path'], meta['sha256'])
        cost_tables.append(json.loads(Path(meta['path']).read_text()))
    require(current['cost_weights']['sha256'] == old['cost_weights']['sha256']
        and cost_tables[0] == cost_tables[1], 'Same exact normalized-cost table at separately owned paths')
    for key in ('name', 'P_lr', 'steps', 'betas', 'eps', 'weight_decay', 'foreach', 'group_gradient_norm_cap'):
        require(current['optimizer'][key] == old['optimizer'][key], 'Same P optimizer '+key)
    require(current['updates'] == old['updates'] == 15 and current['training_graphs'] == old['training_graphs'] == 60,
            'Exactly matched endpoint duration')
    rosters = []
    for item in (current, old):
        meta = item['model_call_roster']; admit.pin(meta['path'], meta['sha256'])
        rosters.append(json.loads(Path(meta['path']).read_text()))
    for key in ('seed', 'generator', 'orders', 'entries'):
        require(rosters[0][key] == rosters[1][key], 'Same realized matched-control order '+key)
    for left, right in zip(current['update_records'], old['update_records']):
        admit.pin(left['path'], left['sha256']); admit.pin(right['path'], right['sha256'])
        a, b = (json.loads(Path(x['path']).read_text()) for x in (left, right))
        require(a['update'] == b['update'] and a['ranks'] == b['ranks']
            and a['learning_rates'][0] == b['learning_rates'][0], 'Same realized P optimizer schedule')
    admit.result['matched_P_only_training_frame_verified'] = True


def endpoint_admission(g, args, arm, admit, env, refs, builder):
    if arm == 'p_only':
        owner = imported(OWNER, 'covariance_archived_score_admission')
        admit.bind(owner.PINS)
        for name, key in (('build.py', 'builder_sha256'), ('BUILD_PROTOCOL.md', 'build_protocol_sha256'),
                          ('native.py', 'native_source_sha256'), ('watch_native.py', 'native_watch_source_sha256'),
                          ('NATIVE_PROTOCOL.md', 'native_protocol_sha256')):
            admit.pin(JOINT/name, MATCHED[key])
        old_builder = imported(JOINT/'build.py', 'covariance_archived_build_admission')
        answer = owner.arm_admission(SimpleNamespace(**MATCHED), 'p_only', admit, env, refs, old_builder)
        archived, _, seals = admit.chain(JOINT/'score_v1', MATCHED_SCORE_SHA, limit=120, inventory=False)
        require(archived['schema'] == 'current_comp_joint_p_node_score_v1' and archived['integrity_passed']
            and archived['exact_COMP_score_replay'] and archived['hashes_unchanged'] and archived['sources_unchanged']
            and archived['ranks'] == list(g['RANKS']) and archived['all_predictions_verified_before_truth'],
            'Frozen completed matched score receipt')
        g['MATCHED_SCORE_RECORD'] = archived
        admit.result['matched_P_only_reused_without_rerun'] = True
        return answer
    require(arm == 'terminal', 'Only one new endpoint and one exact completed control')
    scope = dict(hashlib=hashlib, json=json,
        ZERO_RESULT_SHA='de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625',
        ZERO_END_SHA='0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073')
    exec(compile(COVARIANCE_ADMISSION, str(RD/'score.py')+':covariance-metadata', 'exec'), scope)
    answer = scope['_receiver_admission'](g, args, arm, admit, env, refs, builder)
    path = HERE/'train_v1/result.json'
    require(str(path) in admit.pins, 'New native/build lineage owns its completed fixed training result')
    matched_training(admit, json.loads(path.read_text()))
    return answer


def derived_source():
    require(sha(OWNER) == OWNER_SHA, 'Frozen complete objective scorer')
    original = OWNER.read_text()
    old_local = """    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '4ef8567620f6b2efb2890fda32eb029082fc1a07e084ab2f37ec35b4d29e4118',
    HERE/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',"""
    new_local = """    HERE/'MODEL_PROTOCOL.md': '1117752d081229cb4d6b64fff2fa86f65a6bb1a760ce9abd1458a6f26095833d',
    HERE/'DECISION.md': '814ac1f31121db92f3165ecfa0cb3253e8b66617e3e62d17316f2f288e71be98',
    HERE/'TRAIN_PROTOCOL.md': 'd9f6de519bc1bade3e4b3b76b2085967639b736ea84b645099821b16d88a8591',
    NEW/'current_comp_joint_p_node_v1/score.py': '"""+OWNER_SHA+"""',
    NEW/'current_comp_joint_p_node_v1/watch_score.py': 'cca85002951bedef0891610c086a3984aaa6f887e4a772b8dc85c38701bfd071',
    NEW/'current_comp_receiver_directions_v1/score.py': '"""+RD_SHA+"""',"""
    changes = (
        ("('p_only', 'joint'), 2**41", "('p_only', 'terminal'), 2**41", 1),
        ("MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'",
         "MODEL_SHA = 'f58c687782b1f5ff970dd52366366d83e33b81b85eb4f6d92d44d815973033db'", 1),
        ("SCHEMA = 'current_comp_joint_p_node_score_v1'", "SCHEMA = 'current_comp_joint_p_covariance_score_v1'", 1),
        (old_local, new_local, 1),
        (function(original, 'arm_admission'), 'def arm_admission(args, arm, admit, env, refs, builder):\n'
         '    return endpoint_admission(globals(), args, arm, admit, env, refs, builder)\n', 1),
        ("env, proof = helpers(admit); result['helper_source_proof'] = proof",
         "env, proof = helpers(admit); result['helper_source_proof'] = proof\n"
         "    result['endpoint_source_inverse_proof'] = ENDPOINT_SOURCE_PROOF", 1),
        ('all_ten_new_outputs_verified', 'all_five_new_and_five_matched_outputs_verified', 1),
        ("        row['comparisons'] = {}", "        require(MATCHED_SCORE_RECORD['records'][call]['rank'] == rank\n"
         "            and row['per_role']['p_only'] == MATCHED_SCORE_RECORD['records'][call]['per_role']['p_only'],\n"
         "            'Exact archived matched P-only raw/F/adjusted/layers replay')\n"
         "        row['comparisons'] = {}", 1),
        ("(('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only'))",
         "(('terminal', 'COMP'), ('p_only', 'COMP'), ('terminal', 'p_only'))", 2),
        ("        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],\n"
         "        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],",
         "        covariance_main_lead=all(comparisons['terminal_vs_'+role]['mean_adjusted_ratio'] <= .95\n"
         "            and comparisons['terminal_vs_'+role]['at_least_three_wins'] for role in ('COMP', 'p_only')),\n"
         "        covariance_component=all(comparisons['terminal_vs_'+role]['gate_passed'] for role in ('COMP', 'p_only'))\n"
         "            and any(comparisons['terminal_vs_'+role]['mean_adjusted_ratio'] > .95 for role in ('COMP', 'p_only')),\n"
         "        exact_P_only_score_replay=True,", 1),
        ("for arm in ('p-only', 'joint') for kind", "for arm in ('terminal',) for kind", 1),
        ("'joint_learning_lead', 'p_only_component'", "'covariance_main_lead', 'covariance_component'", 1),
    )
    result = adapt(original, changes)
    return result, dict(owner=str(OWNER), owner_sha256=OWNER_SHA, full_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(result.encode()).hexdigest(),
        admission_owner=str(RD/'score.py'), admission_owner_sha256=RD_SHA,
        admission_inverse_verified=True, admission_sha256=hashlib.sha256(COVARIANCE_ADMISSION.encode()).hexdigest(),
        unchanged_FP64_MSE_and_own_multiplier=True, fixed_controls_no_refit=True)


COVARIANCE_ADMISSION = covariance_admission_source()
_source, _proof = derived_source()
exec(compile(_source, str(OWNER)+':one-covariance-plus-matched-control', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'endpoint_admission': endpoint_admission,
     'ENDPOINT_SOURCE_PROOF': _proof})
