"""Fixed current-d50 P bank/null/terminal native40; inert without --role.

Only admission, role/path selection and receipt fields adapt BD/native_v2.
The official-input preparation and full prediction/bill/reset loop are
literally unchanged. All roles require the same completed fixed fit, terminal
build and all40 sealed MPS predictions. No truths, refit, score or old bills.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
BUILDER_SHA = '8b621d1713841c6271f048504205030ec7bc78df1842be4a3071e00ecf01d084'
HARNESS_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
INITIALIZATION = 1_383_913_786  # Current d50: original initialization plus 120 F cstar.
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
FIXED = {
    HERE / 'build_terminal.py': BUILDER_SHA,
    HERE / 'train.py': '59beee7cb1578ecb6fe523fb1835c4f2832495551be95833d7210fde0883d841',
    HERE / 'watch.py': 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b',
    HERE / 'model.py': '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0',
    HERE / 'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    HERE / 'PROTOCOL_V1.md': 'daac62127665be18a568b90302095c8f83e17bfa85328f87122d39a555bed81c',
    HERE / 'implementation.md': '5dc8e433e13fb1b7cec20dbd61e784e8f7d9a50dd89d4e6f03225f73e378b92d',
    PARENT: PARENT_SHA,
    BD / 'native_v2.py': HARNESS_SHA,
    BD / 'NATIVE_SCOPE_CLARIFICATION.md': 'b43aa00adcc30f434f5e301abcd490deb220020096ea47daebcf908f420bd333',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}


def sha(path):
    path = Path(path).resolve()
    assert 'truthpost' not in path.parts and not path.name.startswith(('mps_truth_', 'fresh_rank_truth'))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def unchanged(pins):
    assert all(sha(path) == digest for path, digest in pins.items()), 'Authenticated source/artifact changed'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_build(training_sha256, build_sha256):
    """Shared stdlib-only admission; never resolves or decodes truth arrays."""
    assert sys.flags.optimize == 0
    assert all(isinstance(x, str) and len(x) == 64 for x in (training_sha256, build_sha256))
    pins = {str(path): digest for path, digest in FIXED.items()}
    unchanged(pins)
    helper = module(HERE / 'build_terminal.py', 'current_d50_p_native_terminal_metadata')
    # Builder admission uses only its public stdlib hash/JSON helpers. Keep
    # the truth-path exclusion active even for every transitive binding.
    helper.sha = sha
    training, training_pins = helper.training_admission(training_sha256)
    _, valid, keys = helper.definitions()
    assert tuple(valid) == VALID and len(keys) == 11
    build_path = HERE / 'build_terminal_v1/build.json'
    assert sha(build_path) == build_sha256
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['schema'] == 'current_d50_p_coadaptation_terminal_build_v1'
    assert build['parent_sha256'] == build['null_sha256'] == PARENT_SHA
    assert build['training_result_sha256'] == training_sha256
    assert build['terminal_epoch'] == 6 and build['updates'] == 240
    assert build['terminal_payload'] == training['terminal_payload']
    assert build['modified_assignment_only'] == '_GNN_LITERAL'
    assert build['all_40_validation_predictions_sealed'] and not build['validation_targets_accessed']
    for key in ('null_source_byte_exact', 'inverse_source_byte_exact', 'all_other_parent_source_byte_exact',
                'original_literal_matches_original_payload', 'terminal_literal_roundtrip_byte_exact',
                'non_PKEYS_byte_exact', 'unused_layer0_byte_exact', 'cstar_byte_exact'):
        assert build[key], key
    assert build['source_bindings'][str(HERE / 'build_terminal.py')] == BUILDER_SHA
    assert all(training['source_bindings'].get(path) == digest
               for path, digest in pins.items() if path in training['source_bindings'])
    for path in (HERE / 'train.py', HERE / 'model.py', HERE / 'qualify.py',
                 HERE / 'PROTOCOL_V1.md', HERE / 'watch.py'):
        assert training['source_bindings'][str(path)] == FIXED[path]
    paths = {'bank': PARENT, 'null': HERE / 'build_terminal_v1/null_estimator.py',
             'terminal': HERE / 'build_terminal_v1/estimator.py'}
    candidates = {role: dict(path=str(path), sha256=sha(path)) for role, path in paths.items()}
    assert candidates['bank']['sha256'] == candidates['null']['sha256'] == PARENT_SHA
    assert candidates['terminal']['sha256'] == build['candidate_sha256']
    parent, null, terminal = (paths[role].read_bytes() for role in ('bank', 'null', 'terminal'))
    assert null == parent
    ps, pe, pliteral = helper.assignment_span(parent)
    ts, te, tliteral = helper.assignment_span(terminal)
    assert ts == ps and terminal[:ts] + parent[ps:pe] + terminal[te:] == parent
    assert set(pliteral) == set(tliteral) == set(training['terminal_payload']['arrays'])
    for key in pliteral:
        if key not in keys:
            assert pliteral[key] == tliteral[key], key
    for key in ('head', 'head_b', 'gain'):
        assert pliteral[key][0] == tliteral[key][0], key
    pbd, tbd = (helper.assignment_span(value, '_BD_COEFFICIENT_LITERAL') for value in (parent, terminal))
    assert parent[pbd[0]:pbd[1]] == terminal[tbd[0]:tbd[1]]
    for group in (training_pins, training['artifacts'], build['source_bindings']):
        unchanged(group)
        for path, digest in group.items():
            assert path not in pins or pins[path] == digest, 'Conflicting lineage pin'
            pins[path] = digest
    pins[str(build_path)] = build_sha256
    for item in candidates.values():
        pins[item['path']] = item['sha256']
    pins[str(Path(__file__).resolve())] = sha(__file__)
    pins[str(HERE / 'NATIVE40_IMPLEMENTATION.md')] = sha(HERE / 'NATIVE40_IMPLEMENTATION.md')
    unchanged(pins)
    return dict(training=training, build=build, source_bindings=pins, candidates=candidates,
                validation_ranks=list(VALID), training_result_sha256=training_sha256, build_sha256=build_sha256)


def native_inputs(candidate, args):
    checked = verify_build(args.training_sha256, args.build_sha256)
    chosen = checked['candidates'][args.role]
    assert str(candidate) == chosen['path'] and sha(candidate) == chosen['sha256']
    assert not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope'))
    _, proof = derived_main()
    prerequisite = dict(role=args.role, completed_fit_required=True,
        training_result_sha256=args.training_sha256, build_sha256=args.build_sha256,
        terminal_payload_sha256=checked['training']['terminal_payload']['sha256'],
        terminal_seal_sha256=checked['training']['terminal_seal_sha256'],
        validation_predictions_sealed_sha256=checked['training']['validation_predictions_sealed_sha256'],
        fixed_validation_ranks=list(VALID), frozen_cstar=True, all_40_mps_predictions_sealed=True,
        expected_initialization_flops=INITIALIZATION, adapter_proof=proof, truth_accessed=False)
    return checked['source_bindings'], prerequisite


def derived_main():
    """Return exact inverse-certified main and the unchanged-loop source proof."""
    path = BD / 'native_v2.py'
    assert sha(path) == HARNESS_SHA
    text = path.read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    assert len(nodes) == 1
    fn = nodes[0]
    original = ''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    old_parser = ("    parser.add_argument('--source',type=Path)\n"
                  "    parser.add_argument('--split',choices=('development','validation'))\n"
                  "    parser.add_argument('--out',type=Path)\n"
                  "    args=parser.parse_args()\n"
                  "    if args.source is None: parser.print_help();return\n"
                  "    assert args.split and args.out and not args.out.exists()")
    new_parser = ("    parser.add_argument('--role',choices=('bank','null','terminal'))\n"
                  "    parser.add_argument('--source-sha256')\n"
                  "    parser.add_argument('--training-sha256')\n"
                  "    parser.add_argument('--build-sha256')\n"
                  "    args=parser.parse_args()\n"
                  "    if args.role is None: parser.print_help();return 0\n"
                  "    assert sha(__file__)==args.source_sha256, 'Reviewed native40 source pin required'\n"
                  "    args.source=ROLE_PATHS[args.role]\n"
                  "    args.split='validation'\n"
                  "    args.out=HERE/f'native_validation_{args.role}_v1'\n"
                  "    assert not args.out.exists(), 'Exclusive role acquisition; no resume'")
    begin = original.index('    if candidate==BANK:\n')
    end = original.index('    manifest=json.loads(', begin)
    edits = [
        (old_parser, new_parser),
        ("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n"
         "        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}",
         "    inputs,prerequisite=native_inputs(candidate,args)"),
        (original[begin:end], '    ranks=VALID\n    assert list(ranks)==prerequisite["fixed_validation_ranks"]\n'),
        ("        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,",
         "        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,\n"
         "        schema='current_d50_p_coadaptation_native40_v1',completed=False,role=args.role,\n"
         "        prerequisites=prerequisite,own_complete_bills=True,frozen_cstar=True,no_fit=True,\n"
         "        training_result_sha256=args.training_sha256,terminal_build_sha256=args.build_sha256,\n"
         "        expected_initialization_flops=INITIALIZATION,score_measured=False,state_instrumentation=False,"),
        ("        result.update(passed=True,hashes_unchanged=True,",
         "        assert len(result['trials'])==41 and [t['rank'] for t in result['trials']]==list(VALID)+[VALID[0]]\n"
         "        assert result['trials'][0]['flops']-result['trials'][-1]['flops']==INITIALIZATION\n"
         "        result['all_outputs_and_bills_sealed_before_scoring']=True\n"
         "        result.update(passed=True,completed=True,hashes_unchanged=True,"),
        ("    except BaseException:\n        result['error']=traceback.format_exc()",
         "    except BaseException:\n        result.update(passed=False,completed=False,hashes_unchanged=False)\n"
         "        result['error']=traceback.format_exc()"),
        ("dict(passed=result['passed'],result_sha256=sha(args.out/'result.json'))",
         "dict(completed=result['completed'],passed=result['passed'],result_sha256=sha(args.out/'result.json'))"),
    ]
    source = original
    for old, new in edits:
        assert source.count(old) == 1, old[:100]
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        assert inverse.count(new) == 1, new[:100]
        inverse = inverse.replace(new, old)
    assert inverse == original
    ast.parse(source)
    first = '            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n'
    last = '        assert all(sha(p)==v for p,v in inputs.items())\n'
    loop = original[original.index(first):original.index(last)]
    assert source[source.index(first):source.index(last)] == loop
    return source, dict(harness_sha256=HARNESS_SHA, main_inverse_byte_exact=True,
        complete_input_prediction_bill_reset_loop_byte_exact=True,
        loop_sha256=hashlib.sha256(loop.encode()).hexdigest(),
        original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_main_sha256=hashlib.sha256(source.encode()).hexdigest(), edit_count=len(edits))


def main():
    if not any(arg == '--role' or arg.startswith('--role=') for arg in sys.argv):
        print('Inert. --role bank|null|terminal --source-sha256 SHA --training-sha256 SHA --build-sha256 SHA')
        return 0
    assert sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope'))
    os.chdir(NEW)
    source, _ = derived_main()
    old = module(BD / 'native_v2.py', 'current_d50_p_preserved_native40_harness')
    # old.PINS already owns its original BD and installed-library paths.
    env = dict(vars(old), __file__=str(Path(__file__).resolve()), HERE=HERE, PARENT=NEW,
               native_inputs=native_inputs, INITIALIZATION=INITIALIZATION,
               ROLE_PATHS={'bank': PARENT, 'null': HERE / 'build_terminal_v1/null_estimator.py',
                           'terminal': HERE / 'build_terminal_v1/estimator.py'})
    exec(compile(source, str(__file__) + ':preserved_main', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
