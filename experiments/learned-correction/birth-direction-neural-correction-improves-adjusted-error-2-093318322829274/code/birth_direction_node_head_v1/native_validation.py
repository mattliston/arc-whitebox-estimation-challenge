"""Fixed native4 plus repeat; inert without --role.

Only parent d50, the qualified zero prototype, and the sealed six-epoch
terminal A are accepted. Each role is a fresh official-owned-FNP process
with one participant initialization and its own complete bills. The exact
native_v2 input/predict/materialize/bill/reset/repeat loop is preserved.
No truth file, fit, score, parameter selection or release action is here.
The parent role is optional: the separate judge may reuse the qualified
parent40 run's exact first-four prefix, never an estimated added bill.
"""
import ast
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
BUILDER_SHA = '261f53e5589edec7d76e9f50cc8e2b7b96139a4a05bcc6232af84aa6d13a0377'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
VALID = (37, 39, 42, 48)
INITIALIZATION = {'parent': 1383913786, 'zero': 1383914298, 'terminal': 1383914298}
ROLE_PATHS = {'parent': BD / 'build_terminal_v1/estimator.py',
              'zero': HERE / 'build_zero_v1/estimator.py',
              'terminal': HERE / 'build_terminal_v1/estimator.py'}
VERSIONS = {'numpy': '2.4.6', 'flopscope': '0.12.0', 'whestbench': '0.16.0',
            'threadpoolctl': '3.6.0'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def native_inputs(candidate, args):
    """Read source/receipt metadata and prediction bytes before FNP imports."""
    require(sha(HERE / 'build_terminal.py') == BUILDER_SHA, 'Reviewed terminal builder')
    builder = module(HERE / 'build_terminal.py', 'bdnh_native4_inert_terminal_builder')
    context = builder.verify_build(args.training_sha256, args.build_sha256)
    require(builder.VALID == VALID and builder.INITIALIZATION == INITIALIZATION
            and builder.ROLE_PATHS == ROLE_PATHS, 'Fixed native4 role/split contract')
    selected = context['candidates'][args.role]
    require(candidate == Path(selected['path']).resolve() and sha(candidate) == selected['sha256'],
            'Exact selected source, not a substituted terminal payload')
    pins = dict(context['pins'])
    store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    for path, expected in ((HERE / 'build_terminal.py', BUILDER_SHA),
                           (BD / 'native_v2.py', NATIVE_SHA),
                           (store / 'manifest.json', STORE_SHA),
                           (Path(__file__).resolve(), args.source_sha256)):
        require(sha(path) == expected, 'Source/fixture pin: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == expected, 'Conflicting source pin')
        pins[str(path)] = expected
    require({name: importlib.metadata.version(name) for name in VERSIONS} == VERSIONS,
            'Exact qualified native versions')
    return pins, dict(role=args.role, fixed_ranks=list(VALID),
        training_result_sha256=args.training_sha256, terminal_build_sha256=args.build_sha256,
        terminal_A=context['terminal_A'],
        validation_predictions_sealed_sha256=context['training']['validation_predictions_sealed_sha256'],
        completed_training_required=True, all_four_training_predictions_sealed=True,
        own_complete_bills=True, input_preparation_in_participant_bill=False,
        frozen_P=True, frozen_cstar=True, truth_accessed=False, versions=VERSIONS)


def finish_native(result, args):
    """Post-context metadata gates only; no participant work is removed."""
    trials = result['trials']
    require([trial['rank'] for trial in trials] == list(VALID) + [VALID[0]], 'All four calls plus exact repeat')
    require([trial['includes_initialization'] for trial in trials] == [True, False, False, False, False],
            'One actual participant initialization')
    initialization = trials[0]['flops'] - trials[-1]['flops']
    require(initialization == INITIALIZATION[args.role], 'Actual cold-minus-repeat initialization')
    require(trials[0]['output_bytes_sha256'] == trials[-1]['output_bytes_sha256'], 'Repeat output seal')
    require(len(result['artifacts']) == 25 and len(result['input_preparation']) == 5,
            'Five complete official-input/output/bill/reset lifecycles')
    require(all(sha(path) == digest for path, digest in result['source_bindings'].items()), 'Inputs changed')
    require(all(sha(path) == digest for path, digest in result['artifacts'].items()), 'Acquisition artifact changed')
    result.update(passed=True, completed=True, hashes_unchanged=True,
        initialization_flops=initialization, all_predictions_and_own_bills_sealed=True,
        score_measured=False, no_fit=True)


def derived_main():
    """Exact inverse for metadata edits; the entire numerical loop is identical."""
    path = BD / 'native_v2.py'
    require(sha(path) == NATIVE_SHA, 'Qualified native-v2 source')
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    first = original.index("    parser.add_argument('--source',type=Path)\n")
    last = original.index('    assert Path(sys.prefix)', first)
    parser = ("    parser.add_argument('--role',choices=('parent','zero','terminal'))\n"
              "    parser.add_argument('--source-sha256')\n"
              "    parser.add_argument('--training-sha256')\n"
              "    parser.add_argument('--build-sha256')\n"
              "    args=parser.parse_args()\n"
              "    if args.role is None: parser.print_help();return 0\n"
              "    assert sha(__file__)==args.source_sha256, 'Reviewed native4 source pin required'\n"
              "    args.source=ROLE_PATHS[args.role]\n"
              "    args.split='validation'\n"
              "    args.out=HERE/f'native_validation_{args.role}_v1'\n"
              "    assert not args.out.exists(), 'Exclusive acquisition; no resume'\n")
    a, b = original.index('    if candidate==BANK:\n'), original.index('    manifest=json.loads(')
    edits = [
        (original[first:last], parser),
        ("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n"
         "        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}",
         '    inputs,prerequisite=native_inputs(candidate,args)'),
        (original[a:b], '    ranks=VALIDATION_RANKS\n'),
        ('        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,',
         '        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,\n'
         "        schema='birth_direction_node_head_native4_v1',role=args.role,completed=False,\n"
         '        own_complete_bills=True,frozen_P=True,frozen_cstar=True,score_measured=False,\n'
         '        training_result_sha256=args.training_sha256,terminal_build_sha256=args.build_sha256,\n'
         '        prerequisites=prerequisite,state_instrumentation=False,\n'
         '        native_original_sha256=NATIVE_FILE_SHA,native_main_sha256=NATIVE_MAIN_SHA,\n'
         '        native_loop_sha256=NATIVE_LOOP_SHA,\n'
         '        main_adapter_inverse_byte_exact=True,numerical_loop_byte_exact=True,'),
        ('        result.update(passed=True,hashes_unchanged=True,\n'
         "                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])",
         '        finish_native(result,args)'),
        ("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))",
         "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result['completed'],"
         "result_sha256=sha(args.out/'result.json')))")]
    source = original
    for old, new in edits:
        require(source.count(old) == 1, 'Unique native adapter edit')
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique native adapter inverse')
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Whole-main byte inverse')
    begin = '            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n'
    end = '        assert all(sha(p)==v for p,v in inputs.items())\n'
    loop = original[original.index(begin):original.index(end)]
    require(source[source.index(begin):source.index(end)] == loop, 'Official-input/predict/bill/reset loop unchanged')
    ast.parse(source)
    return source, hashlib.sha256(original.encode()).hexdigest(), hashlib.sha256(loop.encode()).hexdigest()


def main():
    if '--role' not in sys.argv:
        print('Inert. --role parent|zero|terminal --source-sha256 SHA --training-sha256 SHA --build-sha256 SHA')
        return 0
    os.chdir(NEW)
    source, original_sha, loop_sha = derived_main()
    base = module(BD / 'native_v2.py', 'bdnh_preserved_native4')
    env = dict(vars(base), __file__=str(Path(__file__).resolve()), HERE=HERE,
               native_inputs=native_inputs, finish_native=finish_native,
               ROLE_PATHS=ROLE_PATHS, VALIDATION_RANKS=VALID,
               NATIVE_FILE_SHA=NATIVE_SHA, NATIVE_MAIN_SHA=original_sha, NATIVE_LOOP_SHA=loop_sha)
    exec(compile(source, str(__file__) + ':preserved_main', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
