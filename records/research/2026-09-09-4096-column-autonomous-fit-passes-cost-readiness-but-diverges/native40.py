"""Matched cap null/terminal native40+repeat acquisition; inert without --role.

Only admission, fixed role/path selection and provenance are adapted. The
preserved BD/native_v2 complete prediction/input/bill/reset loop is unchanged,
with NO observer or state instrumentation. OFFPY, BLAS1, nice10, 8GB/120s.
No training, target reads or scoring. Each role owns its actual complete bills;
adapted geometry is not priced from the null or any historical measurement.
"""
import ast
import json
import os
from pathlib import Path
import sys
from common import BD, HERE, NEW, PINS as SOURCE_PINS, module, sha, unchanged

BUILDER_SHA = '5a6a38b0c42a77431302c185b77e359db35c8c597c22564bddba15916a07fc0d'
HARNESS_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
BANK_RESULT_SHA = '6b48bebac1442dca55d093efd194dc2852a770be3420b6995d4aa96f3aa7083a'


def native_inputs(candidate, args):
    assert sha(HERE/'build_terminal.py') == BUILDER_SHA
    helper = module(HERE/'build_terminal.py', 'cap4096_terminal_build_metadata')
    inputs = helper.current_readiness()
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    inputs[str(BD/'native_v2.py')] = HARNESS_SHA
    _, valid, _ = helper.definitions()
    prerequisite = dict(parity_result_sha256=helper.PARITY_SHA,
        initial_native_result_sha256=helper.NATIVE_SHA,
        role=args.role, fixed_validation_ranks=list(valid), truth_accessed=False,
        bank_remeasurement_performed=False,
        bank_reference_for_later_judge=dict(path=str(BD/'validation_bank_v2/result.json'),
            sha256=BANK_RESULT_SHA, consumed_in_this_acquisition=False))
    if args.role == 'null':
        assert args.training_sha256 is None and args.build_sha256 is None
        assert candidate == (HERE/'estimator.py').resolve()
        assert sha(candidate) == helper.CAP_SHA
        prerequisite['completed_fit_required'] = False
    else:
        assert args.role == 'terminal'
        assert candidate == (HERE/'build_terminal_v1/estimator.py').resolve()
        training, training_pins = helper.training_admission(args.training_sha256)
        build_path = HERE/'build_terminal_v1/build.json'
        assert sha(build_path) == args.build_sha256
        build = json.loads(build_path.read_text())
        assert build['passed'] and build['schema'] == 'current_paircap4096_e2e_terminal_build_v1'
        assert build['parent_sha256'] == helper.CAP_SHA
        assert build['candidate_sha256'] == sha(candidate) and Path(build['candidate']).resolve() == candidate
        assert build['training_result_sha256'] == args.training_sha256
        assert build['trainer_sha256'] == helper.TRAINER_SHA
        assert build['terminal_epoch'] == 6 and build['updates'] == 240
        assert build['terminal_payload'] == training['terminal_payload']
        assert build['all_40_validation_predictions_sealed'] and not build['validation_targets_accessed']
        assert build['source_bindings'][str(HERE/'build_terminal.py')] == BUILDER_SHA
        for name in ('inverse_source_byte_exact', 'all_other_parent_source_byte_exact',
                     'original_literal_matches_original_payload', 'terminal_literal_roundtrip_byte_exact',
                     'non_PKEYS_byte_exact', 'unused_layer0_byte_exact'):
            assert build[name], name
        proof = helper.inverse_check((HERE/'estimator.py').read_bytes(), candidate.read_bytes())
        assert all(build[k] == v for k, v in proof.items())
        unchanged(build['source_bindings'])
        inputs.update(training_pins); inputs.update(build['source_bindings'])
        inputs[str(build_path)] = args.build_sha256
        prerequisite.update(completed_fit_required=True, training_result_sha256=args.training_sha256,
            build_sha256=args.build_sha256, terminal_payload_sha256=training['terminal_payload']['sha256'])
    inputs[str(candidate)] = sha(candidate)
    unchanged(inputs)
    return inputs, prerequisite


def derived_main():
    """Exact inverse to the pinned harness main; no call-loop edits."""
    path = BD/'native_v2.py'
    assert sha(path) == HARNESS_SHA == SOURCE_PINS[path]
    text = path.read_text()
    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[function.lineno-1:function.end_lineno])
    old_parser = ("    parser.add_argument('--source',type=Path)\n"
                  "    parser.add_argument('--split',choices=('development','validation'))\n"
                  "    parser.add_argument('--out',type=Path)\n"
                  "    args=parser.parse_args()\n"
                  "    if args.source is None: parser.print_help();return\n"
                  "    assert args.split and args.out and not args.out.exists()")
    new_parser = ("    parser.add_argument('--role',choices=('null','terminal'))\n"
                  "    parser.add_argument('--source-sha256')\n"
                  "    parser.add_argument('--training-sha256')\n"
                  "    parser.add_argument('--build-sha256')\n"
                  "    args=parser.parse_args()\n"
                  "    if args.role is None: parser.print_help();return 0\n"
                  "    assert sha(__file__)==args.source_sha256, 'Reviewed native40 source pin required'\n"
                  "    args.source=HERE/('estimator.py' if args.role=='null' else 'build_terminal_v1/estimator.py')\n"
                  "    args.split='validation'\n"
                  "    args.out=HERE/f'native_validation_{args.role}_v1'\n"
                  "    assert not args.out.exists(), 'Exclusive role acquisition; no automatic resume'")
    begin = original.index('    if candidate==BANK:\n')
    end = original.index('    manifest=json.loads(', begin)
    old_branch = original[begin:end]
    edits = [
        (old_parser, new_parser),
        ("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n"
         "        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}",
         "    inputs,prerequisite=native_inputs(candidate,args)"),
        (old_branch, '    ranks=VALID\n    assert list(ranks)==prerequisite["fixed_validation_ranks"]\n'),
        ("        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,",
         "        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,\n"
         "        schema='current_paircap4096_e2e_native40_v1',role=args.role,prerequisites=prerequisite,\n"
         "        state_instrumentation=False,own_complete_bills=True,score_measured=False,"),
        ("        result.update(passed=True,hashes_unchanged=True,",
         "        assert len(result['trials'])==41 and [t['rank'] for t in result['trials']]==list(VALID)+[VALID[0]]\n"
         "        result['all_outputs_and_bills_sealed_before_scoring']=True\n"
         "        result.update(passed=True,hashes_unchanged=True,"),
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
    # Stronger local check: the entire data/preparation/prediction/bill/reset
    # loop, including context limits and materialization, is literally intact.
    first = '            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n'
    last = '        assert all(sha(p)==v for p,v in inputs.items())\n'
    assert source[source.index(first):source.index(last)] == original[original.index(first):original.index(last)]
    return source


def main():
    if '--role' not in sys.argv:
        print('Inert. --role null|terminal --source-sha256 SHA [--training-sha256 SHA --build-sha256 SHA]')
        return 0
    os.chdir(NEW)
    source = derived_main()
    old = module(BD/'native_v2.py', 'cap4096_preserved_native40_harness')
    # old.PINS was constructed with BD paths before HERE is overridden.
    env = dict(vars(old), __file__=str(Path(__file__).resolve()), HERE=HERE, PARENT=NEW,
               native_inputs=native_inputs)
    exec(compile(source, str(__file__)+':preserved_main', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
