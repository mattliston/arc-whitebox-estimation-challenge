"""Target-free parent/zero/nonzero native fidelity; inert without --role.

The qualified native-v2 official-input, prediction, materialization, bill,
reset and repeat loop is byte-preserved. Only role admission and provenance
change. Each role owns fresh complete bills on fixed35/100/35. No observer,
training, scoring, truth loading, numerical fallback or production claim.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
BUILDER_SHA = '42bef1c5c8aeaf0b60ae96961bb7c29d20535a555bbbfc554a61c5ec0aecb926'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
SCHEDULE_SHA = 'cd7f9ca723c51b9288726a494efde1da673b5db40806f51fd51fdb48c03cf14f'
BUILD_SHAS = {
    'zero': 'a560235a6cbee805b2d37f50b34583996d93e013fafcbb4c2b6243c408aa7f48',
    'nonzero': '3eb96fcdf5553c211a21a1005f2043940b8fbcab033fca090f221e1dad5d8b41',
}
CANDIDATES = {
    'parent': (BD / 'build_terminal_v1/estimator.py',
               'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'),
    'zero': (HERE / 'build_zero_v1/estimator.py',
             '1a99b49fcdc5cbfdaa03a40e676c0a910c5cf4e2d001965bf9e15e18678f1f05'),
    'nonzero': (HERE / 'build_nonzero_v1/estimator.py',
                'e3c242b9dfa53cb348dd90aeb9426f0dd5ecc4b45f062839297d22438124516f'),
}


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def provenance():
    """Source/build authentication only; no model or numerical imports."""
    assert sha(HERE / 'build.py') == BUILDER_SHA
    build = module(HERE / 'build.py', 'bdnh_wide_inert_builder')
    nested = build.authenticate()
    pins = {item['path']: item['sha256'] for item in nested.values()}
    for role, (candidate, digest) in CANDIDATES.items():
        assert sha(candidate) == digest
        pins[str(candidate)] = digest
        if role == 'parent':
            continue
        folder = candidate.parent
        bp, cp = folder / 'build.json', folder / 'completion.json'
        assert sha(bp) == BUILD_SHAS[role]
        record, completion = json.loads(bp.read_text()), json.loads(cp.read_text())
        assert record['passed'] and record['completed'] and not record['executed']
        assert completion['passed'] and completion['completed'] and completion['source_only']
        assert completion['build_sha256'] == BUILD_SHAS[role]
        assert record['fixture'] == role and record['candidate_sha256'] == digest
        assert record['parent_sha256'] == build.PARENT_SHA
        for flag in ('inverse_source_byte_exact', 'all_other_parent_source_byte_exact',
                     'imports_and_learned_P_payloads_unchanged', 'stored_birth_update_byte_exact',
                     'no_new_CP_columns', 'no_numerical_execution'):
            assert record[flag], flag
        build.unchanged(record['source_bindings'])
        expected, audit = build.construct(build.PARENT.read_text(), build.CORE.read_text(),
                                           build.prototype_matrix(role))
        assert candidate.read_bytes() == expected.encode() and record['effective_functions'] == audit
        assert record['A_literal'] == [list(row) for row in build.prototype_matrix(role)]
        for item in record['artifacts'].values():
            assert build.binding(item['path']) == item
            pins[item['path']] = item['sha256']
        pins[str(bp)], pins[str(cp)] = sha(bp), sha(cp)
    assert sha(BD / 'native_v2.py') == NATIVE_SHA
    pins[str(BD / 'native_v2.py')] = NATIVE_SHA
    store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    assert sha(store / 'manifest.json') == STORE_SHA
    pins[str(store / 'manifest.json')] = STORE_SHA
    assert sha(HERE / 'WIDE_SCHEDULING_V1.md') == SCHEDULE_SHA
    pins[str(HERE / 'WIDE_SCHEDULING_V1.md')] = SCHEDULE_SHA
    pins[str(Path(__file__).resolve())] = sha(__file__)
    return pins


def native_inputs(candidate, args):
    pins = provenance()
    assert candidate == CANDIDATES[args.role][0].resolve()
    # This prerequisite is itself an exclusive sealed numerical result. Its
    # exact source/fixture lineage is checked again by the wide judge.
    rp, cp = HERE / 'stage_native/result.json', HERE / 'stage_native/completion.json'
    assert isinstance(args.tiny_sha256, str) and len(args.tiny_sha256) == 64
    assert sha(rp) == args.tiny_sha256
    tiny, completion = json.loads(rp.read_text()), json.loads(cp.read_text())
    assert tiny['passed'] and tiny['completed'] and completion['passed'] and completion['completed']
    assert completion['result_sha256'] == args.tiny_sha256 and tiny['stage'] == 'native'
    assert tiny['schema'] == 'birth_direction_node_head_tiny_v1'
    assert tiny['no_network_or_target_access'] and tiny['no_fit'] and tiny['sources_unchanged']
    for item in tiny['source_bindings'].values():
        assert sha(item['path']) == item['sha256']
        pins[item['path']] = item['sha256']
    assert set(tiny['artifacts']) == {'arrays.npz'}
    for item in tiny['artifacts'].values():
        assert sha(item['path']) == item['sha256']
        pins[item['path']] = item['sha256']
    for p in (rp, cp, HERE / 'check_tiny.py'):
        pins[str(p)] = sha(p)
    return pins, dict(role=args.role, tiny_native_result_sha256=args.tiny_sha256,
                      fixed_ranks=[35, 100], own_complete_bills=True,
                      completed_fit_required=False, truth_accessed=False,
                      full_tiny_judge_still_required_before_wide_judgment=True)


def derived_main():
    path = BD / 'native_v2.py'
    assert sha(path) == NATIVE_SHA
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    first = original.index("    parser.add_argument('--source',type=Path)\n")
    last = original.index('    assert Path(sys.prefix)', first)
    parser = ("    parser.add_argument('--role',choices=('parent','zero','nonzero'))\n"
              "    parser.add_argument('--source-sha256')\n"
              "    parser.add_argument('--tiny-sha256')\n"
              "    args=parser.parse_args()\n"
              "    if args.role is None: parser.print_help();return 0\n"
              "    assert sha(__file__)==args.source_sha256, 'Reviewed wide native source pin required'\n"
              "    args.source=CANDIDATES[args.role][0]\n"
              "    args.split='development'\n"
              "    args.out=HERE/f'wide_native_{args.role}_v1'\n"
              "    assert not args.out.exists(), 'Exclusive acquisition; no resume'\n")
    a, b = original.index('    if candidate==BANK:\n'), original.index('    manifest=json.loads(')
    edits = [
        (original[first:last], parser),
        ("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n"
         "        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}",
         "    inputs,prerequisite=native_inputs(candidate,args)"),
        (original[a:b], '    ranks=(35,100)\n'),
        ('        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,',
         '        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,\n'
         "        schema='birth_direction_node_head_native_wide_v1',role=args.role,prerequisites=prerequisite,\n"
         '        state_instrumentation=False,own_complete_bills=True,score_measured=False,'),
    ]
    source = original
    for old, new in edits:
        assert source.count(old) == 1
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        assert inverse.count(new) == 1
        inverse = inverse.replace(new, old)
    assert inverse == original
    begin = '            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n'
    end = '        assert all(sha(p)==v for p,v in inputs.items())\n'
    assert source[source.index(begin):source.index(end)] == original[original.index(begin):original.index(end)]
    ast.parse(source)
    return source


def main():
    if '--role' not in sys.argv:
        print('Inert. --role parent|zero|nonzero --source-sha256 SHA --tiny-sha256 SHA')
        return 0
    os.chdir(NEW)
    source = derived_main()
    base = module(BD / 'native_v2.py', 'bdnh_preserved_native_wide')
    env = dict(vars(base), __file__=str(Path(__file__).resolve()), HERE=HERE,
               native_inputs=native_inputs, CANDIDATES=CANDIDATES)
    exec(compile(source, str(__file__) + ':preserved_main', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
