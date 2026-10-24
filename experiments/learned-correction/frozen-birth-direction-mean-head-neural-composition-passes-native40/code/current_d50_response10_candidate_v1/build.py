"""Inert lexical-only build: frozen P1, one response selector .01 -> .1.

CLI: --run --source-sha256 HASH --protocol-sha256 HASH.
Only build_v1/{estimator.py,null_estimator.py,result.json,completion.json}
may be created, exclusively. No numerical imports, payload/weight/prediction
array decoding, target access, estimator execution, fit, or next-stage launch.

Historical qualification is bound by exact result/completion receipts and
their Python/Markdown source owners, not by reopening historical numerical
arrays or rerunning their judges. The sole source input is qualified P1's
complete literal estimator. New acquisition must own its own inputs/bills.

verify_build(build_sha256,candidate_sha256) is a read-only stdlib interface
for the native worker: returns build, source_bindings, input_bindings, and
candidates {parent|null|terminal:{path,sha256}}. It redoes the exact inverse
proof from source bytes. Neither import nor verification imports an estimator.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P1 = NEW / 'current_d50_native_objective_p1_v1'
CENSUS = NEW / 'current_d50_response10_census_v1'
PARENT = P1 / 'build_v1/estimator.py'
PARENT_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
PROTOCOL_SHA = 'c931293ede873bf193dec3229b7a02583a9ec96a2e2d95996550b9de20861d7f'
SCHEMA = 'current_d50_response10_candidate_build_v1'
OLD = b'response_indices, response_mask = select(response_score, gate_floor)'
NEW_SELECTOR = b'response_indices, response_mask = select(response_score, 0.1)'

# Exact completed evidence, not interchangeable inputs or CLI-selectable arms.
RECEIPTS = {
    'p1_build': (P1 / 'build_v1',
        '877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1',
        'cdcb0a934224f242cad404dec3bae7eba60a712d8bd430ace0c7ab7111faea5c'),
    'p1_native40_score': (P1 / 'score2_v1',
        '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
        'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587'),
    'p1_native40_watch': (P1 / 'score2_watch_v1',
        'c9d015f7ed6b1da7db58bcf237582874042dd2d1e3315e72e7c55b5f1b9afa34',
        'fa220095fa711c7e80a9268aeb2e0ef410fb94444e775813e758d52eaa01d4f5'),
    'p1_independent14_score': (P1 / 'panel_v1/score',
        '81ebbb54207e4f0060aaa1a91320c17de89b6fdd53e7d5326c17993829e6e526',
        'ee2df444d7e3984993e6367f707b3ffd3a8408a182f85ca9dc7701341167f309'),
    'p1_independent14_watch': (P1 / 'panel_v1/score_watch',
        '46e3e7e910c5b4b50f1e8c41a99c417e7a751f82fd7cc4b37fe6e2576f3f10b9',
        '2e26c1b9c1aa38ec98148a835dbf28b31f06288b3abb27f0927f1452ce0ff50f'),
    'd50_response10_census_price': (CENSUS / 'price_v1',
        'ac147cc8e08523187f5a0be19a80bac59547d2143d885eaada5d3ea93e90a1ba',
        '5a4050a7a77e68f235729f899ccdbad5dc61d422587ace810af8dfaf3d744209'),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and
            set(value) <= set('0123456789abcdef'), 'Lower-case SHA256 required')
    return value


def sha(path):
    path = Path(path).resolve()
    # This builder never reads even the bytes of a historical numerical array.
    require(path.suffix.lower() not in ('.npy', '.npz'), 'No numerical array I/O')
    require('truthpost' not in path.parts, 'No target I/O')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    require(Path(path).suffix == '.json', 'Metadata JSON only')
    return json.loads(Path(path).read_text())


def bind(mapping):
    for path, expected in mapping.items():
        require(sha(path) == digest(expected), 'Immutable binding: ' + str(path))


def merge(mapping, path, expected):
    path = str(Path(path).resolve())
    require(path not in mapping or mapping[path] == expected, 'Conflicting source binding')
    mapping[path] = digest(expected)


def _admission(source_sha256, protocol_sha256):
    require(digest(protocol_sha256) == PROTOCOL_SHA, 'The sole prospective protocol')
    sources = {str(Path(__file__).resolve()): digest(source_sha256),
               str(HERE / 'PROTOCOL.md'): PROTOCOL_SHA, str(PARENT): PARENT_SHA}
    inputs, records = {}, {}
    bind(sources)
    for name, (folder, result_sha, completion_sha) in RECEIPTS.items():
        for filename, expected in (('result.json', result_sha), ('completion.json', completion_sha)):
            merge(inputs, folder / filename, expected)
        require(sha(folder / 'result.json') == result_sha and
                sha(folder / 'completion.json') == completion_sha, 'Fixed qualification receipt: ' + name)
        record, end = read(folder / 'result.json'), read(folder / 'completion.json')
        require(all(record.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged'))
                and end.get('passed') is True and end.get('completed') is True
                and end.get('result_sha256') == result_sha, 'Completed evidence: ' + name)
        owners = record.get('source_bindings')
        require(isinstance(owners, dict) and owners, 'Receipt source owners: ' + name)
        for path, expected in owners.items():
            require(Path(path).is_absolute(), 'Absolute recorded owner')
            if Path(path).suffix.lower() in ('.py', '.md'):
                merge(sources, path, expected)
        records[name] = record
    built = records['p1_build']
    require(built['schema'] == 'current_d50_native_objective_p1_v1' and built['stage'] == 'build'
            and built['candidate_sha256'] == PARENT_SHA
            and built['artifacts'][str(PARENT)] == PARENT_SHA
            and built['modified_assignment_only'] == '_GNN_LITERAL'
            and all(built.get(k) is True for k in ('inverse_source_byte_exact',
                'cstar_byte_exact', 'terminal_literal_roundtrip_byte_exact', 'sources_unchanged')),
            'Sole independently checked literal P1 parent')
    forty, panel = records['p1_native40_score'], records['p1_independent14_score']
    require(forty['schema'] == 'current_d50_native_objective_p1_score_v2' and
            forty['stage'] == 'score' and forty['candidate_sha256'] == PARENT_SHA and
            all(forty.get(k) is True for k in ('integrity_passed',
                'all_predictions_verified_before_truth', 'advances_to_independent14')),
            'Qualified advancing actual-native40 P1')
    require(panel['schema'] == 'current_d50_native_objective_p1_independent14_v1' and
            panel['mode'] == 'score' and panel['candidate_sha256'] == PARENT_SHA and
            all(panel.get(k) is True for k in ('integrity_passed', 'qualification_passed',
                'all_predictions_verified_before_truth', 'retain_local_component', 'sources_unchanged')),
            'Same P1 independently retained on both groups')
    for child_name, watch_name, expected_schema in (
        ('p1_native40_score', 'p1_native40_watch', 'current_d50_native_objective_p1_native_watch_v2'),
        ('p1_independent14_score', 'p1_independent14_watch',
         'current_d50_native_objective_p1_independent14_watch_v1')):
        watch = records[watch_name]
        require(watch['schema'] == expected_schema and watch['stage'] == 'score' and
                watch['child_result_sha256'] == RECEIPTS[child_name][1] and
                watch['child_completion_sha256'] == RECEIPTS[child_name][2] and
                all(watch.get(k) is True for k in ('child_execution_completed',
                    'whole_owned_tree_retired', 'sources_unchanged')), 'Watched qualification: ' + watch_name)
    census = records['d50_response10_census_price']
    require(census['schema'] == 'current_d50_response10_price_stage_v1' and
            census['candidate_executed'] is False and
            census['price']['both_pass_eight_percent_information_gate'] is True and
            census['price']['frozen_trajectory_only'] is True and
            census['price']['candidate_complete_bill_claim'] is False,
            'Completed fixed .1 cost-information precedent, not a candidate bill')
    bind(sources)
    return sources, inputs


def _span(source, node):
    lines = source.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno - 1])) + node.col_offset,
            sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset)


def _selector(source, changed=False):
    tree = ast.parse(source)
    owners = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == '_predict_binary']
    require(owners, 'Active prediction owner exists')
    owner = owners[-1]
    candidates = []
    for node in ast.walk(owner):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and all(isinstance(x, ast.Name) for x in target.elts):
                if [x.id for x in target.elts] == ['response_indices', 'response_mask']:
                    candidates.append(node)
    require(len(candidates) == 1, 'Sole response assignment in the active function')
    node = candidates[0]
    start, stop = _span(source, node)
    expected = NEW_SELECTOR if changed else OLD
    require(source[start:stop] == expected and source.count(expected) == 1,
            'Exact unique lexical response-selector statement')
    call = node.value
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and
            call.func.id == 'select' and len(call.args) == 2 and not call.keywords and
            isinstance(call.args[0], ast.Name) and call.args[0].id == 'response_score',
            'Original selector and response-score owner')
    defaults = dict(zip([x.arg for x in owner.args.args][-len(owner.args.defaults):], owner.args.defaults))
    require(isinstance(defaults['gate_floor'], ast.Constant) and
            defaults['gate_floor'].value == .01, 'Active/birth default floor remains .01')
    return tree, owner, start, stop


def construct(parent):
    """Pure byte/AST proof. No evaluation of P literals or estimator code."""
    require(hashlib.sha256(parent).hexdigest() == PARENT_SHA, 'Exact complete P1 source')
    tree, owner, start, stop = _selector(parent)
    candidate = parent[:start] + NEW_SELECTOR + parent[stop:]
    new_tree, new_owner, a, b = _selector(candidate, changed=True)
    require(candidate[:a] + OLD + candidate[b:] == parent, 'Exact whole-source inverse')
    require((start, owner.lineno) == (a, new_owner.lineno), 'Same active source ownership')
    protected = {}
    for name in ('_GNN_LITERAL', '_BD_COEFFICIENT_LITERAL'):
        def assignments(data, parsed):
            return [data[s:e] for node in parsed.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
                    for s, e in [_span(data, node)]]
        old_values, new_values = assignments(parent, tree), assignments(candidate, new_tree)
        require(old_values and old_values == new_values, 'Every frozen literal owner: ' + name)
        protected[name] = [hashlib.sha256(value).hexdigest() for value in old_values]
    # Syntax compilation only: no code object is executed or imported.
    compile(tree, str(PARENT), 'exec')
    compile(new_tree, str(HERE / 'build_v1/estimator.py'), 'exec')
    proof = dict(active_predict_line=owner.lineno, modified_statement_line=parent[:start].count(b'\n') + 1,
        original_statement=OLD.decode(), replacement_statement=NEW_SELECTOR.decode(),
        modified_statement_count=1, inverse_source_byte_exact=True, null_source_byte_exact=True,
        all_nonresponse_source_byte_exact=True, all_P1_payload_source_byte_exact=True,
        cstar_source_byte_exact=True, active_birth_floors_unchanged=True,
        full_carrier_unchanged=True, exact_diagonal_unchanged=True,
        omitted_radial_source_unchanged=True, packed_schedule_unchanged=True,
        protected_assignment_sha256=protected, syntax_compiled_without_execution=True,
        unchanged_prefix_sha256=hashlib.sha256(parent[:start]).hexdigest(),
        unchanged_suffix_sha256=hashlib.sha256(parent[stop:]).hexdigest())
    return candidate, proof


def verify_build(build_sha256, candidate_sha256):
    folder = HERE / 'build_v1'
    require(sha(folder / 'result.json') == digest(build_sha256), 'Sealed response10 build')
    result, end = read(folder / 'result.json'), read(folder / 'completion.json')
    require(result['schema'] == SCHEMA and result['stage'] == 'build' and
            all(result.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged', 'sources_unchanged'))
            and end.get('passed') is True and end.get('completed') is True
            and end.get('result_sha256') == build_sha256, 'Completed source-only build')
    sources, inputs = _admission(result['source_bindings'][str(Path(__file__).resolve())], PROTOCOL_SHA)
    require(result['source_bindings'] == sources and result['input_bindings'] == inputs,
            'Exact prospective source and historical qualification bindings')
    parent = PARENT.read_bytes()
    candidate, proof = construct(parent)
    require(result['proof'] == proof and result['parent_sha256'] == PARENT_SHA and
            result['null_sha256'] == PARENT_SHA and result['candidate_sha256'] == digest(candidate_sha256)
            == hashlib.sha256(candidate).hexdigest(), 'Exact sole candidate and inverse proof')
    expected = {str(folder / 'estimator.py'): candidate_sha256,
                str(folder / 'null_estimator.py'): PARENT_SHA}
    require(result['artifacts'] == expected and (folder / 'estimator.py').read_bytes() == candidate
            and (folder / 'null_estimator.py').read_bytes() == parent, 'Built source bytes')
    bind(sources); bind(inputs); bind(expected)
    for path, expected_sha in expected.items():
        merge(sources, path, expected_sha)
    merge(inputs, folder / 'result.json', build_sha256)
    merge(inputs, folder / 'completion.json', sha(folder / 'completion.json'))
    return dict(build=result, source_bindings=sources, input_bindings=inputs,
        candidates=dict(parent=dict(path=str(PARENT), sha256=PARENT_SHA),
                        null=dict(path=str(folder / 'null_estimator.py'), sha256=PARENT_SHA),
                        terminal=dict(path=str(folder / 'estimator.py'), sha256=candidate_sha256)))


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(sys.flags.optimize == 0, 'No optimized-away assertions')
    source_sha, protocol_sha = digest(args.source_sha256), digest(args.protocol_sha256)
    out = HERE / 'build_v1'
    require(not out.exists(), 'Exclusive build directory; never overwrite')
    began = time.monotonic()
    def expired(_signum, _frame):
        raise TimeoutError('Lexical builder60s limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    sources, inputs = _admission(source_sha, protocol_sha)
    parent = PARENT.read_bytes()
    candidate, proof = construct(parent)
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope', 'whestbench')),
            'Strict stdlib-only lexical build')
    bind(sources); bind(inputs)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != 'darwin':
        peak *= 1024
    require(time.monotonic() - began < 60 and peak < 2_000_000_000, 'Builder60s/2GB')
    result = dict(schema=SCHEMA, stage='build', passed=True, completed=True,
        hashes_unchanged=True, sources_unchanged=True, source_bindings=sources, input_bindings=inputs,
        parent_sha256=PARENT_SHA, null_sha256=PARENT_SHA,
        candidate_sha256=hashlib.sha256(candidate).hexdigest(), proof=proof,
        unchanged_head_not_readapted=True, response_threshold='.1', active_threshold='.01',
        birth_threshold='.01', historical_qualification_receipts_verified=True,
        historical_numerical_arrays_not_reopened=True, target_arrays_accessed=False,
        numerical_array_imports=False, estimator_imported=False, estimator_executed=False,
        native_execution_admitted=False, production_qualified=False, official_score=False,
        wall_s=time.monotonic()-began, peak_rss_bytes=peak,
        created_utc=datetime.now(timezone.utc).isoformat())
    result.update(proof)  # flat compatibility flags, with the same full proof retained
    out.mkdir(exist_ok=False)
    for filename, data in (('estimator.py', candidate), ('null_estimator.py', parent)):
        with (out / filename).open('xb') as stream:
            stream.write(data)
    result['artifacts'] = {str(out / name): sha(out / name)
                           for name in ('estimator.py', 'null_estimator.py')}
    _write(out / 'result.json', result)
    _write(out / 'completion.json', dict(passed=True, completed=True,
                                         result_sha256=sha(out / 'result.json')))
    signal.alarm(0)
    print(json.dumps({key: result[key] for key in ('passed', 'candidate_sha256', 'wall_s')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
