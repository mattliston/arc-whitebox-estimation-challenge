"""One fixed native P probe through the unchanged official-input BD meter.

Inert unless --run is supplied with reviewed source/build/protocol hashes.
Only candidate admission is replaced in the pinned native_v2 source.  Its
__file__, HERE, input/prediction budgets, numeric loop and cleanup stay intact.
No hybrid/model module is imported and no truth or probe array is decoded.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW / 'birth_direction_fit_v1'
HY = NEW / 'current_d50_hybrid_cp_qualification_v1'
CO = NEW / 'current_bd_p_coadaptation_v1'
ORIGINAL = BD / 'native_v2.py'
ORIGINAL_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
BUILD = HERE / 'build_probes_v1/build.json'
CASES = ('null', 'plus025', 'minus025', 'plus050', 'minus050')
INITIALIZATION = 1383913786
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
PROBE_SHA = 'b21b70ee6571ce908a2f37907dc705a23192c78fd5f1e8ab6c859cca94b4e4be'
STAGES = ('mps_forward', 'cpu_gradient', 'mps_gradient')
STAGE_SHA = dict(
    mps_forward='d6edae6b5828e09939d5afb7be7a3bc619f8e384fd909cbf105b26331a2943fc',
    cpu_gradient='acfe54dfaffc19ad1d1c2c56c2349750ceff0cf4a587b72638827c7767abea0e',
    mps_gradient='f821503ed45372d6b9a14b363914ab255d643b4ae2d90b3eacb389f61d6892ec')
WATCH_SHA = dict(
    mps_forward='288ed547f544a9ec749129f66e88e0655dcca6ec5125fec60d1e7885487d8c1d',
    cpu_gradient='4d880e1b185767feb3db7543c506bf25af832fe83cbdc209822289f8f444261f',
    mps_gradient='f2d319892277473a13cc9d1a21b4a51754682842ab391dc55b9b301cebc19afd')
FIXED = {
    ORIGINAL: ORIGINAL_SHA,
    PARENT: PARENT_SHA,
    HERE / 'build_probes.py': '395c7a1a9dabdd324bb2fd1b82cddc733f4eeee96b101e458cf3db9b499756ac',
    HERE / 'BUILDER_INTERFACE.md': 'fa9f341e2d9fedf7cee1efb1b6a3f49d0b2c3d62dec9999f589780cb98b16ffd',
    BD / 'PROTOCOL.md': '9638ccee8e75e830f205b5cc8304d7333c7b202ae5441db3a71f610645f94212',
    BD / 'NATIVE_SCOPE_CLARIFICATION.md': 'b43aa00adcc30f434f5e301abcd490deb220020096ea47daebcf908f420bd333',
    BD / 'NATIVE_INPUT_AND_LOG_LIFECYCLE_V2.md': 'c08620d45fbcd93385f26c882fbc6c56e988673a19fe89484bc66dcdd2ba5dfa',
    HY / 'model.py': 'c8cb5cfe961b2df2253ca660464793308f12769f97a039fcdf0325c151f1b025',
    HY / 'qualify.py': '2d096569a4622f939facb64117bbe6b9e1408aa5d66eac0f68c2b633d44d3ce9',
    HY / 'watch.py': '5574b0fb3bdd7e8ac8c556521d5f79149785ba6285f345b3fa99d82e62e5d665',
    HY / 'PROTOCOL.md': 'c6281a3fe713798d351684683ae01afe3c144618c5bf3e4db4bff46796d4c9ab',
    CO / 'watch.py': 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b',
    CO / 'stage_cpu_forward_v1/result.json': CPU_SHA,
    CO / 'stage_cpu_forward_v1/completion.json': '716a7f3e310fea326e7d2ed5313440da88d8eb5d57ce5f6ddf7e99803e26a61c',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require(not (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower())
            and 'truthpost' not in path.parts, 'No truth-array authentication or access')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def assignment_span(source, name):
    """Locate a unique top-level assignment, without evaluating its literal."""
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(nodes) == 1 and len(nodes[0].targets) == 1, 'Unique assignment: ' + name)
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno - 1])) + node.col_offset,
            sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset)


def adapted_native_source():
    """Stdlib-only source construction with a literal whole-source inverse."""
    require(sha(ORIGINAL) == ORIGINAL_SHA, 'Frozen official-input meter changed')
    original = ORIGINAL.read_text()
    start = original.index('    if candidate==BANK:\n')
    stop = original.index("    ranks=(35,100) if args.split=='development' else VALID\n", start)
    removed = original[start:stop]
    replacement = '    _bridge_candidate_admission(candidate,inputs)\n'
    adapted = original[:start] + replacement + original[stop:]
    require(adapted[:start] + removed + adapted[start + len(replacement):] == original,
            'Native whole-source inverse failed')
    begin = '            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n'
    end = '        assert all(sha(p)==v for p,v in inputs.items())\n'
    loop = original[original.index(begin):original.index(end)]
    require(adapted[adapted.index(begin):adapted.index(end)] == loop,
            'Input/predict/bill/reset loop is not byte-exact')
    proof = dict(native_original_sha256=ORIGINAL_SHA,
                 native_adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                 native_loop_sha256=hashlib.sha256(loop.encode()).hexdigest(),
                 removed_admission_sha256=hashlib.sha256(removed.encode()).hexdigest(),
                 source_inverse_verified=True, numeric_loop_byte_exact=True,
                 native_file=str(ORIGINAL), native_here=str(BD),
                 only_source_edit='candidate/build admission branch')
    return adapted, proof


def authenticate(case, source_sha256, build_sha256, hybrid_sha256, protocol_sha256):
    """Read-only metadata verifier: return (flat source pins, admission record)."""
    require(case in CASES and hybrid_sha256 == STAGE_SHA['mps_gradient'], 'Fixed case/hybrid only')
    pins = {}

    def pin(path, digest, metadata=None):
        path = Path(path).resolve()
        key = str(path)
        require(path.is_relative_to(NEW.parent), 'Binding outside fixed Phase2 workspace')
        require(isinstance(digest, str) and len(digest) == 64, 'SHA256 required: ' + key)
        if key in pins:
            require(pins[key] == digest, 'Conflicting binding: ' + key)
        else:
            require(sha(path) == digest, 'Changed binding: ' + key)
            pins[key] = digest
        if metadata is not None:
            stat = path.stat()
            require(metadata['path'] == key and metadata['sha256'] == digest
                    and metadata['bytes'] == stat.st_size and metadata['mtime_ns'] == stat.st_mtime_ns,
                    'Changed file metadata: ' + key)

    def bind_map(mapping):
        require(isinstance(mapping, dict), 'Expected flat binding map')
        for path, item in mapping.items():
            pin(path, item['sha256'], item) if isinstance(item, dict) else pin(path, item)

    def sealed(folder, digest, schema, stage):
        rp, cp = folder / 'result.json', folder / 'completion.json'
        pin(rp, digest)
        record, completion = json.loads(rp.read_text()), json.loads(cp.read_text())
        require(record['schema'] == schema and record['stage'] == stage
                and record['passed'] is True and record['completed'] is True
                and record['sources_unchanged'] is True and 'error' not in record
                and completion['passed'] is True and completion['completed'] is True
                and completion['result_sha256'] == digest, 'Unqualified stage: ' + str(folder))
        pin(cp, sha(cp))
        for name in ('source_bindings', 'source_metadata', 'weight_bindings',
                     'dependencies', 'artifacts', 'artifact_bindings'):
            if name in record:
                bind_map(record[name])
        return record

    for path, digest in FIXED.items():
        pin(path, digest)
    pin(Path(__file__), source_sha256)
    pin(HERE / 'PROTOCOL.md', protocol_sha256)
    pin(HERE / 'NATIVE_INTERFACE.md', sha(HERE / 'NATIVE_INTERFACE.md'))
    pin(HERE / 'watch.py', sha(HERE / 'watch.py'))
    pin(BUILD, build_sha256)
    build = json.loads(BUILD.read_text())
    require(build['schema'] == 'current_d50_native_p_probe_builds_v1'
            and build['passed'] is True and build['completed'] is True
            and build['cases'] == list(CASES) and set(build['candidates']) == set(CASES)
            and build['parent_sha256'] == PARENT_SHA and build['cpu_forward_result_sha256'] == CPU_SHA
            and build['probe_seal_sha256'] == PROBE_SHA, 'Fixed completed five-source build required')
    for name in ('all_five_sources_complete', 'all_source_inverses_byte_exact',
                 'null_source_byte_exact', 'non_PKEYS_array_bytes_preserved',
                 'unused_layer0_array_bytes_preserved', 'cstar_source_byte_exact'):
        require(build[name] is True, 'Missing build proof: ' + name)
    require(build['no_fit'] is True and build['parameter_updates'] == 0
            and build['targets_accessed'] is False and build['card_calls'] == 0
            and build['payloads_regenerated'] is False, 'Builder is not target-free literal-only')
    bind_map(build['source_bindings'])
    bind_map(build['artifacts'])
    expected_artifacts = set()
    for item_case in CASES:
        item = build['candidates'][item_case]
        folder = BUILD.parent / item_case
        require(item['path'] == str(folder / 'estimator.py')
                and item['build_path'] == str(folder / 'build.json'), 'Canonical candidate paths')
        expected_artifacts.update((item['path'], item['build_path']))
        require(build['artifacts'][item['path']] == item['sha256']
                and build['artifacts'][item['build_path']] == item['build_sha256'], 'Candidate seal')
        receipt = json.loads(Path(item['build_path']).read_text())
        require(receipt['schema'] == 'current_d50_native_p_probe_build_v1'
                and receipt['passed'] is True and receipt['completed'] is True
                and receipt['case'] == item_case and receipt['candidate_sha256'] == item['sha256']
                and receipt['parent_sha256'] == PARENT_SHA
                and receipt['payload']['path'] == item['payload_path']
                and receipt['payload']['sha256'] == item['payload_sha256']
                and receipt['source_bindings'] == build['source_bindings'], 'Per-case build ownership')
        require(all(receipt[k] is True for k in ('inverse_source_byte_exact',
                    'all_other_parent_source_byte_exact', 'payload_literal_array_byte_exact',
                    'cstar_source_byte_exact')), 'Per-case source/literal proof')
    require(set(build['artifacts']) == expected_artifacts, 'Exact ten build artifacts')
    candidate = Path(build['candidates'][case]['path'])
    parent, source = PARENT.read_bytes(), candidate.read_bytes()
    a, b = assignment_span(parent, '_GNN_LITERAL')
    c, d = assignment_span(source, '_GNN_LITERAL')
    require(a == c and source[:c] + parent[a:b] + source[d:] == parent,
            'Independent selected-candidate whole-source inverse')
    if case == 'null':
        require(source == parent, 'Null is not byte-exact d50')

    stage_seals = {}
    for stage in STAGES:
        child = sealed(HY / ('stage_' + stage + '_v1'), STAGE_SHA[stage],
                       'current_d50_hybrid_cp_qualification_v1', stage)
        require(child['alltruth_accessed'] is False and child['training_targets_accessed'] is False
                and child['validation_targets_accessed'] is False and child['no_fit'] is True
                and child['parameter_updates'] == 0 and child['probe_owner_sha256'] == CPU_SHA
                and child['probe_seal_sha256'] == PROBE_SHA, 'Target-free original probes required')
        require(set(child['payloads']) == set(CASES), 'Exact hybrid payload roster')
        for item_case in CASES:
            item = build['candidates'][item_case]
            require(child['payloads'][item_case]['path'] == item['payload_path']
                    and child['payloads'][item_case]['sha256'] == item['payload_sha256'],
                    'Build/hybrid physical payload mismatch')
        if stage == 'mps_gradient':
            require(child['numerical_qualification_passed'] is True, 'Complete P gradient gate')
        watch = sealed(HY / ('watch_' + stage + '_v1'), WATCH_SHA[stage],
                       'current_d50_hybrid_cp_watch_v1', stage)
        require(watch['child_result_sha256'] == STAGE_SHA[stage]
                and watch['child_exit_code'] == 0 and watch['child_completed'] is True
                and watch['terminated'] is False, 'Child PASS alone does not admit native acquisition')
        stage_seals[stage] = dict(result_sha256=STAGE_SHA[stage], watch_sha256=WATCH_SHA[stage])
        del child, watch
    _, proof = adapted_native_source()
    record = dict(schema='current_d50_native_p_bridge_admission_v1', passed=True,
                  case=case, candidate=str(candidate), candidate_sha256=sha(candidate),
                  build_sha256=build_sha256, hybrid_sha256=hybrid_sha256,
                  protocol_sha256=protocol_sha256, stage_seals=stage_seals,
                  source_bindings=dict(pins), **proof, fixed_ranks=[35, 100, 35],
                  expected_initialization_flops=INITIALIZATION, alltruth_accessed=False,
                  no_fit=True, score_measured=False, bridge_numerically_qualified=False)
    return pins, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', choices=CASES)
    for name in ('source-sha256', 'build-sha256', 'hybrid-sha256', 'protocol-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1',
            'Assertions enabled and no bytecode writes')
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope', 'whestbench')),
            'Fresh process before native metadata admission')
    out = HERE / ('run_' + str(args.case) + '_v1')
    admission_path = HERE / ('admission_' + str(args.case) + '_v1.json')
    require(not out.exists() and not admission_path.exists(), 'Exclusive case; no overwrite/retry')
    pins, admission = authenticate(args.case, args.source_sha256, args.build_sha256,
                                   args.hybrid_sha256, args.protocol_sha256)
    adapted, proof = adapted_native_source()
    require(all(admission[k] == v for k, v in proof.items()), 'Source proof changed')
    admission['created_utc'] = datetime.now(timezone.utc).isoformat()
    write(admission_path, admission)
    pins[str(admission_path)] = sha(admission_path)
    candidate = Path(admission['candidate'])

    def candidate_admission(actual, inputs):
        require(actual == candidate and sha(actual) == admission['candidate_sha256'], 'Fixed candidate')
        for path, digest in pins.items():
            require(sha(path) == digest and (path not in inputs or inputs[path] == digest),
                    'Admission input changed before estimator import: ' + path)
            inputs[path] = digest

    # Keep the qualified module's original path-derived globals.  Only the
    # explicit admission callback is injected; no imported downstream draft.
    namespace = dict(__name__='native_p_bridge_owned_meter', __file__=str(ORIGINAL),
                     _bridge_candidate_admission=candidate_admission)
    exec(compile(adapted, str(ORIGINAL), 'exec'), namespace)
    require(namespace['HERE'] == BD and namespace['PARENT'] == NEW, 'Original meter path ownership')
    gc.collect()
    saved_argv = sys.argv
    try:
        sys.argv = [str(ORIGINAL), '--source', str(candidate), '--split', 'development', '--out', str(out)]
        status = namespace['main']()
    finally:
        sys.argv = saved_argv
    if status != 0:
        return 1
    result = json.loads((out / 'result.json').read_text())
    require(result['passed'] is True and result['hashes_unchanged'] is True and 'error' not in result
            and [trial['rank'] for trial in result['trials']] == [35, 100, 35]
            and result['initialization_flops'] == INITIALIZATION, 'Native completion/init gate')
    require('torch' not in sys.modules, 'Native acquisition must not import Torch')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
