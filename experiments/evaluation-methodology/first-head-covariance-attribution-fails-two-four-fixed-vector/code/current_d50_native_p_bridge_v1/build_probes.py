"""Build five target-free d50 P probes; inert without --build and reviewed hashes.

Only the fixed original/P-probe NPZ files are decoded, after metadata admission.
No estimator, Torch, FNP, weights, targets, optimizer or network is executed.
This builder does not admit native execution, training or a score.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CO = NEW / 'current_bd_p_coadaptation_v1'
CPU = CO / 'stage_cpu_forward_v1'
OUT = HERE / 'build_probes_v1'
NOTE = HERE / 'BUILDER_INTERFACE.md'
PARENT = NEW / 'birth_direction_fit_v1/build_terminal_v1/estimator.py'
ORIGINAL = NEW / 'payload_response_rows_v1.npz'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
ORIGINAL_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
CPU_COMPLETION_SHA = '716a7f3e310fea326e7d2ed5313440da88d8eb5d57ce5f6ddf7e99803e26a61c'
PROBE_SEAL_SHA = 'b21b70ee6571ce908a2f37907dc705a23192c78fd5f1e8ab6c859cca94b4e4be'
QUALIFIER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
CO_MODEL_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
CSTAR_SHA = 'd622c0aed6d62e0453648df32cfe46da73b220163fb5fe32877d430be1540227'
CASES = ('null', 'plus025', 'minus025', 'plus050', 'minus050')
AMPLITUDES = dict(null=0.0, plus025=.025, minus025=-.025, plus050=.05, minus050=-.05)
PAYLOAD_SHA = {
    'null': ORIGINAL_SHA,
    'plus025': '621e646402499443b5b29129171288916e1dd556e6c6d5fa95a17df3e474484b',
    'minus025': 'b7be4369cda27389600090961b1bf4e8b4b0f972894739768d3ba3aa6a413d21',
    'plus050': '99cc37aa5a856f09d678be965126624060d258abc5536bf371bd4d813c9fe841',
    'minus050': '86faf9b52f05070697b1255701256e2d30bfa29d748d618772371618dcf843db',
}
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
FROZEN_KEYS = ('centers', 'scales', 'rms', 'first')
INACTIVE_KEYS = ('head', 'head_b', 'gain')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def unchanged(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, 'Builder input changed: ' + path)


def admission(source_sha256, cpu_forward_sha256):
    """Authenticate source and fixed probe metadata; no numerical imports."""
    require(cpu_forward_sha256 == CPU_SHA, 'Only the frozen passed CPU probe owner')
    require(isinstance(source_sha256, str) and len(source_sha256) == 64,
            'Reviewed builder SHA256 is required')
    pins = {}

    def pin(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(NEW), 'Only fixed local source/probe inputs')
        actual = sha(path)
        require(actual == expected, 'Source/probe digest mismatch: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == actual, 'Conflicting binding')
        pins[str(path)] = actual

    pin(Path(__file__), source_sha256)
    pin(NOTE, sha(NOTE))
    pin(PARENT, PARENT_SHA)
    pin(ORIGINAL, ORIGINAL_SHA)
    pin(CPU / 'result.json', CPU_SHA)
    pin(CPU / 'completion.json', CPU_COMPLETION_SHA)
    record = json.loads((CPU / 'result.json').read_text())
    complete = json.loads((CPU / 'completion.json').read_text())
    require(record['schema'] == 'current_d50_p_coadaptation_qualification_v1'
            and record['stage'] == 'cpu_forward' and record['probe_owner_stage'] == 'cpu_forward',
            'Fixed CPU qualification schema/owner')
    require(record['passed'] and record['completed'] and record['sources_unchanged']
            and complete['passed'] and complete['completed'] and complete['result_sha256'] == CPU_SHA,
            'Completed, unchanged CPU probe receipt required')
    require(record['alltruth_accessed'] is False and record['training_targets_accessed'] is False
            and record['validation_targets_accessed'] is False and record['no_fit']
            and record['parameter_updates'] == 0, 'Target-free, untrained probe owner')
    require(record['source_bindings'][str(CO / 'qualify.py')] == QUALIFIER_SHA
            and record['source_bindings'][str(CO / 'model.py')] == CO_MODEL_SHA,
            'Frozen probe generator/model')
    for path, digest in record['source_bindings'].items():
        p = Path(path).resolve()
        require(p.suffix in ('.py', '.md', '.json') or p == ORIGINAL,
                'Source authentication must not open target or weight arrays')
        pin(p, digest)
    seal_path = CPU / 'probes_sealed_before_card_calls.json'
    require(record['probe_seal_sha256'] == PROBE_SEAL_SHA, 'Original pre-card seal')
    pin(seal_path, PROBE_SEAL_SHA)
    seal = json.loads(seal_path.read_text())
    require(seal['card_calls'] == 0 and seal['parameter_updates'] == 0
            and seal['alltruth_accessed'] is False, 'Probes sealed before any card')
    for key in ('payloads', 'cotangents', 'direction', 'direction_scales', 'frozen_cstar_sha256'):
        require(seal[key] == record[key], 'Probe seal/result mismatch: ' + key)
    require(record['frozen_cstar_sha256'] == CSTAR_SHA and set(record['payloads']) == set(CASES),
            'Exact cstar and five fixed payloads')
    for case in CASES:
        item = record['payloads'][case]
        path = CPU / ('payload_' + case + '.npz')
        require(Path(item['path']).resolve() == path and item['sha256'] == PAYLOAD_SHA[case],
                'Canonical fixed payload: ' + case)
        require(record['artifacts'][str(path)] == item['sha256']
                and path.stat().st_size == item['bytes'], 'Payload file binding: ' + case)
        pin(path, item['sha256'])
    require(set(record['cotangents']) == {'35', '100'}, 'Fixed two cotangents')
    # Bind these files without decoding or using them to alter the payloads.
    for rank in (35, 100):
        item = record['cotangents'][str(rank)]
        path = CPU / f'cotangent_{rank:04d}.npy'
        require(Path(item['path']).resolve() == path
                and record['artifacts'][str(path)] == item['sha256'], 'Canonical cotangent')
        pin(path, item['sha256'])
    direction = record['direction']
    path = CPU / 'direction64.npz'
    require(Path(direction['path']).resolve() == path
            and record['artifacts'][str(path)] == direction['sha256'], 'Original direction artifact')
    pin(path, direction['sha256'])
    return record, pins


def assignment_span(source, name):
    """AST UTF8 byte offsets, not reformatting of unrelated source."""
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(nodes) == 1 and len(nodes[0].targets) == 1, 'Unique literal assignment: ' + name)
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    stop = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    return start, stop, ast.literal_eval(node.value)


def array_metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--cpu-forward-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 0
    require(not OUT.exists(), 'Exclusive build_probes_v1; no overwrite or automatic retry')
    record, pins = admission(args.source_sha256, args.cpu_forward_sha256)
    require(not any(name in sys.modules for name in ('torch', 'flopscope', 'whestbench')),
            'Standalone source/payload builder, not a numerical/card process')
    import numpy as np
    with np.load(ORIGINAL, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    parent = PARENT.read_bytes()
    start, stop, literal = assignment_span(parent, '_GNN_LITERAL')
    bd_start, bd_stop, cstar = assignment_span(parent, '_BD_COEFFICIENT_LITERAL')
    cstar_array = np.asarray(cstar, dtype=np.float32)
    require(cstar_array.shape == (15, 4) and np.isfinite(cstar_array).all()
            and hashlib.sha256(cstar_array.tobytes()).hexdigest() == CSTAR_SHA,
            'Original exported FP32 cstar, never recomputed')
    require(set(original) == set(literal) == set(PKEYS + FROZEN_KEYS), 'Original payload roster')
    require(sum(original[key].size for key in PKEYS) == 45200
            and sum(original[key][0].size for key in INACTIVE_KEYS) == 66,
            '45200 allocated / 45134 active P coordinates')
    for key, value in original.items():
        decoded = np.asarray(literal[key], dtype=np.float32)
        require(value.dtype == np.float32 and np.isfinite(value).all()
                and decoded.shape == value.shape and decoded.tobytes() == value.tobytes(),
                'Parent literal/original ARRAY byte identity: ' + key)
    prepared = {}
    for case in CASES:
        item = record['payloads'][case]
        with np.load(item['path'], allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        require(set(values) == set(original) == set(item['arrays']), 'Probe payload roster: ' + case)
        for key, old in original.items():
            value = values[key]
            require(value.dtype == old.dtype and value.shape == old.shape
                    and np.isfinite(value).all() and array_metadata(value) == item['arrays'][key],
                    'Sealed probe ARRAY metadata: ' + case + '/' + key)
            if key in FROZEN_KEYS or case == 'null':
                require(value.tobytes() == old.tobytes(), 'Frozen/null array changed: ' + key)
        for key in INACTIVE_KEYS:
            require(values[key][0].tobytes() == original[key][0].tobytes(), 'Inactive row changed: ' + key)
        replacement = ('_GNN_LITERAL = ' + repr({key: values[key].tolist() for key in literal})).encode()
        candidate = parent if case == 'null' else parent[:start] + replacement + parent[stop:]
        new_start, new_stop, replay = assignment_span(candidate, '_GNN_LITERAL')
        require(new_start == start and candidate[:new_start] + parent[start:stop] + candidate[new_stop:] == parent,
                'Literal-only exact inverse: ' + case)
        for key, value in values.items():
            restored = np.asarray(replay[key], dtype=value.dtype)
            require(restored.shape == value.shape and restored.tobytes() == value.tobytes(),
                    'Literal round-trip ARRAY bytes: ' + case + '/' + key)
        a, b, _ = assignment_span(candidate, '_BD_COEFFICIENT_LITERAL')
        require(candidate[a:b] == parent[bd_start:bd_stop], 'cstar source bytes changed')
        prepared[case] = (candidate, item)
    unchanged(pins)
    OUT.mkdir(exist_ok=False)
    candidates, artifacts = {}, {}
    common = dict(parent_sha256=PARENT_SHA, cpu_forward_result_sha256=CPU_SHA,
                  cpu_forward_completion_sha256=CPU_COMPLETION_SHA, probe_seal_sha256=PROBE_SEAL_SHA,
                  source_bindings=pins, frozen_cstar_sha256=CSTAR_SHA,
                  original_literal_matches_original_payload_arrays=True,
                  null_payload_file_sha256_equals_original=True,
                  non_PKEYS_array_bytes_preserved=True, unused_layer0_array_bytes_preserved=True,
                  cstar_source_byte_exact=True, allocated_parameters=45200, active_parameters=45134,
                  no_fit=True, parameter_updates=0, targets_accessed=False, weights_accessed=False,
                  estimator_imported=False, card_calls=0, complete_cost_measured=False,
                  native_execution_admitted=False, training_admitted=False)
    for case in CASES:
        candidate, item = prepared[case]
        folder = OUT / case
        folder.mkdir(exist_ok=False)
        source_path, build_path = folder / 'estimator.py', folder / 'build.json'
        with source_path.open('xb') as stream:
            stream.write(candidate)
        digest = hashlib.sha256(candidate).hexdigest()
        require(sha(source_path) == digest, 'Saved candidate source changed')
        build = dict(common, schema='current_d50_native_p_probe_build_v1', passed=True, completed=True,
                     case=case, amplitude=AMPLITUDES[case], payload=item, candidate_sha256=digest,
                     modified_assignment_only=None if case == 'null' else '_GNN_LITERAL',
                     null_source_byte_exact=case == 'null' and candidate == parent,
                     inverse_source_byte_exact=True, all_other_parent_source_byte_exact=True,
                     payload_literal_array_byte_exact=True,
                     created_utc=datetime.now(timezone.utc).isoformat())
        write(build_path, build)
        candidates[case] = dict(path=str(source_path), sha256=digest, build_path=str(build_path),
                                build_sha256=sha(build_path), payload_path=item['path'],
                                payload_sha256=item['sha256'], amplitude=AMPLITUDES[case])
        artifacts[str(source_path)], artifacts[str(build_path)] = digest, sha(build_path)
    require(candidates['null']['sha256'] == PARENT_SHA, 'Saved null is not exact d50')
    unchanged(pins)
    unchanged(artifacts)
    aggregate = dict(common, schema='current_d50_native_p_probe_builds_v1', passed=True, completed=True,
                     cases=list(CASES), candidates=candidates, artifacts=artifacts,
                     all_five_sources_complete=True, all_source_inverses_byte_exact=True,
                     null_source_byte_exact=True, payloads_regenerated=False,
                     hybrid_qualification_verified=False, numpy_version=np.__version__,
                     created_utc=datetime.now(timezone.utc).isoformat())
    # This final receipt is the only aggregate completion marker.
    write(OUT / 'build.json', aggregate)
    print(json.dumps(dict(build_path=str(OUT / 'build.json'), build_sha256=sha(OUT / 'build.json'),
                          candidate_sha256={key: value['sha256'] for key, value in candidates.items()},
                          native_execution_admitted=False, training_admitted=False), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
