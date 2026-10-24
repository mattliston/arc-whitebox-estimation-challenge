"""Fold the single shared first-affine fit into bank8's existing literal only."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import fit_frontier_v1 as f

SCI = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL = 'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
FIT = '844af9916c47a65b1768b3b7357af7adbe8af060da579ad49d7ad36b15930187'
MATH = '998de4f3b9dd5ad8a01ef8dfc6ba931b00f30b9c7dff3de6c7865e8d94b585c7'


def main():
    parser = argparse.ArgumentParser()
    for key in ('fit', 'guard', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    report, guard = (json.loads(path.read_text()) for path in (args.fit, args.guard))
    assert report['status'] == 'FIXED_SHARED_BANK_CM8_ACTUAL_INPUT_FIRST_AFFINE_ORIGINAL16_TARGET_FIT_NO_NATIVE_SCORE'
    assert report['fit_source_sha256'] == FIT and report['math_source_sha256'] == MATH
    assert report['fixed_fidelity_gate_pass'] and all(value <= 1.05 for value in report['validation_fidelity_ratios'].values())
    assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit'] == 0
    assert guard['guard_sha256'] == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
    spec = report['source_spec']
    science, release = (Path(spec[role]['path']) for role in ('science', 'release'))
    assert f.sha(science) == spec['science']['sha256'] == SCI
    assert f.sha(release) == spec['release']['sha256'] == REL
    oldtail = Path(spec['tail']); assert f.sha(oldtail) == spec['tail_sha256']
    literals, functions = f.literals_and_functions(science.read_text())
    original = f.decode(literals['_GNN_LITERAL'])
    parameters = {key: value.copy() for key, value in original.items()}
    updates = f.decode(report['shared_parameters'])
    assert set(updates) == {'enc0_w', 'enc0_b'}
    assert updates['enc0_w'].shape == (128, 78) and updates['enc0_b'].shape == (128,)
    for key, value in updates.items():
        assert value.shape == parameters[key].shape and np.isfinite(value).all()
        parameters[key] = value
    assert all(np.array_equal(value, original[key]) for key, value in parameters.items() if key not in updates)
    assert '_VMFR_ORIGINAL_W3_ENCODER' in functions['_w3r_metric']
    assert "_GNN_PARAMS['enc0_w']" not in functions['_w3r_metric']
    tail = '\n# One shared first-affine teacher repair; bank8 operators and every other field unchanged.\n'
    tail += 'assert _GNN_PARAMS is None and not _E2_GNN_STATE and not _E2_MSG\n'
    tail += 'assert not _HC_CONST and not _HCF_CONST and _MESSAGE_EDGE_PARAMETERS is None\n'
    tail += 'assert _VMFR_CM == 8 and _O20_FLATEXTRA_ON is False\n'
    tail += 'assert _W3R_METRIC is None and _VMFR_W3_ENCODER_ARRAY is None\n'
    tail += '_GNN_LITERAL = ' + repr(f.encode(parameters)) + '\n_FAST_RESCUE_FRAC = {}\n'
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.parse(tail).body)
    args.out.mkdir(parents=True, exist_ok=False)
    name = 'msg8_actual_input_shared_firstaffine'
    append, fixture = (args.out / filename for filename in ('tail_' + name + '_v1.py', 'composed_tail_' + name + '_v1.py'))
    f.write(append, tail); f.write(fixture, oldtail.read_text() + tail)
    candidate = {key: value for key, value in spec.items() if key not in
        ('name', 'science', 'release', 'tail', 'tail_sha256', 'append_tail', 'append_tail_sha256', 'teacher_errors')}
    candidate.update(name=name, kind='shared_actual777a_inputs_original09fb_targets_first_affine_only',
        tail=str(fixture), tail_sha256=f.sha(fixture), append_tail=str(append), append_tail_sha256=f.sha(append),
        fit_sha256=f.sha(args.fit), fit_guard_sha256=f.sha(args.guard),
        changed_parameter_fields=['enc0_w', 'enc0_b'], original15_layer_weight_sharing_preserved=True,
        all128_nonlinear_units_preserved=True, no_runtime_function_changes=True,
        original_bank_message_projectors_preserved=True, frozen_W3_metric_preserved=True,
        cold_and_warm_arithmetic_shapes_unchanged=True,
        validation_fidelity_ratios=report['validation_fidelity_ratios'],
        qualification='Teacher-function fidelity gate passed; actual native32 score remains unknown.')
    for role, parent in [('science', science), ('release', release)]:
        target = args.out / ('candidate_' + name + ('_sci_v1.py' if role == 'science' else '_rel_v1.py'))
        source = parent.read_text() + tail; compile(source, str(target), 'exec'); f.write(target, source)
        candidate[role] = dict(path=str(target), sha256=f.sha(target), parent_path=str(parent), parent_sha256=f.sha(parent))
    manifest = dict(utc=f.utc(), status='IMMUTABLE_SHARED_FIRST_AFFINE_LITERAL_ONLY_NO_NATIVE_SCORE',
        candidates=[candidate], reference_submission=337422, fit_sha256=f.sha(args.fit), exporter_sha256=f.sha(__file__))
    f.jwrite(args.out / 'CANDIDATES_v1.json', manifest)
    f.jwrite(args.out / 'TINY_WITH_EXACT_PARENT_v1.json', dict(candidates=[spec, candidate], reference_submission=337422))
    print(json.dumps(dict(manifest=str(args.out / 'CANDIDATES_v1.json'), science=candidate['science'], release=candidate['release']), indent=2))


if __name__ == '__main__':
    main()
