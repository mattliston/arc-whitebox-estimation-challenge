"""One fixed bank CM8 first-affine repair on actual777a inputs and aligned original16 targets.

Only enc0_w/b change. All128 nonlinear units, later affine layers, state64,
message8 projections, heads, physics and coefficients remain fixed. No truth.
"""
import fit_frontier_v1 as f
import fit_frontier_v1 as capture
import fit_current_msg8_messages_v2 as current_capture
import shared_firstaffine_math_v1 as shared
import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np

STEPS = 3000
RATE = 1e-4
ANCHOR = 1e-4
WORKERS = 2
FIELDS = ('enc0_w', 'enc0_b')
FORWARD_FIELDS = FIELDS + ('enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')
CM8 = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
TEACHER = '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded'
LAYERS = None
INITIAL = None


def activation(z):
    sigmoid = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    return z * sigmoid, sigmoid + z * sigmoid * (1 - sigmoid)


def forward(data, parameters):
    h1, d1 = activation(data['x'] @ parameters['enc0_w'].T + data['extra_bias'] + parameters['enc0_b'])
    h2, d2 = activation(h1 @ parameters['enc2_w'].T + parameters['enc2_b'])
    return h2 @ parameters['enc4_w'].T + parameters['enc4_b'], (h1, d1, h2, d2)


def metrics(data, parameters):
    y, _ = forward(data, parameters)
    error = y - data['target']
    delta = (y @ data['head'] + data['head_b']) * data['scale'] - data['target_delta']
    message = error @ data['msg_w'].T
    return dict(state_mse=float(np.mean(error.astype('f8') ** 2)),
                head_delta_mse=float(np.mean(delta.astype('f8') ** 2)),
                message_mse=float(np.mean(message.astype('f8') ** 2)))


def loss_gradient(data, parameters, initial, norms, anchor_scales):
    y, (_, d1, _, d2) = forward(data, parameters)
    error = y - data['target']
    delta = (y @ data['head'] + data['head_b']) * data['scale'] - data['target_delta']
    message = error @ data['msg_w'].T
    n, state_width = error.shape
    loss = (np.mean(error.astype('f8') ** 2) / norms[0]
            + np.mean(delta.astype('f8') ** 2) / norms[1]
            + np.mean(message.astype('f8') ** 2) / norms[2]) / 3
    dy = (error * (2 / (3 * n * state_width * norms[0]))
          + ((delta * data['scale'])[:, None] * data['head'][None, :]) * (2 / (3 * n * norms[1]))
          + (message @ data['msg_w']) * (2 / (3 * n * data['msg_w'].shape[0] * norms[2])))
    dz1 = (((dy @ parameters['enc4_w']) * d2) @ parameters['enc2_w']) * d1
    gradients = dict(enc0_w=dz1.T @ data['x'], enc0_b=dz1.sum(axis=0))
    for key in FIELDS:
        difference = parameters[key] - initial[key]
        loss += ANCHOR * np.mean(difference.astype('f8') ** 2) / anchor_scales[key]
        gradients[key] += (2 * ANCHOR / (parameters[key].size * anchor_scales[key])) * difference
    return float(loss), gradients


def fit_layer(layer):
    started = time.monotonic()
    data, valid = LAYERS[layer]['train'], LAYERS[layer]['valid']
    parameters = {key: value.copy() for key, value in INITIAL.items()}
    initial = {key: value.copy() for key, value in parameters.items()}
    moment = {key: np.zeros_like(parameters[key]) for key in FIELDS}
    variance = {key: np.zeros_like(parameters[key]) for key in FIELDS}
    before, valid_before = metrics(data, parameters), metrics(valid, parameters)
    norms = np.asarray([max(before[key], 1e-24) for key in ('state_mse', 'head_delta_mse', 'message_mse')])
    scales = {key: max(float(np.mean(parameters[key].astype('f8') ** 2)), 1e-4) for key in FIELDS}
    trace = []
    for step in range(1, STEPS + 1):
        loss, gradients = loss_gradient(data, parameters, initial, norms, scales)
        for key in FIELDS:
            moment[key] = .9 * moment[key] + .1 * gradients[key]
            variance[key] = .999 * variance[key] + .001 * gradients[key] ** 2
            parameters[key] -= RATE * (moment[key] / (1 - .9 ** step)) / (np.sqrt(variance[key] / (1 - .999 ** step)) + 1e-8)
        if step == 1 or step % 250 == 0:
            assert np.isfinite(loss) and all(np.isfinite(value).all() for value in parameters.values())
            trace.append(dict(step=step, normalized_train_loss=loss))
        if step >= 500 and loss <= .1:
            break
    for key in set(FORWARD_FIELDS) - set(FIELDS):
        assert np.array_equal(parameters[key], initial[key]), key
    return dict(layer=layer, steps=step, wall_s=time.monotonic() - started,
                train_before=before, valid_before=valid_before,
                train_after=metrics(data, parameters), valid_after=metrics(valid, parameters),
                trace=trace, final_normalized_train_loss=loss,
                parameters=f.encode({key: parameters[key] for key in FIELDS}))


def rss(pid):
    try:
        for line in Path('/proc/%d/status' % pid).read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def main():
    global LAYERS, INITIAL
    parser = argparse.ArgumentParser()
    for key in ('captures', 'current-captures', 'input-manifest', 'original-fit-manifest', 'teacher421', 'gradient-proof', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert f.sha(shared.__file__) == '998de4f3b9dd5ad8a01ef8dfc6ba931b00f30b9c7dff3de6c7865e8d94b585c7'
    gradient = json.loads(args.gradient_proof.read_text())
    assert gradient['status'] == 'SHARED_PER_ROW_HEAD_WEIGHTED_GRADIENT_AND_PARTITION_PASS'
    assert gradient['math_source_sha256'] == f.sha(shared.__file__)
    assert f.sha(f.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(capture.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(current_capture.__file__) == '56d3c63e1d766e6b1870e6aa07b88b84c02172d71edf28a6ad6a1724ad3f53ea'
    assert f.sha(args.original_fit_manifest) == '36911ea677f05152193e3f0a5cd55ac7d092c794967aede501dcae8fd4aef574'
    assert f.sha(args.teacher421) == '9af5617cf1a8be8e8a6d0734c5779d277a0febc1f0c6a73d777bd204d2318e44'
    original_manifest = json.loads(args.original_fit_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    f.jwrite(args.out / 'PREREG_v1.json', dict(utc=f.utc(), source_sha256=f.sha(__file__),
        source_candidate='msg8_teacher_loaderfix', source_sha256_cm8=CM8,
        teacher_source_sha256=TEACHER, train=f.TRAIN, valid=f.VALID,
        steps=STEPS, rate=RATE, anchor=ANCHOR, workers=WORKERS, blas_threads=1,
        parameter_fields=FIELDS, fixed128_nonlinear_units=True, current_message_width=8,
        shared_across_all_15_layers=True, training_rows=7680, original_parameter_sharing_retained=True,
        per_layer_fixed_train_normalization=True, fidelity_gate_max_validation_ratio_each=1.05,
        shared_math_sha256=f.sha(shared.__file__), gradient_proof_sha256=f.sha(args.gradient_proof),
        objective='Equal initial-train-error-normalized native state64, physical native head delta and retained message-source MSE',
        train_stop=dict(min_steps=500, normalized_loss_at_most=.1),
        no_validation_selection=True, no_network_truth=True, reference_submission=337422,
        input_policy='Actual777a H78/bias/sigma with unchanged bank projectors; targets are original09fb native state/head at identical rank/layer/unit rows. Propagated fitted states still require native qualification',
        original_fit_manifest_sha256=f.sha(args.original_fit_manifest), teacher421_source_sha256=f.sha(args.teacher421), maximum_combined_fit_rss_bytes=4 * 1024**3))
    manifest = json.loads(args.input_manifest.read_text())
    spec = next(row for row in manifest['candidates'] if row['name'] == 'msg8_teacher_loaderfix')
    source = Path(spec['science']['path'])
    assert f.sha(source) == spec['science']['sha256'] == CM8
    literals, _ = f.literals_and_functions(source.read_text())
    student = f.decode(literals['_GNN_LITERAL'])
    assert student['enc0_w'].shape == (128, 78) and student['msg_w'].shape == (24, 64)
    rows, paths, metadata = capture.collect(args.captures)
    assert metadata['source_sha256'] == TEACHER
    assert metadata['hook_sha256'] == 'e5a982510b02c2ebaba2dbe98a46e39600c5f55bf0db68366dff17ea79f74612'
    assert metadata['gnn_literal'] == original_manifest['capture_metadata']['gnn_literal']
    assert {p['path']: p['sha256'] for p in paths} == {p['path']: p['sha256'] for p in original_manifest['capture_files']}
    complete = json.loads((args.captures / 'COMPLETE.json').read_text())
    assert complete['successful']
    for part in range(1, 4):
        receipt = json.loads((args.captures / ('part%02d_GUARD.json' % part)).read_text())
        assert receipt['successful'] and receipt['whole_group_retired'] and receipt['child_exit'] == 0
        assert receipt['guard_sha256'] == '942218a46cb224fed969288b8652a29223fa82623d15c18663ac0b5e640b39e5'
    teacher = f.decode(metadata['gnn_literal'])
    literals421, _ = f.literals_and_functions(args.teacher421.read_text())
    teacher421 = f.decode(literals421['_GNN_LITERAL'])
    assert teacher.keys() == teacher421.keys()
    assert all(np.array_equal(teacher[key], teacher421[key]) for key in teacher)
    assert np.array_equal(np.asarray(metadata['hc_u']), np.asarray(literals421['_HC_U']))
    assert np.array_equal(np.asarray(metadata['hcf_u']), np.asarray(literals421['_HCF_U']))
    for key in ('enc2_w', 'enc2_b', 'enc4_w', 'enc4_b', 'head', 'head_b', 'rms', 'gain'):
        assert np.array_equal(student[key], teacher[key]), key
    original_rows = rows
    target_by = {(row['rank'], row['layer']): row for row in original_rows}
    rows, current_paths, current_metadata = current_capture.collect(args.current_captures)
    assert current_metadata['source_sha256'] == CM8
    current_P = f.decode(current_metadata['gnn_literal'])
    assert all(np.array_equal(student[key], current_P[key]) for key in student)
    for row in rows:
        target_row = target_by[(row['rank'], row['layer'])]
        assert row['rows'] == target_row['rows'] == list(range(row['rank'] % 8, 1024, 8))
        assert row['phase'] == target_row['phase'] == 'scored'
    columns = np.arange(78)
    LAYERS = {layer: {} for layer in range(1, 16)}
    replay = {}
    for label, ranks in [('train', f.TRAIN), ('valid', f.VALID)]:
        selected = [row for row in rows if row['rank'] in ranks]
        data = current_capture.matrices(selected, student)
        target_selected = [target_by[(row['rank'], row['layer'])] for row in selected]
        target_data = capture.matrices(target_selected, teacher)
        labels = np.concatenate([np.full(128, row['layer'], dtype=int) for row in selected])
        sigma = np.concatenate([f.array(row, 'sigma') for row in selected])
        delta = np.concatenate([f.array(row, 'delta') for row in target_selected])
        replay[label] = {'current_' + key: value for key, value in data.items() if key.startswith('replay')}
        replay[label].update({'target_' + key: value for key, value in target_data.items() if key.startswith('replay')})
        head_replay = np.concatenate([(f.array(row, 'sigma') * teacher['rms'][row['layer']]) *
            (f.array(row, 'state') @ teacher['head'][row['layer']] + teacher['head_b'][row['layer']]) *
            (1 + .25 * np.tanh(teacher['gain'][row['layer']:row['layer'] + 1])) for row in target_selected])
        np.testing.assert_allclose(head_replay, delta, atol=2e-8, rtol=1e-4)
        replay[label]['native_head_delta_max_abs'] = float(np.max(np.abs(head_replay - delta)))
        for layer in range(1, 16):
            mask = labels == layer
            LAYERS[layer][label] = dict(x=data['H'][mask][:, columns].copy(),
                extra_bias=(data['bias'][mask] - student['enc0_b']).copy(),
                target=target_data['state'][mask].copy(), target_delta=delta[mask].copy(),
                head=student['head'][layer].copy(), head_b=student['head_b'][layer].copy(),
                msg_w=student['msg_w'].copy(),
                scale=(sigma[mask] * student['rms'][layer] * (1 + .25 * np.tanh(student['gain'][layer]))).copy())
    INITIAL = {key: student[key].copy() for key in FORWARD_FIELDS}
    del rows, original_rows, target_by, target_selected, target_data, selected, data, labels, sigma, delta
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    result = shared.fit_shared(LAYERS, INITIAL, args.out)
    ratios = {key: sum(row['valid_after'][key] for row in result['layers']) /
              sum(row['valid_before'][key] for row in result['layers'])
              for key in ('state_mse', 'head_delta_mse', 'message_mse')}
    f.jwrite(args.out / 'FIT_COMPLETE_v1.json', dict(utc=f.utc(),
        status='FIXED_SHARED_BANK_CM8_ACTUAL_INPUT_FIRST_AFFINE_ORIGINAL16_TARGET_FIT_NO_NATIVE_SCORE',
        source_spec=spec, layers=result['layers'], shared_parameters=result['shared_parameters'],
        steps=result['steps'], trace=result['trace'], final_normalized_train_loss=result['final_normalized_train_loss'],
        peak_combined_fit_rss_bytes=result['peak_combined_fit_rss_bytes'],
        source_manifest_sha256=f.sha(args.input_manifest), fit_source_sha256=f.sha(__file__),
        math_source_sha256=f.sha(shared.__file__), gradient_proof_sha256=f.sha(args.gradient_proof),
        capture_files=paths, current_capture_files=current_paths, exact_rank_layer_unit_row_join=True,
        replay=replay, reference_submission=337422, validation_fidelity_ratios=ratios,
        fixed_fidelity_gate_pass=all(value <= 1.05 for value in ratios.values()),
        export_and_native_checks_pending=True))


if __name__ == '__main__':
    main()
