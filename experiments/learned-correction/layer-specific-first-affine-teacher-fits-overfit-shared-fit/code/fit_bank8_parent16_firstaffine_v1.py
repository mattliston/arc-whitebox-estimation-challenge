"""One fixed bank CM8 first-affine repair using original16 teacher captures and restored original16 carriers.

Only enc0_w/b change. All128 nonlinear units, later affine layers, state64,
message8 projections, heads, physics and coefficients remain fixed. No truth.
"""
import fit_frontier_v1 as f
import fit_frontier_v1 as capture
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
CM8 = '6995e7dba58c1d637fae9de58852e5a9f371e7490c126df0e6eb5482f5e96e38'
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
    for key in ('captures', 'input-manifest', 'original-fit-manifest', 'teacher421', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert f.sha(f.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(capture.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(args.original_fit_manifest) == '36911ea677f05152193e3f0a5cd55ac7d092c794967aede501dcae8fd4aef574'
    assert f.sha(args.teacher421) == '9af5617cf1a8be8e8a6d0734c5779d277a0febc1f0c6a73d777bd204d2318e44'
    original_manifest = json.loads(args.original_fit_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    f.jwrite(args.out / 'PREREG_v1.json', dict(utc=f.utc(), source_sha256=f.sha(__file__),
        source_candidate='msg8_parent16_carriers', source_sha256_cm8=CM8,
        teacher_source_sha256=TEACHER, train=f.TRAIN, valid=f.VALID,
        steps=STEPS, rate=RATE, anchor=ANCHOR, workers=WORKERS, blas_threads=1,
        parameter_fields=FIELDS, fixed128_nonlinear_units=True, current_message_width=8,
        objective='Equal initial-train-error-normalized native state64, physical native head delta and retained message-source MSE',
        train_stop=dict(min_steps=500, normalized_loss_at_most=.1),
        no_validation_selection=True, no_network_truth=True, reference_submission=337422,
        input_policy='Original09fb native inputs selected to bank CM8 features; corrected native16 carriers make same-incoming-state features exact. Propagated candidate states still require native qualification',
        original_fit_manifest_sha256=f.sha(args.original_fit_manifest), teacher421_source_sha256=f.sha(args.teacher421), maximum_combined_fit_rss_bytes=4 * 1024**3))
    manifest = json.loads(args.input_manifest.read_text())
    spec = next(row for row in manifest['candidates'] if row['name'] == 'msg8_parent16_carriers')
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
    keep = [np.asarray(indices, dtype=int) for indices in spec['retained_by_power']]
    columns = np.concatenate([np.arange(30)] + [30 + 16 * group + keep[group % 3] for group in range(6)])
    LAYERS = {layer: {} for layer in range(1, 16)}
    replay = {}
    for label, ranks in [('train', f.TRAIN), ('valid', f.VALID)]:
        selected = [row for row in rows if row['rank'] in ranks]
        data = capture.matrices(selected, teacher)
        labels = np.concatenate([np.full(128, row['layer'], dtype=int) for row in selected])
        sigma = np.concatenate([f.array(row, 'sigma') for row in selected])
        delta = np.concatenate([f.array(row, 'delta') for row in selected])
        replay[label] = {key: value for key, value in data.items() if key.startswith('replay')}
        head_replay = np.concatenate([(f.array(row, 'sigma') * teacher['rms'][row['layer']]) *
            (f.array(row, 'state') @ teacher['head'][row['layer']] + teacher['head_b'][row['layer']]) *
            (1 + .25 * np.tanh(teacher['gain'][row['layer']:row['layer'] + 1])) for row in selected])
        np.testing.assert_allclose(head_replay, delta, atol=2e-8, rtol=1e-4)
        replay[label]['native_head_delta_max_abs'] = float(np.max(np.abs(head_replay - delta)))
        for layer in range(1, 16):
            mask = labels == layer
            LAYERS[layer][label] = dict(x=data['H'][mask][:, columns].copy(),
                extra_bias=(data['bias'][mask] - teacher['enc0_b']).copy(),
                target=data['state'][mask].copy(), target_delta=delta[mask].copy(),
                head=student['head'][layer].copy(), head_b=student['head_b'][layer].copy(),
                msg_w=student['msg_w'].copy(),
                scale=(sigma[mask] * student['rms'][layer] * (1 + .25 * np.tanh(student['gain'][layer]))).copy())
    INITIAL = {key: student[key].copy() for key in FORWARD_FIELDS}
    del rows, selected, data, labels, sigma, delta
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    pool = mp.get_context('fork').Pool(WORKERS)
    pids = [worker.pid for worker in pool._pool]
    f.jwrite(args.out / 'RUNNING_v1.json', dict(utc=f.utc(), parent_pid=os.getpid(), worker_pids=pids, blas_threads=1))
    pending = {layer: pool.apply_async(fit_layer, (layer,)) for layer in range(1, 16)}
    results, peak = [], 0
    try:
        while pending:
            total = rss(os.getpid()) + sum(rss(pid) for pid in pids)
            peak = max(peak, total)
            assert total <= 4 * 1024**3, ('combined fitter RSS exceeded fixed4GiB ceiling', total)
            for layer, job in list(pending.items()):
                if job.ready():
                    result = job.get()
                    f.jwrite(args.out / ('L%02d_RESULT_v1.json' % layer), result)
                    results.append(result)
                    del pending[layer]
                    print(json.dumps(dict(event='layer_done', layer=layer, steps=result['steps'],
                        train=result['train_after'], valid=result['valid_after'])), flush=True)
            if pending:
                time.sleep(1)
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    f.jwrite(args.out / 'FIT_COMPLETE_v1.json', dict(utc=f.utc(),
        status='FIXED_BANK_CM8_FIRST_AFFINE_ORIGINAL16_TEACHER_FIT_NO_NATIVE_SCORE', source_spec=spec,
        layers=sorted(results, key=lambda row: row['layer']), peak_combined_fit_rss_bytes=peak,
        source_manifest_sha256=f.sha(args.input_manifest), fit_source_sha256=f.sha(__file__),
        capture_files=paths, replay=replay, reference_submission=337422,
        export_and_native_checks_pending=True))


if __name__ == '__main__':
    main()
