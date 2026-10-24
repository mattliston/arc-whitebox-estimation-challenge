"""One shared first-affine fit; fixed per-layer train normalization, no val tuning."""
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import fit_frontier_v1 as f

FIELDS = ('enc0_w', 'enc0_b')
STEPS = 3000
RATE = 1e-4
ANCHOR = 1e-4


def activation(z):
    sigmoid = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    return z * sigmoid, sigmoid + z * sigmoid * (1 - sigmoid)


def forward(data, parameters):
    h1, d1 = activation(data['x'] @ parameters['enc0_w'].T + data['extra_bias'] + parameters['enc0_b'])
    h2, d2 = activation(h1 @ parameters['enc2_w'].T + parameters['enc2_b'])
    return h2 @ parameters['enc4_w'].T + parameters['enc4_b'], (d1, d2)


def metrics(data, parameters):
    y, _ = forward(data, parameters)
    error = y - data['target']
    delta = (np.sum(y * data['head'], axis=1) + data['head_b']) * data['scale'] - data['target_delta']
    message = error @ data['msg_w'].T
    return dict(state_mse=float(np.mean(error.astype('f8') ** 2)),
                head_delta_mse=float(np.mean(delta.astype('f8') ** 2)),
                message_mse=float(np.mean(message.astype('f8') ** 2)))


def loss_gradient(data, parameters, initial, anchor_scales):
    y, (d1, d2) = forward(data, parameters)
    error = y - data['target']
    delta = (np.sum(y * data['head'], axis=1) + data['head_b']) * data['scale'] - data['target_delta']
    message = error @ data['msg_w'].T
    n, state_width = error.shape
    ws, wh, wm = data['weights'].T
    loss = (np.mean(error.astype('f8') ** 2 * ws[:, None])
            + np.mean(delta.astype('f8') ** 2 * wh)
            + np.mean(message.astype('f8') ** 2 * wm[:, None])) / 3
    dy = (error * ws[:, None] * (2 / (3 * n * state_width))
          + (delta * data['scale'] * wh)[:, None] * data['head'] * (2 / (3 * n))
          + (message @ data['msg_w']) * wm[:, None] * (2 / (3 * n * data['msg_w'].shape[0])))
    dz1 = (((dy @ parameters['enc4_w']) * d2) @ parameters['enc2_w']) * d1
    gradients = dict(enc0_w=dz1.T @ data['x'], enc0_b=dz1.sum(axis=0))
    for key in FIELDS:
        difference = parameters[key] - initial[key]
        loss += ANCHOR * np.mean(difference.astype('f8') ** 2) / anchor_scales[key]
        gradients[key] += (2 * ANCHOR / (parameters[key].size * anchor_scales[key])) * difference
    return float(loss), gradients


def row_data(data):
    result = dict(data)
    n = len(data['x'])
    result['head'] = np.broadcast_to(data['head'], (n, len(data['head']))).copy()
    result['head_b'] = np.broadcast_to(data['head_b'], (n,)).copy()
    return result


def worker(connection, data, initial, scales):
    parameters = {key: value.copy() for key, value in initial.items()}
    while True:
        values = connection.recv()
        if values is None:
            break
        parameters['enc0_w'], parameters['enc0_b'] = values
        connection.send(loss_gradient(data, parameters, initial, scales))
    connection.close()


def rss(pid):
    try:
        for line in Path('/proc/%d/status' % pid).read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def fit_shared(layers, initial, out):
    started = time.monotonic()
    parameters = {key: value.copy() for key, value in initial.items()}
    scales = {key: max(float(np.mean(parameters[key].astype('f8') ** 2)), 1e-4) for key in FIELDS}
    prepared, before = {}, {}
    for layer in range(1, 16):
        prepared[layer] = {split: row_data(layers[layer][split]) for split in ('train', 'valid')}
        before[layer] = {split: metrics(prepared[layer][split], parameters) for split in ('train', 'valid')}
        inv = np.asarray([1 / max(before[layer]['train'][key], 1e-24)
                          for key in ('state_mse', 'head_delta_mse', 'message_mse')], dtype=parameters['enc0_w'].dtype)
        for split in ('train', 'valid'):
            prepared[layer][split]['weights'] = np.broadcast_to(inv, (len(prepared[layer][split]['x']), 3)).copy()
    keys = ('x', 'extra_bias', 'target', 'target_delta', 'head', 'head_b', 'scale', 'weights')
    train = {key: np.concatenate([prepared[layer]['train'][key] for layer in range(1, 16)]) for key in keys}
    train['msg_w'] = initial_msg = prepared[1]['train']['msg_w']
    assert len(train['x']) == 4 * 128 * 15
    assert all(np.array_equal(prepared[layer]['train']['msg_w'], initial_msg) for layer in range(1, 16))
    n = len(train['x']); parts = (slice(0, n // 2), slice(n // 2, n))
    ctx = mp.get_context('fork')
    connections, processes, weights = [], [], []
    for part in parts:
        parent, child = ctx.Pipe()
        batch = {key: (value if key == 'msg_w' else value[part]) for key, value in train.items()}
        process = ctx.Process(target=worker, args=(child, batch, initial, scales))
        process.start(); child.close()
        connections.append(parent); processes.append(process); weights.append(len(batch['x']) / n)
    f.jwrite(out / 'RUNNING_v1.json', dict(utc=f.utc(), parent_pid=os.getpid(), worker_pids=[p.pid for p in processes],
        blas_threads=1, exact_fullbatch_partition=True, shared_parameter_count=sum(parameters[key].size for key in FIELDS),
        training_rows=n))
    moment = {key: np.zeros_like(parameters[key]) for key in FIELDS}
    variance = {key: np.zeros_like(parameters[key]) for key in FIELDS}
    trace, peak = [], 0
    try:
        for step in range(1, STEPS + 1):
            for connection in connections:
                connection.send(tuple(parameters[key] for key in FIELDS))
            pieces = []
            for connection in connections:
                assert connection.poll(120), 'bounded gradient-worker response timeout'
                pieces.append(connection.recv())
            loss = sum(weight * result[0] for weight, result in zip(weights, pieces))
            gradients = {key: sum(weight * result[1][key] for weight, result in zip(weights, pieces)) for key in FIELDS}
            for key in FIELDS:
                moment[key] = .9 * moment[key] + .1 * gradients[key]
                variance[key] = .999 * variance[key] + .001 * gradients[key] ** 2
                parameters[key] -= RATE * (moment[key] / (1 - .9 ** step)) / (np.sqrt(variance[key] / (1 - .999 ** step)) + 1e-8)
            if step == 1 or step % 250 == 0:
                assert np.isfinite(loss) and all(np.isfinite(value).all() for value in parameters.values())
                peak = max(peak, rss(os.getpid()) + sum(rss(p.pid) for p in processes))
                assert peak <= 4 * 1024**3
                point = dict(step=step, normalized_train_loss=loss, combined_rss_bytes=peak)
                trace.append(point); print(point, flush=True)
            if step >= 500 and loss <= .1:
                break
    finally:
        for connection in connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.terminate(); process.join()
        for connection in connections:
            connection.close()
    assert all(p.exitcode == 0 for p in processes)
    for key in set(parameters) - set(FIELDS):
        assert np.array_equal(parameters[key], initial[key]), key
    result = dict(steps=step, wall_s=time.monotonic() - started, final_normalized_train_loss=loss,
        trace=trace, peak_combined_fit_rss_bytes=peak, shared_parameters=f.encode({key: parameters[key] for key in FIELDS}),
        layers=[dict(layer=layer, train_before=before[layer]['train'], valid_before=before[layer]['valid'],
            train_after=metrics(prepared[layer]['train'], parameters),
            valid_after=metrics(prepared[layer]['valid'], parameters)) for layer in range(1, 16)])
    f.jwrite(out / 'SHARED_RESULT_v1.json', result)
    return result


def gradient_check():
    rng = np.random.default_rng(20261021)
    n, nx, h1, h2, state_width, messages = 9, 5, 4, 3, 2, 3
    initial = dict(enc0_w=rng.normal(0, .2, (h1, nx)), enc0_b=rng.normal(0, .1, h1),
        enc2_w=rng.normal(0, .2, (h2, h1)), enc2_b=rng.normal(0, .1, h2),
        enc4_w=rng.normal(0, .2, (state_width, h2)), enc4_b=rng.normal(0, .1, state_width))
    parameters = {key: value.copy() for key, value in initial.items()}
    parameters['enc0_w'] += .01
    data = dict(x=rng.normal(0, .3, (n, nx)), extra_bias=rng.normal(0, .05, (n, h1)),
        target=rng.normal(0, .2, (n, state_width)), target_delta=rng.normal(0, .2, n),
        head=rng.normal(0, .2, (n, state_width)), head_b=rng.normal(0, .05, n),
        scale=rng.uniform(.1, 1.5, n), msg_w=rng.normal(0, .2, (messages, state_width)),
        weights=rng.uniform(.2, 3, (n, 3)))
    scales = {key: max(float(np.mean(parameters[key] ** 2)), 1e-4) for key in FIELDS}
    loss, gradients = loss_gradient(data, parameters, initial, scales)
    checks = []
    for key in FIELDS:
        direction = rng.normal(size=parameters[key].shape); direction /= np.linalg.norm(direction)
        plus = {k: v.copy() for k, v in parameters.items()}; minus = {k: v.copy() for k, v in parameters.items()}
        plus[key] += 1e-6 * direction; minus[key] -= 1e-6 * direction
        numerical = (loss_gradient(data, plus, initial, scales)[0] - loss_gradient(data, minus, initial, scales)[0]) / 2e-6
        analytic = float(np.sum(gradients[key] * direction))
        error = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-9)
        assert error < 2e-6
        checks.append(dict(field=key, numerical=numerical, analytic=analytic, relative_error=error))
    pieces = []
    for part in (slice(0, 4), slice(4, n)):
        batch = {key: (value if key == 'msg_w' else value[part]) for key, value in data.items()}
        pieces.append((len(batch['x']) / n, loss_gradient(batch, parameters, initial, scales)))
    np.testing.assert_allclose(sum(w * item[0] for w, item in pieces), loss, atol=1e-15, rtol=1e-13)
    for key in FIELDS:
        np.testing.assert_allclose(sum(w * item[1][key] for w, item in pieces), gradients[key], atol=1e-15, rtol=1e-13)
    return dict(status='SHARED_PER_ROW_HEAD_WEIGHTED_GRADIENT_AND_PARTITION_PASS', checks=checks,
        unequal_partition_matches_full_batch=True)
