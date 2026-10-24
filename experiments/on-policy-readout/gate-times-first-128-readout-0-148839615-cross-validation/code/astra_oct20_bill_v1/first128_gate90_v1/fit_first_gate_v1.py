"""One fixed first-hidden × captured transport-gate readout, TRAIN90 only.

Five independent TRAIN72 first128 parents; no encoder/state update or new capture.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, gzip, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
OUT = RUNS/'first128_gate90_fit_v1'
PREREG = 'bfb1f9a1de49665b5427e98a5d83e3d13fac151741f450e73403e98f58ee7438'
N = 1024
CHANGED = ('head', 'head_b', 'hidden_head', 'first_hidden_head', 'gate_first_head')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def module(name, path, pin):
    assert sha(path) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ef = module('firstgate_frozen_custody_and_baseline', D/'enc2_affine_local90_v1/fit_enc2_affine_v1.py',
            '4c1c540117d808d42e53be9a45c29189b23fe81e41a06983130c00fd89820341')
cf = ef.cf
assert cf.RIDGE == .1 and cf.DAMPING == .5


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def guardcheck(path, threads):
    pin = ef.guardcheck(path, threads)
    guard = read(path)
    assert all(value == 0 for value in guard['cgroup_events_final'].values())
    return pin


def gate_features(first, gate):
    assert first.dtype == gate.dtype == np.dtype('f4')
    assert first.shape[1] == 128 and gate.shape == (len(first),)
    answer = first*gate[:, None]
    assert answer.dtype == np.dtype('f4') and np.isfinite(answer).all()
    return answer


def successor(grams, crosses, nnet, original, U, baseline=None):
    assert grams.shape == (15, 449, 449) and crosses.shape == (15, 449)
    if baseline is None:
        _, baseline = cf.coefficients(grams[:, :321, :321], crosses[:, :321], nnet, original, U)
    parent = ef.parent_parameters(original, baseline)
    Q = ef.mapping(U)
    combined, extra = [], []
    for layer in range(15):
        gram, cross = grams[layer], crosses[layer]
        XX, XB, BB = gram[:321, :321], gram[:321, 321:], gram[321:, 321:]
        QQ, QB = Q.T@XX@Q, Q.T@XB
        n = nnet*N
        scale = np.maximum(np.sqrt(np.maximum(np.diag(QQ), 0)/n), 1e-8)
        A = np.linalg.solve(QQ/(scale[:, None]*scale[None, :])+1e-10*n*np.eye(257),
                            QB/scale[:, None])/scale[:, None]
        residual_gram = BB-A.T@QB-QB.T@A+A.T@QQ@A
        residual_gram = .5*(residual_gram+residual_gram.T)
        residual_cross = cross[321:]-XB.T@baseline[layer]-A.T@(Q.T@(cross[:321]-XX@baseline[layer]))
        beta = cf.solve(residual_gram, residual_cross, n)
        combined.append(baseline[layer]-Q@(A@beta))
        extra.append(beta)
    combined, extra = np.stack(combined), np.stack(extra)
    candidate = ef.parent_parameters(original, combined)
    candidate['gate_first_head'] = np.zeros((16, 128), 'f4')
    candidate['gate_first_head'][1:] = extra.astype('f4')
    assert set(candidate) == set(parent)|{'gate_first_head'}
    for key in parent:
        assert candidate[key].shape == parent[key].shape and candidate[key].dtype == np.dtype('f4')
        if key not in CHANGED:
            np.testing.assert_array_equal(candidate[key], parent[key])
    linear = np.column_stack([combined-baseline, extra])
    return parent, candidate, linear, baseline


def fit_excluding_block(grams, crosses, excluded, original, U):
    assert len(grams) == len(crosses) == 5 and 0 <= excluded < 5
    chosen = [index for index in range(5) if index != excluded]
    return successor(sum(grams[index] for index in chosen),
                     sum(crosses[index] for index in chosen), 72, original, U)


def model_digest(params):
    digest = hashlib.sha256()
    for key in sorted(params):
        value = params[key]
        digest.update(key.encode()+str(value.shape).encode()+str(value.dtype).encode()+value.tobytes())
    return digest.hexdigest()


def deployed(first, gate, sigma, layer, params):
    first_scalar = first@params['first_hidden_head'][layer]
    hidden = ef.silu(first@params['enc2_w'].T+params['enc2_b'])
    state = hidden@params['enc4_w'].T+params['enc4_b']
    inner = (state@params['head'][layer]+hidden@params['hidden_head'][layer])+first_scalar
    if 'gate_first_head' in params:
        inner = inner+gate_features(first, gate)@params['gate_first_head'][layer]
    inner = inner+params['head_b'][layer]
    gain = np.float32(1)+np.float32(.25)*np.tanh(params['gain'][layer:layer+1])
    delta = ((sigma.astype('f4')*params['rms'][layer])*inner)*gain
    message = state@params['msg_w'].T+params['msg_b']
    assert all(value.dtype == np.dtype('f4') and np.isfinite(value).all() for value in (hidden, state, delta, message))
    return dict(hidden=hidden, state=state, delta=delta, message=message)


def data_rows(seeds, records, original):
    """Same authentic source, local residual and mask as the frozen first128 baseline."""
    for seed in seeds:
        pin = records[seed]
        assert sha(pin['truth_path']) == pin['truth_sha256'] and sha(pin['prediction_path']) == pin['prediction_sha256']
        pred = np.fromfile(pin['prediction_path'], dtype='<f4').reshape(16, N).astype('f8')
        assert np.isfinite(pred).all()
        weights = np.load(pin['weights_path'], allow_pickle=False, mmap_mode='r')
        assert weights.shape == (16, N, N)
        with np.load(pin['truth_path'], allow_pickle=False) as truth:
            assert int(truth['seed']) == seed and str(truth['weights_sha256']) == pin['raw_weights_sha256']
            residual_a = truth['mc_post_A'].astype('f8')-pred
            residual_b = truth['mc_post_B'].astype('f8')-pred
        for item in pin['layers']:
            layer = item['layer']
            assert sha(item['capture_path']) == item['capture_sha256'] and sha(item['first']['path']) == item['first']['sha256']
            first = np.fromfile(item['first']['path'], dtype='<f4').reshape(N, 128)
            with gzip.open(item['capture_path'], 'rt', encoding='utf8') as stream:
                cap = json.load(stream)
            assert cap['rank'] == seed and cap['layer'] == layer and cap['phase'] == 'scored'
            state, hidden, sigma, ng, saved = [cf.arr(cap, key) for key in ('state', 'hidden2', 'sigma', 'gate_feature', 'delta')]
            assert state.shape == (N, 64) and hidden.shape == (N, 128) and sigma.shape == ng.shape == saved.shape == (N,)
            gate = ng*float(original['scales'][layer, 5])+float(original['centers'][layer, 5])
            assert np.max(np.abs(ng)) < 7.999 and gate.min() > -2e-6 and gate.max() < 1+2e-6
            gate = np.clip(gate, 0, 1)
            physical = sigma*float(original['rms'][layer])*(1+.25*np.tanh(float(original['gain'][layer])))
            assert physical.min() > 1e-12
            replay = physical*(state@original['head'][layer].astype('f8')+float(original['head_b'][layer]))
            assert np.max(np.abs(replay-saved)) < 1e-8+2e-5*np.max(np.abs(saved))
            assert np.max(np.abs(state-(hidden@original['enc4_w'].astype('f8').T+original['enc4_b'].astype('f8')))) < 3e-5
            mask = (pred[layer] > 1e-12).astype('f8') if layer < 15 else np.ones(N)
            X = np.column_stack([state, np.ones(N), hidden, first.astype('f8')])*mask[:, None]
            B = gate_features(first, ng.astype('f4')).astype('f8')*mask[:, None]
            inherited = np.asarray(weights[layer], dtype='f8').T@np.stack([residual_a[layer-1], residual_b[layer-1]], axis=1)
            y_a = (residual_a[layer]-gate*inherited[:, 0])/physical
            y_b = (residual_b[layer]-gate*inherited[:, 1])/physical
            assert np.isfinite(y_a).all() and np.isfinite(y_b).all()
            yield dict(seed=seed, layer=layer, X=X, Z=np.column_stack([X, B]), first=first,
                       normalized_gate=ng.astype('f4'), sigma=sigma, physical=physical, mask=mask,
                       saved=saved, y_a=y_a, y_b=y_b)
        del weights


def trust_summary(rows):
    error = sum(row['head_error2'] for row in rows)
    energy = sum(row['head_energy'] for row in rows)
    assert energy > 0
    per_layer = [float(np.sqrt(sum(row['head_error2'] for row in rows if row['layer'] == layer)/
                              sum(row['head_energy'] for row in rows if row['layer'] == layer)))
                 for layer in range(1, 16)]
    rms = float(np.sqrt(error/energy))
    return dict(passed=bool(rms <= .25 and max(per_layer) <= 1), physical_head_rms=rms,
                max_layer_physical_rms=max(per_layer), per_layer_physical_rms=per_layer,
                state_message_byte_identical=True)


def evaluate(seeds, records, original, parent, candidate, baseline):
    losses, trust = {}, []
    for row in data_rows(seeds, records, original):
        seed, layer = row['seed'], row['layer']
        arguments = row['first'], row['normalized_gate'], row['sigma'], layer
        old, new = deployed(*arguments, parent), deployed(*arguments, candidate)
        for key in ('hidden', 'state', 'message'):
            assert old[key].tobytes() == new[key].tobytes(), key
        acc = losses.setdefault(seed, dict(seed=seed, baseline=0., candidate=0., legacy_first128_moment=0.))
        acc['baseline'] += ef.crosshalf_loss(row['y_a'], row['y_b'], old['delta'], row['saved'], row['physical'], row['mask'])
        acc['candidate'] += ef.crosshalf_loss(row['y_a'], row['y_b'], new['delta'], row['saved'], row['physical'], row['mask'])
        correction = row['X']@baseline[layer-1]
        acc['legacy_first128_moment'] += float((row['y_a']-correction)@(row['y_b']-correction))
        difference = row['mask']*(new['delta'].astype('f8')-old['delta'].astype('f8'))
        trust.append(dict(seed=seed, layer=layer, head_error2=float(difference@difference),
                          head_energy=float(np.sum((row['mask']*old['delta'].astype('f8'))**2))))
    rows = []
    for seed in seeds:
        row = losses[seed]
        assert min(row['baseline'], row['candidate'], row['legacy_first128_moment']) > 0
        row['incremental_pct'] = 100*(row['candidate']/row['baseline']-1)
        row['float32_baseline_vs_legacy_moment_pct'] = 100*(row['baseline']/row['legacy_first128_moment']-1)
        rows.append(row)
    return rows, trust_summary(trust)


def selftest():
    rng = np.random.default_rng(202610210854)
    original = dict(enc2_w=(rng.normal(size=(128, 128))*.05).astype('f4'), enc2_b=np.zeros(128, 'f4'),
                    enc4_w=(rng.normal(size=(64, 128))*.05).astype('f4'), enc4_b=np.zeros(64, 'f4'),
                    head=(rng.normal(size=(16, 64))*.01).astype('f4'), head_b=np.zeros(16, 'f4'),
                    msg_w=(rng.normal(size=(8, 64))*.05).astype('f4'), msg_b=np.zeros(8, 'f4'),
                    gain=np.zeros(16, 'f4'), rms=np.ones(16, 'f4'))
    first = rng.normal(size=(703, 128)).astype('f4')
    gate = rng.uniform(-7.99, 7.99, 703).astype('f4')
    B32 = gate_features(first, gate)
    independent = np.asarray([[np.float32(first[i, j]*gate[i]) for j in range(128)] for i in range(703)], 'f4')
    assert B32.tobytes() == independent.tobytes()
    perturbed = first+np.float32(2e-6)
    error = np.max(np.abs(gate_features(perturbed, gate).astype('f8')-B32.astype('f8')))
    assert error <= 8*np.max(np.abs(perturbed.astype('f8')-first.astype('f8')))+5e-6
    hidden = ef.silu(first@original['enc2_w'].T+original['enc2_b'])
    state = hidden@original['enc4_w'].T+original['enc4_b']
    mask = (np.arange(703)%7 != 0).astype('f8')
    X = np.column_stack([state, np.ones(703), hidden, first])*mask[:, None]
    B = B32.astype('f8')*mask[:, None]
    Z = np.column_stack([X, B])
    assert np.all(Z[mask == 0] == 0)
    U = cf.nullspace(original['enc4_w'])
    Q = ef.mapping(U)
    np.testing.assert_allclose(X@Q, np.column_stack([state, np.ones(703), hidden@U, first])*mask[:, None], rtol=1e-6, atol=1e-7)
    target = rng.normal(size=703)
    G, b = Z.T@Z, Z.T@target
    grams = np.broadcast_to(G, (15, 449, 449)).copy()
    crosses = np.broadcast_to(b, (15, 449)).copy()
    parent, candidate, linear, baseline = successor(grams, crosses, 90, original, U)
    _, expected_baseline = cf.coefficients(grams[:, :321, :321], crosses[:, :321], 90, original, U)
    np.testing.assert_array_equal(baseline, expected_baseline)
    for key, value in ef.parent_parameters(original, expected_baseline).items():
        np.testing.assert_array_equal(value, parent[key])
    QQ, QB = (X@Q).T@(X@Q), (X@Q).T@B
    n = 90*N
    scale = np.maximum(np.sqrt(np.maximum(np.diag(QQ), 0)/n), 1e-8)
    A = np.linalg.solve(QQ/(scale[:, None]*scale[None, :])+1e-10*n*np.eye(257), QB/scale[:, None])/scale[:, None]
    R = B-(X@Q)@A
    beta = cf.solve(R.T@R, R.T@(target-X@baseline[0]), n)
    np.testing.assert_allclose(linear[0, 321:], beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(Z@linear[0], R@linear[0, 321:], rtol=1e-7, atol=2e-8)
    for key in set(parent)-set(CHANGED):
        np.testing.assert_array_equal(parent[key], candidate[key])
    block_g = [grams]*5
    block_b = [crosses]*5
    a = fit_excluding_block(block_g, block_b, 4, original, U)
    block_g[4], block_b[4] = grams+1e5, crosses+1e12
    bfit = fit_excluding_block(block_g, block_b, 4, original, U)
    assert model_digest(a[0]) == model_digest(bfit[0]) and model_digest(a[1]) == model_digest(bfit[1])
    np.testing.assert_array_equal(a[2], bfit[2])
    np.testing.assert_array_equal(a[3], bfit[3])
    sigma = rng.uniform(.2, 1.2, 703).astype('f4')
    old, new = deployed(first, gate, sigma, 3, parent), deployed(first, gate, sigma, 3, candidate)
    reference = ef.deployed(first, sigma, 3, parent)
    for key in old:
        assert old[key].tobytes() == reference[key].tobytes()
    for key in ('hidden', 'state', 'message'):
        assert old[key].tobytes() == new[key].tobytes()
    zero = {key: value.copy() for key, value in parent.items()}
    zero['gate_first_head'] = np.zeros((16, 128), 'f4')
    for key, value in deployed(first, gate, sigma, 3, zero).items():
        assert value.tobytes() == old[key].tobytes()
    ya, yb, physical, saved = rng.normal(size=(4, 703))
    physical = np.abs(physical)+.2
    correction = mask*(new['delta'].astype('f8')-saved)/physical
    expected = float((ya-correction)@(yb-correction))
    assert expected == ef.crosshalf_loss(ya, yb, new['delta'], saved, physical, mask)
    return dict(float32_gate_product_all128=True, reconstruction_product_error_bound=True,
                q257_x321_raw449_mapping=True, clipping_mask_all_features=True,
                dense_residualized_head_solve=True, projection_compensation_identity=True,
                independent_first128_baseline_replay=True, excluded_features_and_labels_invariance=True,
                original_parent_float32_expression=True, own_state_message_byte_identity=True,
                zero_extra_head_byte_identity=True, physical_crosshalf_metric_identity=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = selftest()
    if args.self_test:
        print(json.dumps(checks), flush=True)
        return
    assert sys.platform == 'linux' and sha(HERE/'PREREG_v1.json') == PREREG
    prereg, review = read(HERE/'PREREG_v1.json'), read(HERE/'FIT_REVIEW_v1.json')
    assert review['successful'] and review['one_fixed_fit_authorized']
    assert review['prereg_sha256'] == PREREG and review['fitter_sha256'] == sha(__file__)
    pricepath = RUNS/'first128_gate90_price_v1/PRICE_v1.json'
    price = read(pricepath)
    assert sha(pricepath) == review['price_sha256'] and price['passes_price_gate'] and price['no_capture_or_truth_arrays_read']
    assert guardcheck(RUNS/'first128_gate90_price_v1_GUARD_v1.json', 1) == review['price_guard_sha256']
    assert guardcheck(RUNS/'first128_gate90_selftest_v1_GUARD_v1.json', 1) == review['selftest_guard_sha256']
    original, deployed_parent, records, proof = ef.custody(prereg)
    train, folds = prereg['train'], prereg['folds']
    assert folds == [train[index::5] for index in range(5)]
    U = cf.nullspace(original['enc4_w'])
    OUT.mkdir(exist_ok=False)
    write(OUT/'PREREG_COPY_v1.json', prereg)
    grams, crosses = np.zeros((5, 15, 449, 449)), np.zeros((5, 15, 449))
    full_G, full_b = np.zeros((15, 321, 321)), np.zeros((15, 321))
    block = {seed: index for index, seeds in enumerate(folds) for seed in seeds}
    loaded = []
    for row in data_rows(train, records, original):
        seed, layer = row['seed'], row['layer']-1
        Z, X, target = row['Z'], row['X'], .5*(row['y_a']+row['y_b'])
        gram, cross = Z.T@Z, Z.T@target
        gram[:321, :321], cross[:321] = X.T@X, X.T@target
        grams[block[seed], layer] += gram
        crosses[block[seed], layer] += cross
        full_G[layer] += gram[:321, :321]
        full_b[layer] += cross[:321]
        if layer == 14:
            loaded.append(seed)
            print(json.dumps(dict(utc=utc(), moments_complete=seed, completed=len(loaded))), flush=True)
    assert loaded == train
    _, full_baseline = cf.coefficients(full_G, full_b, 90, original, U)
    expected = ef.parent_parameters(original, full_baseline)
    assert set(expected) == set(deployed_parent)
    for key in expected:
        np.testing.assert_array_equal(expected[key], deployed_parent[key], err_msg='full90 frozen parent '+key)
    write(OUT/'BASELINE_PARITY_v1.json', dict(utc=utc(), successful=True, exact_all_deployed_tensors=True,
          parent_sha256=prereg['baseline_science_sha256'], parent_digest=model_digest(expected)))
    validation, reports = [], []
    for index, held in enumerate(folds):
        training = [seed for seed in train if seed not in held]
        parent, candidate, _, baseline = fit_excluding_block(grams, crosses, index, original, U)
        seal = model_digest(candidate)
        write(OUT/('FOLD%02d_SEALED_v1.json'%index), dict(utc=utc(), training=training, validation=held,
              independent_parent_digest=model_digest(parent), candidate_digest=seal, excluded_block=index,
              fitted_block_indices=[j for j in range(5) if j != index], sealed_before_validation_replay=True))
        _, train_trust = evaluate(training, records, original, parent, candidate, baseline)
        rows, validation_trust = evaluate(held, records, original, parent, candidate, baseline)
        assert model_digest(candidate) == seal
        report = dict(fold=index, summary=ef.summarize(rows), training_trust=train_trust,
                      validation_trust_descriptive_only=validation_trust, rows=rows)
        write(OUT/('FOLD%02d_RESULT_v1.json'%index), report)
        reports.append(report)
        validation.extend(rows)
        print(json.dumps(dict(utc=utc(), fold=index, summary=report['summary'], train_trust=train_trust)), flush=True)
    validation.sort(key=lambda row: train.index(row['seed']))
    cv = ef.summarize(validation)
    passed = cv['mean_incremental_pct'] <= -.15 and cv['incrementally_lower'] >= 54 and sum(row['summary']['mean_incremental_pct'] < 0 for row in reports) >= 4 and all(row['training_trust']['passed'] for row in reports)
    artifact, full_trust = None, None
    if passed:
        parent, candidate, _, baseline = successor(sum(grams), sum(crosses), 90, original, U, baseline=full_baseline)
        _, full_trust = evaluate(train, records, original, parent, candidate, baseline)
        passed = bool(full_trust['passed'])
        if passed:
            artifact = OUT/'GATE_FIRST128_MODEL_v1.npz'
            np.savez(artifact, **{key: candidate[key] for key in CHANGED})
    result = dict(utc=utc(), status='FIRST_GATE_FIXED_PROXY_PASS_NATIVE_REQUIRED' if passed else 'CLOSED_FIRST_GATE_FIXED_GATES_FAIL',
                  passed=bool(passed), prereg_sha256=PREREG, fitter_sha256=sha(__file__), root_review_sha256=sha(HERE/'FIT_REVIEW_v1.json'),
                  price_sha256=sha(pricepath), selftests=checks, exact_full90_parent_tensor_parity=True,
                  cv=cv, folds=reports, validation=validation, full90_training_trust=full_trust,
                  model_sha256=None if artifact is None else sha(artifact), custody=proof,
                  no_new_capture_or_truth=True, no_protected_or_fresh_truth_fit=True, no_candidate_export=True,
                  no_native_or_adjusted_score_claim=True, no_recurrent_trajectory_claim=True, no_grid_or_retry=True)
    write(OUT/'FITTED_v1.json', result)
    print(json.dumps(dict(utc=result['utc'], status=result['status'], cv=cv, full90_training_trust=full_trust)), flush=True)


if __name__ == '__main__':
    main()
