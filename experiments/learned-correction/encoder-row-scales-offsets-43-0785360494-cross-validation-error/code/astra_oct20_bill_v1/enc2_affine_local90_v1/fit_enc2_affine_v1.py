"""One fixed shared enc2 affine local-innovation step; TRAIN90 only.

Independent TRAIN72 first128 parents, unchanged reconstruction, float32 replay.
No capture, native execution, source export, parameter sweep or automatic retry.
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
H1 = D/'breakthrough_v1/first_hidden_readout_v1'
SIB = D/'truth_response95_v1'
V = D.parent/'astra_oct20_vm_frontier_v1'
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
CAP = RUNS/'full_state_response_train90_v1'
OUT = RUNS/'enc2_affine_local90_fit_v1'
PREREG = '6394ce7b5295a8a41b3d73b20cf43d9db298873211c0315a4115828fdf0591eb'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
SCI = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
N = 1024
CHANGED = ('enc2_w', 'enc2_b', 'head', 'head_b', 'hidden_head', 'first_hidden_head')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path, pin):
    assert sha(path) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cf = module('frozen_enc2_first128_baseline', H1/'fit_first128_local90_v1.py',
            '8e252f559879be4d6c2bceb033bf9d84408ada1c418942bcf4d4d939d009c526')


def guardcheck(path, threads):
    guard = read(path)
    assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit'] == 0
    assert not guard['cleanup_errors'] and guard['guard_sha256'] == GUARD
    assert guard['max_owned_swap_bytes'] == 0 and guard['thread_provenance']['blas_threads'] == threads
    assert guard['cap_bytes'] == 6500000000 and guard['reserve_bytes'] == 19000000000 and guard['max_active'] == 36
    return sha(path)


def silu(value):
    assert value.dtype == np.dtype('f4')
    with np.errstate(over='ignore'):
        answer = value/(np.float32(1)+np.exp(-value))
    assert answer.dtype == np.dtype('f4') and np.isfinite(answer).all()
    return answer


def derivative(value):
    value = np.asarray(value, dtype='f8')
    exponential = np.exp(-np.abs(value))
    sigmoid = np.where(value >= 0, 1/(1+exponential), exponential/(1+exponential))
    return sigmoid*(1+value*(1-sigmoid))


def mapping(U):
    assert U.shape == (128, 64)
    Q = np.zeros((321, 257))
    Q[:65, :65] = np.eye(65)
    Q[65:193, 65:129] = U
    Q[193:321, 129:] = np.eye(128)
    return Q


def parent_parameters(original, coefficients):
    assert coefficients.shape == (15, 321)
    params = {key: value.copy() for key, value in original.items()}
    params['head'][1:] = (original['head'][1:].astype('f8')+coefficients[:, :64]).astype('f4')
    params['head_b'][1:] = (original['head_b'][1:].astype('f8')+coefficients[:, 64]).astype('f4')
    params['hidden_head'] = np.zeros((16, 128), 'f4')
    params['first_hidden_head'] = np.zeros((16, 128), 'f4')
    params['hidden_head'][1:] = coefficients[:, 65:193].astype('f4')
    params['first_hidden_head'][1:] = coefficients[:, 193:321].astype('f4')
    return params


def layer_projection(gram, cross, baseline, params, layer, Q, n):
    """Raw577 -> folded shared256 Jacobian without opening another data row."""
    assert gram.shape == (577, 577) and cross.shape == (577,) and baseline.shape == (321,)
    sensitivity = params['enc4_w'].astype('f8').T@params['head'][layer].astype('f8')+params['hidden_head'][layer].astype('f8')
    g = np.concatenate([sensitivity, sensitivity])
    XX, XT, TT = gram[:321, :321], gram[:321, 321:], gram[321:, 321:]
    QQ = Q.T@XX@Q
    QJ = (Q.T@XT)*g[None, :]
    scale = np.maximum(np.sqrt(np.maximum(np.diag(QQ), 0)/n), 1e-8)
    A = np.linalg.solve(QQ/(scale[:, None]*scale[None, :])+1e-10*n*np.eye(257), QJ/scale[:, None])/scale[:, None]
    JJ = TT*(g[:, None]*g[None, :])
    residual_gram = JJ-A.T@QJ-QJ.T@A+A.T@QQ@A
    residual_gram = .5*(residual_gram+residual_gram.T)
    residual_target = cross[:321]-XX@baseline
    target = (cross[321:]-XT.T@baseline)*g-A.T@(Q.T@residual_target)
    return residual_gram, target, A, g


def apply_update(parent, theta, parts, Q):
    assert theta.shape == (256,) and np.isfinite(theta).all() and len(parts) == 15
    candidate = {key: value.copy() for key, value in parent.items()}
    alpha, offset = theta[:128], theta[128:]
    candidate['enc2_w'] = ((1+alpha[:, None])*parent['enc2_w'].astype('f8')).astype('f4')
    candidate['enc2_b'] = ((1+alpha)*parent['enc2_b'].astype('f8')+offset).astype('f4')
    linear = []
    for i, (_, _, A, g) in enumerate(parts):
        compensation = -Q@(A@theta)
        layer = i+1
        candidate['head'][layer] = (parent['head'][layer].astype('f8')+compensation[:64]).astype('f4')
        candidate['head_b'][layer] = np.float32(float(parent['head_b'][layer])+compensation[64])
        candidate['hidden_head'][layer] = (parent['hidden_head'][layer].astype('f8')+compensation[65:193]).astype('f4')
        candidate['first_hidden_head'][layer] = (parent['first_hidden_head'][layer].astype('f8')+compensation[193:321]).astype('f4')
        linear.append(np.concatenate([compensation, theta*g]))
    assert set(parent) == set(candidate)
    for key in parent:
        assert parent[key].shape == candidate[key].shape and candidate[key].dtype == np.dtype('f4')
        assert np.isfinite(candidate[key]).all()
        if key not in CHANGED:
            np.testing.assert_array_equal(parent[key], candidate[key])
    return candidate, np.stack(linear)


def successor(grams, crosses, nnet, original, U, baseline=None):
    """Exactly one shared solve; no trial steps or metric-driven choices."""
    if baseline is None:
        _, baseline = cf.coefficients(grams[:, :321, :321], crosses[:, :321], nnet, original, U)
    parent = parent_parameters(original, baseline)
    Q = mapping(U)
    parts = [layer_projection(grams[i], crosses[i], baseline[i], parent, i+1, Q, nnet*N) for i in range(15)]
    theta = cf.solve(sum(part[0] for part in parts), sum(part[1] for part in parts), nnet*N*15)
    candidate, linear = apply_update(parent, theta, parts, Q)
    return parent, candidate, linear, baseline, theta


def fit_excluding_block(grams, crosses, excluded, original, U):
    assert len(grams) == len(crosses) == 5 and 0 <= excluded < 5
    chosen = [index for index in range(5) if index != excluded]
    # Never sum the excluded block, even temporarily.
    G = sum(grams[index] for index in chosen)
    b = sum(crosses[index] for index in chosen)
    return successor(G, b, 72, original, U)


def model_digest(params):
    digest = hashlib.sha256()
    for key in CHANGED:
        value = params[key]
        digest.update(key.encode()+str(value.shape).encode()+str(value.dtype).encode()+value.tobytes())
    return digest.hexdigest()


def deployed(first, sigma, layer, params):
    """Exact active paid58c float32 expression order after the first SiLU."""
    first_scalar = first@params['first_hidden_head'][layer]
    hidden = silu(first@params['enc2_w'].T+params['enc2_b'])
    state = hidden@params['enc4_w'].T+params['enc4_b']
    inner = (state@params['head'][layer]+hidden@params['hidden_head'][layer])+first_scalar
    inner = inner+params['head_b'][layer]
    gain = np.float32(1)+np.float32(.25)*np.tanh(params['gain'][layer:layer+1])
    delta = ((sigma.astype('f4')*params['rms'][layer])*inner)*gain
    message = state@params['msg_w'].T+params['msg_b']
    assert all(value.dtype == np.dtype('f4') and np.isfinite(value).all() for value in (hidden, state, delta, message))
    return dict(hidden=hidden, state=state, delta=delta, message=message)


def crosshalf_loss(target_a, target_b, delta, saved, physical, mask):
    correction = mask*(delta.astype('f8')-saved)/physical
    return float((target_a-correction)@(target_b-correction))


def trust_summary(rows):
    totals = {key: sum(row[key] for row in rows) for key in ('head_error2', 'head_energy', 'state_error2', 'state_energy',
                                                            'message_error2', 'message_energy', 'curvature2', 'linear2')}
    assert min(totals['head_energy'], totals['state_energy'], totals['message_energy']) > 0
    layer_ratios = []
    for layer in range(1, 16):
        selected = [row for row in rows if row['layer'] == layer]
        layer_ratios.append(float(np.sqrt(sum(row['head_error2'] for row in selected)/sum(row['head_energy'] for row in selected))))
    ratios = dict(physical_head_rms=float(np.sqrt(totals['head_error2']/totals['head_energy'])),
                  max_layer_physical_rms=max(layer_ratios), state_rms=float(np.sqrt(totals['state_error2']/totals['state_energy'])),
                  message_rms=float(np.sqrt(totals['message_error2']/totals['message_energy'])),
                  curvature_to_linear_rms=float(np.sqrt(totals['curvature2']/max(totals['linear2'], 1e-300))))
    passed = ratios['physical_head_rms'] <= .25 and ratios['max_layer_physical_rms'] <= 1 and ratios['state_rms'] <= .05 and ratios['message_rms'] <= .05 and ratios['curvature_to_linear_rms'] <= .25
    return dict(passed=bool(passed), **ratios, per_layer_physical_rms=layer_ratios)


def selftest():
    rng = np.random.default_rng(202610210805)
    z = rng.normal(size=(113, 128))*2
    alpha, offset = rng.normal(size=128), rng.normal(size=128)
    smooth = lambda x: x/(1+np.exp(-x))
    step = 1e-6
    direction = z*alpha+offset
    numerical = (smooth(z+step*direction)-smooth(z-step*direction))/(2*step)
    np.testing.assert_allclose(numerical, derivative(z)*direction, rtol=2e-7, atol=2e-9)
    enc4 = rng.normal(size=(64, 128))*.1
    U = cf.nullspace(enc4)
    Q = mapping(U)
    hidden, first = rng.normal(size=(703, 128)), rng.normal(size=(703, 128))
    state = hidden@enc4.T+rng.normal(size=64)
    X = np.column_stack([state, np.ones(703), hidden, first])
    mask = (np.arange(703)%7 != 0).astype('f8')
    X *= mask[:, None]
    T = rng.normal(size=(703, 256))*mask[:, None]
    Z = np.column_stack([X, T])
    np.testing.assert_allclose(X@Q, np.column_stack([X[:, :65], X[:, 65:193]@U, X[:, 193:321]]), atol=1e-12)
    params = dict(enc4_w=enc4.astype('f4'), head=rng.normal(size=(16, 64)).astype('f4'),
                  hidden_head=rng.normal(size=(16, 128)).astype('f4'))
    baseline = rng.normal(size=321)*.01
    labels = rng.normal(size=703)
    G, b = Z.T@Z, Z.T@labels
    GG, bb, A, g = layer_projection(G, b, baseline, params, 1, Q, len(X))
    direct_J = T*g
    direct_R = direct_J-X@Q@A
    np.testing.assert_allclose(GG, direct_R.T@direct_R, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(bb, direct_R.T@(labels-X@baseline), rtol=2e-8, atol=2e-8)
    theta = cf.solve(GG, bb, len(X))
    dense_theta = cf.solve(direct_R.T@direct_R, direct_R.T@(labels-X@baseline), len(X))
    np.testing.assert_allclose(theta, dense_theta, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(Z@np.concatenate([-Q@(A@theta), g*theta]), direct_R@theta, atol=1e-11)
    sensitivity = params['head'][1].astype('f8')@params['enc4_w'].astype('f8')+params['hidden_head'][1].astype('f8')
    np.testing.assert_allclose(g, np.concatenate([sensitivity, sensitivity]), rtol=1e-13, atol=1e-13)
    # The excluded block never participates in any aggregate passed to the solver.
    blocks = [(Z[ids].T@Z[ids], Z[ids].T@labels[ids]) for ids in (np.arange(0, 351), np.arange(351, 703))]
    before = layer_projection(blocks[0][0], blocks[0][1], baseline, params, 1, Q, 351)
    blocks[1] = (blocks[1][0]*1e6, blocks[1][1]+1e12)
    after = layer_projection(blocks[0][0], blocks[0][1], baseline, params, 1, Q, 351)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)
    head = params['head'][1]
    _, baseline_a = cf.one_layer(G[:321, :321], b[:321], len(X), head, np.float32(.1), U)
    _, baseline_b = cf.one_layer(X.T@X, X.T@labels, len(X), head, np.float32(.1), U)
    np.testing.assert_allclose(baseline_a, baseline_b, rtol=1e-7, atol=1e-9)
    residual_a, residual_b = labels-X@baseline_a, labels+.1-X@baseline_a
    correction = direct_R@theta
    direct_loss = (residual_a-correction)@(residual_b-correction)
    moment_loss = residual_a@residual_b-correction@(residual_a+residual_b)+correction@correction
    np.testing.assert_allclose(direct_loss, moment_loss, rtol=1e-12)
    # Apply the same production update constructor at zero and nonzero theta.
    pp = dict(enc2_w=(rng.normal(size=(128, 128))*.02).astype('f4'), enc2_b=np.zeros(128, 'f4'),
              enc4_w=params['enc4_w'], enc4_b=np.zeros(64, 'f4'), head=params['head'], hidden_head=params['hidden_head'],
              first_hidden_head=(rng.normal(size=(16, 128))*.02).astype('f4'), head_b=np.zeros(16, 'f4'),
              gain=np.zeros(16, 'f4'), rms=np.ones(16, 'f4'), msg_w=(rng.normal(size=(24, 64))*.01).astype('f4'),
              msg_b=np.zeros(24, 'f4'))
    parts = [(GG, bb, A, g)]*15
    zero, zero_linear = apply_update(pp, np.zeros(256), parts, Q)
    first32, sigma = first[:31].astype('f4'), np.linspace(.5, 1.5, 31, dtype='f4')
    old, zero_out = deployed(first32, sigma, 1, pp), deployed(first32, sigma, 1, zero)
    for key in pp:
        np.testing.assert_array_equal(pp[key], zero[key])
    for key in old:
        np.testing.assert_array_equal(old[key], zero_out[key])
    np.testing.assert_array_equal(zero_linear, 0)
    changed, _ = apply_update(pp, theta*.01, parts, Q)
    new = deployed(first32, sigma, 1, changed)
    physical = sigma.astype('f8')
    saved = rng.normal(size=31)*.01
    ta, tb = rng.normal(size=31), rng.normal(size=31)
    clip = (np.arange(31)%3 != 0).astype('f8')
    physical_a = ta*physical-clip*(new['delta'].astype('f8')-saved)
    physical_b = tb*physical-clip*(new['delta'].astype('f8')-saved)
    independent = float(np.sum(physical_a*physical_b/(physical*physical)))
    np.testing.assert_allclose(crosshalf_loss(ta, tb, new['delta'], saved, physical, clip), independent, rtol=1e-12, atol=1e-12)
    # Exercise the exact production exclusion path, including the fold parent.
    block_grams = [np.broadcast_to(G, (15, 577, 577))]*5
    block_targets = [np.broadcast_to(b, (15, 577))]*5
    first_fit = fit_excluding_block(block_grams, block_targets, 4, pp, U)
    block_targets[4] = np.broadcast_to(b+1e12, (15, 577))
    second_fit = fit_excluding_block(block_grams, block_targets, 4, pp, U)
    for key in first_fit[0]:
        np.testing.assert_array_equal(first_fit[0][key], second_fit[0][key])
        np.testing.assert_array_equal(first_fit[1][key], second_fit[1][key])
    for index in (2, 3, 4):
        np.testing.assert_array_equal(first_fit[index], second_fit[index])
    return dict(silu_row_affine_derivative=True, q257_x321_raw577_mapping=True, head_sensitivity_orientation=True,
                dense_residual_projection_and_shared_solve=True, compensation_to_existing_slots=True,
                excluded_block_mutation_invariant=True, independent_first128_baseline_replay=True,
                production_parent_and_candidate_excluded_label_invariance=True,
                direct_crosshalf_metric_identity=True, clipping_mask_applied_to_all_features=True,
                zero_affine_update_exact_float32_forward=True, nonlinear_float32_crosshalf_metric=True)


def custody(prereg):
    """Verify all source/capture/reconstruction custody before opening labels."""
    assert sha(SIB/'DESIGN_LOCAL_v1.json') == prereg['design_sha256']
    design = read(SIB/'DESIGN_LOCAL_v1.json')
    train = prereg['train']
    assert train == design['train'] and len(set(train)) == len(train) == 90 and design['weight_hash_disjointness']
    assert not set(train)&set(design['protected_seed_ids'])
    hfiles, reconstruction = cf.validate_reconstructed_first128(train)
    ready = read(SIB/'CAPTURE_READY_v1.json')
    assert ready['successful'] and ready['design_sha256'] == prereg['design_sha256']
    assert ready['hook_sha256'] == cf.HOOK and ready['plan_sha256'] == cf.PLAN
    assert sha(ready['plan']) == cf.PLAN and sha(ready['recipe']) == ready['recipe_sha256']
    plan = read(ready['plan'])
    assert [seed for group in plan['groups'] for seed in group] == train
    complete, manifest = read(CAP/'COMPLETE.json'), read(CAP/'RUN_MANIFEST.json')
    assert complete['successful'] and complete['plan_sha256'] == cf.PLAN
    assert manifest['candidate_sha256'] == SCI and manifest['blas_threads'] == 4
    decoder = module('enc2_parameter_decoder', V/'fit_frontier_v1.py', 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363')
    source = V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
    assert sha(source) == SCI
    literal, _ = decoder.literals_and_functions(source.read_text())
    original = decoder.decode(literal['_GNN_LITERAL'])
    current = H1/'native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py'
    assert sha(current) == prereg['baseline_science_sha256']
    current_literal, _ = decoder.literals_and_functions(current.read_text())
    deployed_parent = decoder.decode(current_literal['_GNN_LITERAL'])
    for key in original:
        if key not in ('head', 'head_b'):
            np.testing.assert_array_equal(original[key], deployed_parent[key])
    records, proof = {}, []
    for index, group in enumerate(plan['groups'], 1):
        folder = CAP/('part%02d'%index)
        gp = CAP/('part%02d_GUARD.json'%index)
        guardcheck(gp, 4)
        part = read(folder/'manifest.json')
        assert part['sha256'] == SCI and part['diagnostic_only'] and part['environment']['thread_provenance']['blas_threads'] == 4
        capdir = folder/'gnn_full_state_v1'
        meta, done = read(capdir/'METADATA_v1.json'), read(capdir/'CAPTURE_COMPLETE_v1.json')
        assert meta['source_sha256'] == SCI and meta['hook_sha256'] == cf.HOOK and meta['target_free']
        captured = decoder.decode(meta['gnn_literal'])
        assert set(captured) == set(original) and all(np.array_equal(original[key], captured[key]) for key in original)
        assert done['complete'] and done['target_free']
        for seed in group:
            weights, truth = plan['rows'][str(seed)]['weights'], design['rows'][str(seed)]
            wp, tp = ROOT/weights['path'], ROOT/truth['truth']['path']
            pred = folder/('diagnostic_pred_%d.bin'%seed)
            assert pred.stat().st_size == 65536 and wp.stat().st_size == weights['bytes'] and sha(wp) == weights['sha256']
            assert tp.stat().st_size == truth['truth']['bytes'] and sha(tp) == truth['truth']['sha256']
            assert truth['raw_float32_sha256'] == weights['raw_float32_sha256'] and weights['raw_float32_sha256'] not in design['protected_raw_weight_hashes']
            layers = []
            for layer in range(1, 16):
                matches = [row for row in done['files'] if row['rank'] == seed and row['phase'] == 'scored' and row['layer'] == layer]
                assert len(matches) == 1
                item = matches[0]
                path = capdir/item['path']
                assert sha(path) == item['sha256']
                first = hfiles[seed, layer]
                assert sha(first['path']) == first['sha256']
                layers.append(dict(layer=layer, capture_path=str(path), capture_sha256=item['sha256'], first=first))
            record = dict(seed=seed, weights_path=str(wp), weights_sha256=sha(wp), raw_weights_sha256=weights['raw_float32_sha256'],
                          truth_path=str(tp), truth_sha256=sha(tp), prediction_path=str(pred), prediction_sha256=sha(pred),
                          guard_sha256=sha(gp), layers=layers)
            records[seed] = record
            proof.append(record)
    assert list(records) == train
    return original, deployed_parent, records, dict(reconstruction=reconstruction, rows=proof, all_custody_before_labels=True)


def data_rows(seeds, records, original):
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
            z = first@original['enc2_w'].T+original['enc2_b']
            dphi = derivative(z)
            T = np.column_stack([z.astype('f8')*dphi, dphi])*mask[:, None]
            inherited = np.asarray(weights[layer], dtype='f8').T@np.stack([residual_a[layer-1], residual_b[layer-1]], axis=1)
            y_a = (residual_a[layer]-gate*inherited[:, 0])/physical
            y_b = (residual_b[layer]-gate*inherited[:, 1])/physical
            assert np.isfinite(y_a).all() and np.isfinite(y_b).all()
            yield dict(seed=seed, layer=layer, X=X, Z=np.column_stack([X, T]), first=first, sigma=sigma,
                       physical=physical, mask=mask, saved=saved, y_a=y_a, y_b=y_b)
        del weights


def evaluate(seeds, records, original, parent, candidate, linear, baseline):
    losses, trust = {}, []
    for row in data_rows(seeds, records, original):
        seed, layer = row['seed'], row['layer']
        old = deployed(row['first'], row['sigma'], layer, parent)
        new = deployed(row['first'], row['sigma'], layer, candidate)
        target_a, target_b = row['y_a'], row['y_b']
        acc = losses.setdefault(seed, dict(seed=seed, baseline=0., candidate=0., legacy_first128_moment=0.))
        acc['baseline'] += crosshalf_loss(target_a, target_b, old['delta'], row['saved'], row['physical'], row['mask'])
        acc['candidate'] += crosshalf_loss(target_a, target_b, new['delta'], row['saved'], row['physical'], row['mask'])
        moment_correction = row['X']@baseline[layer-1]
        acc['legacy_first128_moment'] += float((target_a-moment_correction)@(target_b-moment_correction))
        difference = row['mask']*(new['delta'].astype('f8')-old['delta'].astype('f8'))
        prediction = row['physical']*(row['Z']@linear[layer-1])
        trust.append(dict(seed=seed, layer=layer, head_error2=float(difference@difference),
                          head_energy=float(np.sum((row['mask']*old['delta'].astype('f8'))**2)),
                          state_error2=float(np.sum((new['state'].astype('f8')-old['state'].astype('f8'))**2)),
                          state_energy=float(np.sum(old['state'].astype('f8')**2)),
                          message_error2=float(np.sum((new['message'].astype('f8')-old['message'].astype('f8'))**2)),
                          message_energy=float(np.sum(old['message'].astype('f8')**2)),
                          curvature2=float(np.sum((difference-prediction)**2)), linear2=float(prediction@prediction)))
    result = []
    for seed in seeds:
        row = losses[seed]
        assert row['baseline'] > 0 and row['candidate'] > 0 and row['legacy_first128_moment'] > 0
        row['incremental_pct'] = 100*(row['candidate']/row['baseline']-1)
        row['float32_baseline_vs_legacy_moment_pct'] = 100*(row['baseline']/row['legacy_first128_moment']-1)
        result.append(row)
    return result, trust_summary(trust)


def summarize(rows):
    change = np.asarray([row['incremental_pct'] for row in rows])
    return dict(n=len(rows), mean_incremental_pct=float(change.mean()), incrementally_lower=int(np.sum(change < 0)),
                se_incremental_pct=float(change.std(ddof=1)/np.sqrt(len(change))),
                pooled_incremental_pct=100*(sum(row['candidate'] for row in rows)/sum(row['baseline'] for row in rows)-1),
                mean_float32_baseline_vs_legacy_moment_pct=float(np.mean([row['float32_baseline_vs_legacy_moment_pct'] for row in rows])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = selftest()
    if args.self_test:
        print(json.dumps(checks), flush=True)
        return
    assert sys.platform == 'linux' and sha(HERE/'PREREG_v1.json') == PREREG
    prereg = read(HERE/'PREREG_v1.json')
    review = read(HERE/'FIT_REVIEW_v1.json')
    assert review['successful'] and review['one_fixed_fit_authorized']
    assert review['prereg_sha256'] == PREREG and review['fitter_sha256'] == sha(__file__)
    pricepath = RUNS/'enc2_affine_local90_price_v1/PRICE_v1.json'
    price = read(pricepath)
    assert sha(pricepath) == review['price_sha256'] and price['passes_price_gate'] and price['no_capture_or_truth_arrays_read']
    assert guardcheck(RUNS/'enc2_affine_local90_price_v1_GUARD_v1.json', 1) == review['price_guard_sha256']
    assert guardcheck(RUNS/'enc2_affine_local90_selftest_v1_GUARD_v1.json', 1) == review['selftest_guard_sha256']
    original, deployed_parent, records, proof = custody(prereg)
    train, folds = prereg['train'], prereg['folds']
    assert folds == [train[index::5] for index in range(5)]
    U = cf.nullspace(original['enc4_w'])
    OUT.mkdir(exist_ok=False)
    write(OUT/'PREREG_COPY_v1.json', prereg)
    grams = np.zeros((5, 15, 577, 577))
    crosses = np.zeros((5, 15, 577))
    full_G = np.zeros((15, 321, 321))
    full_b = np.zeros((15, 321))
    block = {seed: index for index, seeds in enumerate(folds) for seed in seeds}
    loaded = []
    for row in data_rows(train, records, original):
        seed, layer = row['seed'], row['layer']-1
        Z, X, target = row['Z'], row['X'], .5*(row['y_a']+row['y_b'])
        gram, cross = Z.T@Z, Z.T@target
        # Preserve the original321-column products/order for whole90 parent parity.
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
    expected = parent_parameters(original, full_baseline)
    assert set(expected) == set(deployed_parent)
    for key in expected:
        np.testing.assert_array_equal(expected[key], deployed_parent[key], err_msg='full90 frozen parent '+key)
    write(OUT/'BASELINE_PARITY_v1.json', dict(utc=utc(), successful=True, exact_all_deployed_tensors=True,
          parent_sha256=prereg['baseline_science_sha256'], changed_tensor_digest=model_digest(expected)))
    validation, reports = [], []
    for index, held in enumerate(folds):
        chosen = [j for j in range(5) if j != index]
        training = [seed for seed in train if seed not in held]
        parent, candidate, linear, baseline, theta = fit_excluding_block(grams, crosses, index, original, U)
        seal = model_digest(candidate)
        write(OUT/('FOLD%02d_SEALED_v1.json'%index), dict(utc=utc(), training=training, validation=held,
              independent_parent_digest=model_digest(parent), candidate_digest=seal,
              theta_sha256=hashlib.sha256(theta.tobytes()).hexdigest(), excluded_block=index,
              fitted_block_indices=chosen, sealed_before_validation_replay=True))
        _, train_trust = evaluate(training, records, original, parent, candidate, linear, baseline)
        rows, validation_trust = evaluate(held, records, original, parent, candidate, linear, baseline)
        assert model_digest(candidate) == seal
        report = dict(fold=index, summary=summarize(rows), training_trust=train_trust,
                      validation_trust_descriptive_only=validation_trust, rows=rows)
        write(OUT/('FOLD%02d_RESULT_v1.json'%index), report)
        reports.append(report)
        validation.extend(rows)
        print(json.dumps(dict(utc=utc(), fold=index, summary=report['summary'], train_trust=train_trust)), flush=True)
    validation.sort(key=lambda row: train.index(row['seed']))
    cv = summarize(validation)
    passed = cv['mean_incremental_pct'] <= -.15 and cv['incrementally_lower'] >= 54 and sum(row['summary']['mean_incremental_pct'] < 0 for row in reports) >= 4 and all(row['training_trust']['passed'] for row in reports)
    full_trust = None
    artifact = None
    if passed:
        parent, candidate, linear, baseline, theta = successor(sum(grams), sum(crosses), 90, original, U, baseline=full_baseline)
        _, full_trust = evaluate(train, records, original, parent, candidate, linear, baseline)
        passed = bool(full_trust['passed'])
        if passed:
            artifact = OUT/'ENC2_AFFINE_MODEL_v1.npz'
            np.savez(artifact, **{key: candidate[key] for key in CHANGED})
    result = dict(utc=utc(), status='ENC2_AFFINE_FIXED_PROXY_PASS_NATIVE_REQUIRED' if passed else 'CLOSED_ENC2_AFFINE_FIXED_GATES_FAIL',
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
