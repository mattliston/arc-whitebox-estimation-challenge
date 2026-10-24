"""Fixed first64 teacher-fidelity screen, current first skip preserved explicitly.

Stream existing authentic TRAIN90 first-feature files; never open truth arrays.
Five independent TRAIN72 selections/refits, then evaluate18 teacher-only rows.
No source export, native launch, new capture, parameter search or retry.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, ast, gzip, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent
H1 = D/'breakthrough_v1/first_hidden_readout_v1'
V = D.parent/'astra_oct20_vm_frontier_v1'
RUNS = Path('/home/arcworker/arc-runs')
CAP = RUNS/'full_state_response_train90_v1'
OUT = RUNS/'first_hidden64_current_fit_v1'
PREREG = '7f7ea6d62e2440e2f59dc69eb76ed5af971e3d46b2794039ce2363b154c241b6'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
SCI = '4da68821a5db9212035531a25dc8ab0c197a44d84a1e5482168d4223d349032c'
CAPSCI = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
RIDGE = 1e-6
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def module(name, path, pin):
    assert sha(path) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def guardcheck(path):
    guard = read(path)
    assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit'] == 0 and not guard['cleanup_errors']
    assert guard['guard_sha256'] == GUARD and guard['max_owned_swap_bytes'] == 0
    assert guard['thread_provenance']['blas_threads'] == 1
    assert guard['cap_bytes'] == 6500000000 and guard['reserve_bytes'] == 19000000000 and guard['max_active'] == 36
    return sha(path)


def moments(first, seed, layer):
    wide = first.astype('f8')
    return dict(seed=seed, layer=layer, G=wide.T@wide/len(wide), mean=wide.mean(0))


def pooled(rows):
    assert rows
    mean = sum(row['mean'] for row in rows)/len(rows)
    gram = sum(row['G'] for row in rows)/len(rows)
    covariance = gram-np.outer(mean, mean)
    return mean, (covariance+covariance.T)/2


def pivots(covariance, params, width):
    enc2 = params['enc2_w'].astype('f8')
    skip = params['first_hidden_head'][1:].astype('f8')
    norm2 = float(np.trace(enc2@covariance@enc2.T))
    norm_skip = float(np.trace(skip@covariance@skip.T))
    assert norm2 > 0 and norm_skip > 0
    target = np.concatenate([enc2.T/np.sqrt(norm2), skip.T/np.sqrt(norm_skip)], axis=1)
    residual = covariance.copy()
    cross = covariance@target
    selected = []
    for _ in range(width):
        diagonal = np.diag(residual).copy()
        score = np.sum(cross*cross, axis=1)/np.maximum(diagonal, 1e-20)
        score[diagonal <= 1e-20] = -np.inf
        score[selected] = -np.inf
        index = int(np.argmax(score))
        assert np.isfinite(score[index]) and score[index] >= 0
        selected.append(index)
        column = residual[:, index].copy()
        row = cross[index].copy()
        denominator = residual[index, index]
        cross -= np.outer(column, row)/denominator
        residual -= np.outer(column, column)/denominator
        residual = (residual+residual.T)/2
    assert len(set(selected)) == width
    return np.asarray(selected), dict(preactivation_variance=norm2, all_layer_first_skip_variance=norm_skip)


def affine(covariance, mean, selected, target_weights, target_bias):
    gram = covariance[np.ix_(selected, selected)]
    ridge = RIDGE*float(np.trace(gram))/len(selected)
    assert ridge > 0
    weights = np.linalg.solve(gram+ridge*np.eye(len(selected)), covariance[selected]@target_weights)
    bias = mean@target_weights+target_bias-mean[selected]@weights
    assert np.isfinite(weights).all() and np.isfinite(bias).all()
    return weights.astype('f4'), np.asarray(bias, 'f8'), ridge


def fit(rows, params, width=64):
    mean, covariance = pooled(rows)
    selected, normalization = pivots(covariance, params, width)
    w2, b2, ridge2 = affine(covariance, mean, selected, params['enc2_w'].astype('f8').T,
                           params['enc2_b'].astype('f8'))
    skip = params['first_hidden_head'][:, selected].copy()
    bias = params['head_b'].copy()
    ridges = {}
    for layer in range(1, len(skip)):
        layer_rows = [row for row in rows if row['layer'] == layer]
        layer_mean, layer_covariance = pooled(layer_rows)
        coefficients, intercept, ridge = affine(layer_covariance, layer_mean, selected,
                                                params['first_hidden_head'][layer].astype('f8'), 0.)
        skip[layer] = coefficients
        # Match actual deployment: the affine skip constant lives in head_b.
        bias[layer] = np.float32(float(params['head_b'][layer])+float(intercept))
        ridges[str(layer)] = ridge
    return dict(selected=selected, enc2_w=w2.T.copy(), enc2_b=b2.astype('f4'), first_hidden_head=skip,
                head_b=bias, enc2_ridge=ridge2, first_skip_ridges=ridges, selection_normalization=normalization)


def silu(value):
    assert value.dtype == np.dtype('f4')
    return value/(np.float32(1)+np.exp(-value))


def deployed(first, sigma, layer, params, model=None):
    if model is None:
        h1, w2, b2 = first, params['enc2_w'], params['enc2_b']
        skip, bias = params['first_hidden_head'][layer], params['head_b'][layer]
    else:
        h1 = first[:, model['selected']]
        w2, b2 = model['enc2_w'], model['enc2_b']
        skip, bias = model['first_hidden_head'][layer], model['head_b'][layer]
    first_scalar = h1@skip
    hidden = silu(h1@w2.T+b2)
    state = hidden@params['enc4_w'].T+params['enc4_b']
    inner = state@params['head'][layer]+hidden@params['hidden_head'][layer]+first_scalar
    inner = inner+bias
    gain = np.float32(1)+np.float32(.25)*np.tanh(params['gain'][layer:layer+1])
    delta = (sigma.astype('f4')*params['rms'][layer])*inner*gain
    message = state@params['msg_w'].T+params['msg_b']
    if model is not None:
        # Include the deployed float32 folded intercept when scoring this skip.
        first_scalar = first_scalar+(np.float32(bias)-params['head_b'][layer])
    result = dict(second_hidden=hidden, state=state, physical_complete_head=delta,
                  message_source=message, first_skip_scalar=first_scalar)
    assert all(value.dtype == np.dtype('f4') and np.isfinite(value).all() for value in result.values())
    return result


def errors(teacher, student):
    result = {}
    for name, original in teacher.items():
        difference = student[name].astype('f8')-original.astype('f8')
        result[name+'_error2'] = float(np.mean(difference*difference))
        result[name+'_energy'] = float(np.mean(original.astype('f8')**2))
    return result


LIMITS = dict(second_hidden=.005, state=.005, physical_complete_head=.01, message_source=.005, first_skip_scalar=.01)


def summarize(rows):
    result = dict(network_layers=len(rows))
    for name in LIMITS:
        energy = sum(row[name+'_energy'] for row in rows)
        assert energy > 0
        result[name+'_RMS_relative'] = float(np.sqrt(sum(row[name+'_error2'] for row in rows)/energy))
    return result


def passing(result, multiplier=1.):
    return all(result[name+'_RMS_relative'] <= limit*multiplier for name, limit in LIMITS.items())


def model_digest(model):
    digest = hashlib.sha256()
    for name in ('selected', 'enc2_w', 'enc2_b', 'first_hidden_head', 'head_b'):
        value = model[name]
        digest.update(name.encode()+str(value.shape).encode()+str(value.dtype).encode()+value.tobytes())
    return digest.hexdigest()


def selftest():
    rng = np.random.default_rng(202610210726)
    params = dict(enc2_w=rng.normal(size=(7, 10)).astype('f4'), enc2_b=rng.normal(size=7).astype('f4'),
                  first_hidden_head=rng.normal(size=(4, 10)).astype('f4'), head_b=rng.normal(size=4).astype('f4'),
                  enc4_w=rng.normal(size=(5, 7)).astype('f4'), enc4_b=rng.normal(size=5).astype('f4'),
                  head=rng.normal(size=(4, 5)).astype('f4'), hidden_head=rng.normal(size=(4, 7)).astype('f4'),
                  gain=rng.normal(size=4).astype('f4'), rms=rng.uniform(.2, 1, 4).astype('f4'),
                  msg_w=rng.normal(size=(3, 5)).astype('f4'), msg_b=rng.normal(size=3).astype('f4'))
    arrays = {(seed, layer): rng.normal(size=(97, 10)).astype('f4') for seed in range(4) for layer in range(1, 4)}
    records = [moments(value, seed, layer) for (seed, layer), value in arrays.items()]
    training = [row for row in records if row['seed'] != 3]
    model = fit(training, params, width=5)
    dense = np.concatenate([arrays[row['seed'], row['layer']].astype('f8') for row in training])
    selected = model['selected']
    x = dense[:, selected]
    y = dense@params['enc2_w'].astype('f8').T+params['enc2_b'].astype('f8')
    xc, yc = x-x.mean(0), y-y.mean(0)
    ridge = RIDGE*np.mean(np.var(x, axis=0))
    direct = np.linalg.solve(xc.T@xc/len(x)+ridge*np.eye(5), xc.T@yc/len(x))
    np.testing.assert_array_equal(model['enc2_w'], direct.astype('f4').T)
    np.testing.assert_array_equal(model['enc2_b'], (y.mean(0)-x.mean(0)@direct).astype('f4'))
    for layer in range(1, 4):
        h = np.concatenate([arrays[seed, layer].astype('f8') for seed in range(3)])
        x, y = h[:, selected], h@params['first_hidden_head'][layer].astype('f8')
        xc, yc = x-x.mean(0), y-y.mean()
        ridge = RIDGE*np.mean(np.var(x, axis=0))
        direct = np.linalg.solve(xc.T@xc/len(x)+ridge*np.eye(5), xc.T@yc/len(x))
        np.testing.assert_array_equal(model['first_hidden_head'][layer], direct.astype('f4'))
        np.testing.assert_array_equal(model['head_b'][layer], np.float32(float(params['head_b'][layer])+float(y.mean()-x.mean(0)@direct)))
    contaminated = [dict(row, G=row['G']*10000, mean=row['mean']*100) if row['seed'] == 3 else row for row in records]
    repeat = fit([row for row in contaminated if row['seed'] != 3], params, width=5)
    assert model_digest(model) == model_digest(repeat)
    sigma = np.ones(97, 'f4')
    teacher = deployed(arrays[3, 1], sigma, 1, params)
    student = deployed(arrays[3, 1], sigma, 1, params, model)
    summary = summarize([errors(teacher, student)])
    assert all(np.isfinite(value) for value in summary.values())
    # Full-width affine replay must preserve all branches to float32 tolerance.
    full = fit(training, params, width=10)
    full_output = deployed(arrays[3, 1], sigma, 1, params, full)
    for name in teacher:
        np.testing.assert_allclose(full_output[name], teacher[name], rtol=2e-4, atol=5e-5)
    return dict(centered_enc2_solver_matches_independent_dense_fit=True,
                all_first_skip_fits_and_folded_intercepts_match_dense_fit=True,
                excluded_network_statistics_cannot_change_model=True,
                float32_complete_state_hidden_message_head_paths_exercised=True,
                full_width_replay_preserves_current_skip=True)


def load_array(record, name):
    value = np.asarray(record[name]['data'], 'f4')
    assert list(value.shape) == record[name]['shape'] and np.isfinite(value).all()
    return value


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
    assert review['fitter_sha256'] == sha(__file__) and review['prereg_sha256'] == PREREG
    pricepath = RUNS/'first_hidden64_current_price_v1/PRICE_v1.json'
    price = read(pricepath)
    assert sha(pricepath) == review['price_sha256'] and price['successful'] and price['passes_price_gate']
    assert price['prereg_sha256'] == PREREG and price['no_capture_or_truth_arrays_read']
    assert guardcheck(RUNS/'first_hidden64_current_price_v1_GUARD_v1.json') == review['price_guard_sha256']
    assert guardcheck(RUNS/'first_hidden64_current_selftest_v1_GUARD_v1.json') == review['selftest_guard_sha256']
    decoder = module('first64_parameter_decoder', V/'fit_frontier_v1.py', 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363')
    reconstruction = module('first64_frozen_reconstruction_custody', H1/'reconstruct_first128_train90_v1.py', 'a88658a907430f7a45fad64f1242023d9d4176714cc5091ad7f2a38857312c8e')
    design = read(D/'truth_response95_v1/DESIGN_LOCAL_v1.json')
    assert sha(D/'truth_response95_v1/DESIGN_LOCAL_v1.json') == prereg['design_sha256']
    train = prereg['teacher_seeds']
    assert train == design['train'] and len(train) == len(set(train)) == 90 and design['weight_hash_disjointness']
    assert not set(train)&set(design['protected_seed_ids'])
    assert prereg['folds'] == [train[j::5] for j in range(5)]
    files, reconstruction_proof = reconstruction.validate_all_recon()
    custody = read(RUNS/'first_hidden128_reconstruct90_v1/CUSTODY_v1.json')
    custody_rows = {(row['seed'], row['layer']): row for row in custody['rows']}
    source = H1/'native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py'
    capture_source = V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
    assert sha(source) == SCI and sha(capture_source) == CAPSCI
    parent_literals, _ = decoder.literals_and_functions(source.read_text())
    capture_literals, _ = decoder.literals_and_functions(capture_source.read_text())
    params = decoder.decode(parent_literals['_GNN_LITERAL'])
    capture_params = decoder.decode(capture_literals['_GNN_LITERAL'])
    for name in capture_params:
        if name not in ('head', 'head_b'):
            np.testing.assert_array_equal(params[name], capture_params[name])
    def literal_extra(text):
        result = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('_HC_U', '_HCF_U', '_VMFR_ORIGINAL_W3_ENCODER'):
                        result[target.id] = np.asarray(ast.literal_eval(node.value), 'f4')
        return result
    extra, capture_extra = literal_extra(source.read_text()), literal_extra(capture_source.read_text())
    for name in extra:
        np.testing.assert_array_equal(extra[name], capture_extra[name])
    assert extra['_VMFR_ORIGINAL_W3_ENCODER'].shape == (128, 8)
    OUT.mkdir(exist_ok=False)
    write(OUT/'PREREG_COPY_v1.json', prereg)
    records = []
    for seed in train:
        for layer in range(1, 16):
            pin = files[seed, layer]
            path = Path(pin['path'])
            assert sha(path) == pin['sha256'] and path.stat().st_size == 1024*128*4
            first = np.fromfile(path, dtype='<f4').reshape(1024, 128)
            assert np.isfinite(first).all()
            record = moments(first, seed, layer)
            original = custody_rows[seed, layer]
            assert original['state_capture_sha256'] == pin['state_capture_sha256']
            record.update(first_path=str(path), first_sha256=pin['sha256'], capture_path=original['state_capture_path'],
                          capture_sha256=original['state_capture_sha256'])
            records.append(record)
        print(json.dumps(dict(utc=utc(), moments_complete=seed, network_layers=len(records))), flush=True)
    assert len(records) == 1350
    folds, all_errors, provenance = [], [], []
    for index, held in enumerate(prereg['folds']):
        training = [row for row in records if row['seed'] not in held]
        validation = [row for row in records if row['seed'] in held]
        assert len(training) == 72*15 and len(validation) == 18*15
        model = fit(training, params)
        seal = model_digest(model)
        write(OUT/('FOLD%02d_SEALED_v1.json'%index), dict(utc=utc(), training_seeds=[seed for seed in train if seed not in held],
              validation_seeds=held, selected=model['selected'].tolist(), deployed_model_sha256=seal,
              sealed_before_validation_capture_open=True))
        fold_errors = []
        for row in validation:
            assert sha(row['first_path']) == row['first_sha256'] and sha(row['capture_path']) == row['capture_sha256']
            first = np.fromfile(row['first_path'], dtype='<f4').reshape(1024, 128)
            with gzip.open(row['capture_path'], 'rt', encoding='utf8') as stream:
                captured = json.load(stream)
            layer = row['layer']
            assert captured['rank'] == row['seed'] and captured['layer'] == layer and captured['phase'] == 'scored'
            sigma = load_array(captured, 'sigma')
            target_hidden = load_array(captured, 'hidden2')
            target_state = load_array(captured, 'state')
            assert sigma.shape == (1024,) and target_hidden.shape == (1024, 128) and target_state.shape == (1024, 64)
            teacher = deployed(first, sigma, layer, params)
            # Recheck the original float32 replay gate on every actual feature.
            difference = teacher['second_hidden'].astype('f8')-target_hidden
            scale = np.finfo(np.float32).eps*(1+np.abs(target_hidden.astype('f8')))
            scaled_max = float(np.max(np.abs(difference)/scale))
            scaled_rms = float(np.sqrt(np.mean(difference*difference))/(np.finfo(np.float32).eps*(1+np.sqrt(np.mean(target_hidden.astype('f8')**2)))))
            assert scaled_max <= 128 and scaled_rms <= 16
            state_replay = float(np.max(np.abs(target_hidden.astype('f8')@capture_params['enc4_w'].astype('f8').T+capture_params['enc4_b']-target_state)))
            assert state_replay < 3e-5
            candidate = deployed(first, sigma, layer, params, model)
            error = dict(seed=row['seed'], layer=layer, **errors(teacher, candidate))
            fold_errors.append(error)
            provenance.append(dict(seed=row['seed'], layer=layer, first_sha256=row['first_sha256'],
                                   capture_sha256=row['capture_sha256'], hidden_replay_scaled_max=scaled_max,
                                   hidden_replay_scaled_rms=scaled_rms, state_replay_max_abs=state_replay))
        assert model_digest(model) == seal
        score = summarize(fold_errors)
        folds.append(dict(fold=index, validation_seeds=held, selected=model['selected'].tolist(), deployed_model_sha256=seal,
                          metrics=score, passed=passing(score)))
        all_errors.extend(fold_errors)
        print(json.dumps(dict(utc=utc(), fold=index, metrics=score, passed=passing(score))), flush=True)
    assert len(all_errors) == len(provenance) == 1350
    aggregate = summarize(all_errors)
    passed = bool(passing(aggregate) and sum(row['passed'] for row in folds) >= 4 and all(passing(row['metrics'], 2.) for row in folds))
    result = dict(utc=utc(), status='CURRENT_FIRST64_TEACHER_PASS_NATIVE_REQUIRED' if passed else 'CLOSED_CURRENT_FIRST64_TEACHER_FIDELITY_FAIL',
                  passed=passed, source_sha256=SCI, capture_source_sha256=CAPSCI, script_sha256=sha(__file__), prereg_sha256=PREREG,
                  price_sha256=sha(pricepath), root_review_sha256=sha(HERE/'FIT_REVIEW_v1.json'), selftests=checks,
                  folds=folds, aggregate=aggregate, per_network_layer=all_errors, provenance=provenance,
                  reconstructed_first128_custody=reconstruction_proof, parameter_and_hc_hcf_parity=True,
                  original_w3_metric_all128_rows_preserved=True, no_network_truth_arrays_opened=True,
                  no_candidate_source_export=True, no_native_or_adjusted_score_claim=True, no_new_capture=True)
    if passed:
        model = fit(records, params)
        selected = model['selected']
        path = OUT/'FIRST64_MODEL_v1.npz'
        np.savez(path, selected=selected, enc0_w=params['enc0_w'][selected], enc0_b=params['enc0_b'][selected],
                 enc2_w=model['enc2_w'], enc2_b=model['enc2_b'], first_hidden_head=model['first_hidden_head'], head_b=model['head_b'],
                 HC_U=extra['_HC_U'][:, selected], HCF_U=extra['_HCF_U'][:, selected])
        result.update(full90_model_sha256=sha(path), full90_deployed_model_sha256=model_digest(model), full90_selected=selected.tolist())
    write(OUT/'FITTED_v1.json', result)
    print(json.dumps(dict(utc=result['utc'], status=result['status'], aggregate=aggregate)), flush=True)


if __name__ == '__main__':
    main()
