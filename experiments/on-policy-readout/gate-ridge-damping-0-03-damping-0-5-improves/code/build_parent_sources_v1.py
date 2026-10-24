"""Build exactly six independent first128 first-stage parents; no native launch.

All TRAIN90 adaptation heads are recomputed for each selected TRAIN72 fold.
The historical777a estimator remains a frozen prior, not nested pretraining.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, ast, gzip, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = HERE.parent
V = D.parent/'astra_oct20_vm_frontier_v1'
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
PREREG = '42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
EF_SHA = '4c1c540117d808d42e53be9a45c29189b23fe81e41a06983130c00fd89820341'
CF_SHA = '8e252f559879be4d6c2bceb033bf9d84408ada1c418942bcf4d4d939d009c526'
F_SHA = 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
FIRST_FIT = 'd1bca8878d62fe96ebd18780a5c560f8e1a04d557a41eb928546449906376125'
N = 1024
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
def module(name, path, pin):
    assert sha(path) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
ef = module('op1_frozen_custody_and_parameters', D/'enc2_affine_local90_v1/fit_enc2_affine_v1.py', EF_SHA)
cf = ef.cf
assert sha(cf.__file__) == CF_SHA and (cf.RIDGE, cf.DAMPING, cf.PROJECTION_RIDGE) == (.1, .5, 1e-10)
f = module('op1_frozen_parameter_codec', V/'fit_frontier_v1.py', F_SHA)
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
def file_record(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path), effective_sha256=sha(path))
def definitions(source):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
def add_selected(grams, crosses, selections, seed, layer, gram, cross):
    for index, selected in enumerate(selections):
        if seed in selected:
            grams[index, layer] += gram
            crosses[index, layer] += cross

def selftest():
    original = cf.selftest()
    rng = np.random.default_rng(202610210939)
    first = rng.normal(size=(703, 128))
    hidden = rng.normal(size=(703, 128))
    enc = (rng.normal(size=(64, 128))*.1).astype('f4')
    state = hidden@enc.T
    X = np.column_stack([state, np.ones(703), hidden, first])
    target = rng.normal(size=703)
    parts = [slice(0, 210), slice(210, 450), slice(450, None)]
    G = [X[p].T@X[p] for p in parts]
    b = [X[p].T@target[p] for p in parts]
    selections = [{0, 1}, {1, 2}, {0, 1, 2}]
    before_G, before_b = np.zeros((3, 15, 321, 321)), np.zeros((3, 15, 321))
    after_G, after_b = np.zeros_like(before_G), np.zeros_like(before_b)
    for seed in range(3):
        for layer in range(15):
            add_selected(before_G, before_b, selections, seed, layer, G[seed], b[seed])
            add_selected(after_G, after_b, selections, seed, layer, G[seed]*(100 if seed == 2 else 1), b[seed]+(1e6 if seed == 2 else 0))
    np.testing.assert_array_equal(before_G[0], after_G[0])
    np.testing.assert_array_equal(before_b[0], after_b[0])
    np.testing.assert_array_equal(before_G[0, 0], G[0]+G[1])
    params = dict(head=(rng.normal(size=(16, 64))*.01).astype('f4'), head_b=np.zeros(16, 'f4'), enc4_w=enc)
    U = cf.nullspace(enc)
    _, c1 = cf.coefficients(before_G[0], before_b[0], 72, params, U)
    _, c2 = cf.coefficients(after_G[0], after_b[0], 72, params, U)
    np.testing.assert_array_equal(c1, c2)
    p1, p2 = ef.parent_parameters(params, c1), ef.parent_parameters(params, c2)
    for key in p1:
        np.testing.assert_array_equal(p1[key], p2[key])
    assert np.any(p1['hidden_head'][1:]) and np.any(p1['first_hidden_head'][1:])
    return dict(original_first128_selftests=original, explicit_selected_sums=True,
                production_excluded_features_and_labels_cannot_change_any_parent_tensor=True,
                state_bias_hidden_and_first_heads_all_recomputed=True, six_small_moment_accumulators=True)

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
            inherited = np.asarray(weights[layer], dtype='f8').T@np.stack([residual_a[layer-1], residual_b[layer-1]], axis=1)
            y_a = (residual_a[layer]-gate*inherited[:, 0])/physical
            y_b = (residual_b[layer]-gate*inherited[:, 1])/physical
            assert np.isfinite(y_a).all() and np.isfinite(y_b).all()
            yield dict(seed=seed, layer=layer, X=X, first=first, sigma=sigma,
                       physical=physical, mask=mask, saved=saved, y_a=y_a, y_b=y_b)
        del weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = selftest()
    if args.self_test:
        print(json.dumps(checks), flush=True)
        return
    assert sys.platform == 'linux' and sha(HERE/'PREREG_v1.json') == PREREG
    prereg, review = read(HERE/'PREREG_v1.json'), read(HERE/'ROOT_BUILD_REVIEW_v1.json')
    assert review['successful'] and review['one_parent_build_authorized']
    assert review['prereg_sha256'] == PREREG and review['builder_sha256'] == sha(__file__)
    assert not (HERE/'parents_v1').exists() and not (HERE/'PARENT_SOURCES_v1.json').exists()
    prior_view = dict(design_sha256=prereg['design_sha256'], train=prereg['train'], baseline_science_sha256=prereg['scientific_first128_parent_sha256'])
    original, first_parent, records, proof = ef.custody(prior_view)
    train = prereg['train']
    families = prereg['folds']+[dict(key='full90', train=train, validation=[])]
    for index, row in enumerate(families[:5]):
        assert row['validation'] == train[index::5]
        assert row['train'] == [seed for seed in train if seed not in row['validation']]
        assert len(row['train']) == 72 and not set(row['train'])&set(row['validation'])
    selections = [set(row['train']) for row in families]
    grams, crosses = np.zeros((6, 15, 321, 321)), np.zeros((6, 15, 321))
    loaded = []
    for row in data_rows(train, records, original):
        X = row['X']
        gram, cross = X.T@X, X.T@(.5*(row['y_a']+row['y_b']))
        add_selected(grams, crosses, selections, row['seed'], row['layer']-1, gram, cross)
        if row['layer'] == 15:
            loaded.append(row['seed'])
            print(json.dumps(dict(utc=utc(), parent_moments_loaded=row['seed'], count=len(loaded))), flush=True)
    assert loaded == train
    parents = dict(science=ROOT/prereg['parent_science_path'], release=D/'runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_release_v1.py')
    for role, path in parents.items():
        assert sha(path) == prereg['parent_'+role+'_sha256']
    literal, _ = f.literals_and_functions(parents['science'].read_text())
    paid = f.decode(literal['_GNN_LITERAL'])
    assert set(paid) == set(first_parent)
    for key in paid:
        np.testing.assert_array_equal(paid[key], first_parent[key])
    oldfit_path = RUNS/'first_hidden128_local90_fit_v1/FITTED_v1.json'
    assert sha(oldfit_path) == FIRST_FIT
    oldfit = read(oldfit_path)
    assert oldfit['passed'] and oldfit['source_sha256'] == prereg['trajectory_first_stage_source_sha256']
    U = cf.nullspace(original['enc4_w'])
    out = HERE/'parents_v1'
    out.mkdir(exist_ok=False)
    built = []
    for index, definition in enumerate(families):
        key = definition['key']
        _, coefficients = cf.coefficients(grams[index], crosses[index], len(definition['train']), original, U)
        model = ef.parent_parameters(original, coefficients)
        if key == 'full90':
            np.testing.assert_array_equal(coefficients, np.asarray(oldfit['coef_by_layer']))
            for name in paid:
                np.testing.assert_array_equal(model[name], paid[name])
        folder = out/key
        folder.mkdir()
        seal = folder/'COEFFICIENTS_v1.json'
        write(seal, dict(utc=utc(), key=key, train=definition['train'], validation=definition['validation'],
              coef_by_layer=coefficients.tolist(), prereg_sha256=PREREG, source_prior_sha256=prereg['trajectory_first_stage_source_sha256'],
              original_first128_fit_sha256=FIRST_FIT, baseline_fitter_sha256=CF_SHA, explicit_train_sum_only=True,
              all_adaptation_heads_recomputed=True, shared_historical_prior_not_nested=True))
        suffix = '\n# Independent TRAIN72 first128 diagnostic parent: payload only.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\n_GNN_LITERAL = '+repr(f.encode(model))+'\n_FAST_RESCUE_FRAC = {}\n'
        outputs = {}
        for role, path in parents.items():
            target = path if key == 'full90' else folder/('candidate_'+key+'_'+role+'_v1.py')
            if key != 'full90':
                source = path.read_text()+suffix
                compile(source, str(target), 'exec')
                assert definitions(source) == definitions(path.read_text())
                with target.open('x') as stream:
                    stream.write(source)
            encoded, _ = f.literals_and_functions(target.read_text())
            actual = f.decode(encoded['_GNN_LITERAL'])
            assert set(actual) == set(model)
            for name in actual:
                np.testing.assert_array_equal(actual[name], model[name])
            outputs[role] = file_record(target)
        build = folder/'BUILD_v1.json'
        write(build, dict(utc=utc(), key=key, outputs=outputs, coefficient_sha256=sha(seal),
              all_function_class_definitions_unchanged=True, only_existing_head_payloads_changed=True,
              no_new_shapes_ops_or_source_aliases=True, full90_exact_49bd=key == 'full90', no_native_launch=True))
        built.append(dict(key=key, train=definition['train'], validation=definition['validation'], **outputs,
                     coef_path=str(seal.relative_to(ROOT)), coef_sha256=sha(seal), build_path=str(build.relative_to(ROOT)),
                     build_sha256=sha(build), parent_fit_target_seed_ids=definition['train'],
                     parent_fit_excludes_validation=True, full90_exact_49bd=key == 'full90'))
    result = dict(utc=utc(), successful=True, prereg_sha256=PREREG, script_sha256=sha(__file__),
                  root_build_review_sha256=sha(HERE/'ROOT_BUILD_REVIEW_v1.json'), baseline_fitter_sha256=CF_SHA,
                  original_first128_fit_sha256=FIRST_FIT, families=built, selftests=checks, original_custody=proof,
                  full90_exact_49bd_tensor_parity=True, all_new_adaptation_heads_fold_fit=True,
                  shared_pretrained777a_prior_not_nested=True, no_capture_or_native_launch=True, no_bank_gain_claim=True)
    write(HERE/'PARENT_SOURCES_v1.json', result)
    print(json.dumps(dict(utc=utc(), successful=True, families=6, source_manifest_sha256=sha(HERE/'PARENT_SOURCES_v1.json'))), flush=True)

if __name__ == '__main__':
    main()
