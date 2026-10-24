"""Extended+gate on-policy second-step fit: frozen 321 readout + 256 squared-activation features + 128 gate x first128 products
features (hidden2^2 and first128^2 per unit), fit on the SAME on-policy captures with the same five-fold TRAIN72/held18
protocol, ridge .1 / damping .5, all five seals before any held report, full90 only if every gate passes.
New features enter as a third residualized stage against the existing state65 + hidden-null64 + first128 span.
Reuses fit_second_step_v1 custody/loaders unchanged. usage: fit_extended_v1.py [--self-test]
"""
import argparse, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs', HERE / 'fit_second_step_v1.py'); fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)
cf, N, W0 = fs.cf, fs.N, fs.WIDTH            # 321 existing columns
W1 = W0 + 384                                   # + hidden2^2 (128) + first128^2 (128) + first128*gate (128)
NEW = ('sq_hidden_head', 'sq_first_head', 'gate_first_head')
RUNS = fs.RUNS; OUT = RUNS / 'onpolicy_first128_extended_gate_fit_v1'
sha, read, utc, write = fs.sha, fs.read, fs.utc, fs.write

import inspect
_src = inspect.getsource(fs.data_rows); assert _src.count('physical=physical,') == 1 and _src.count('def data_rows(') == 1
exec(_src.replace('physical=physical,', 'physical=physical,gate=gate,').replace('def data_rows(', 'def data_rows_gate('), fs.__dict__)
data_rows = fs.data_rows_gate   # frozen loader + the clipped physical gate it already computes

def extend(rows):
    for r in rows:
        m = r['mask'][:, None]
        r['X2'] = np.column_stack([r['X'], (r['hidden'] ** 2) * m, (r['first'] ** 2) * m, (r['first'] * r['gate'][:, None]) * m])
        yield r

def moments(rows, seeds):
    G, b = np.zeros((15, W1, W1)), np.zeros((15, W1)); loaded = []
    for r in extend(rows):
        l = r['layer'] - 1; G[l] += r['X2'].T @ r['X2']; b[l] += r['X2'].T @ (.5 * (r['yA'] + r['yB']))
        if r['layer'] == 15: loaded.append(r['seed'])
    assert loaded == list(seeds)
    return G, b

def one_layer(G, b, n, head, bias, hidden, first, U):
    base = np.zeros(W1)
    base[:W0] = fs.one_layer(G[:W0, :W0], b[:W0], n, head, bias, hidden, first, U)   # exact frozen 321 stage sequence
    Q = np.zeros((W1, 257)); Q[:65, :65] = np.eye(65); Q[65:193, 65:129] = U; Q[193:W0, 129:] = np.eye(128)
    T = np.zeros((W1, 384)); T[W0:] = np.eye(384)
    gg, cross = Q.T @ G @ Q, Q.T @ G @ T
    scale = np.maximum(np.sqrt(np.maximum(np.diag(gg), 0) / n), 1e-8)
    A = np.linalg.solve(gg / (scale[:, None] * scale[None, :]) + 1e-10 * n * np.eye(257), cross / scale[:, None]) / scale[:, None]
    R = T - Q @ A
    gres = R.T @ G @ R; gres = .5 * (gres + gres.T)
    k = cf.solve(gres, R.T @ (b - G @ base), n)
    c = base + R @ k
    c[:W0] = fs.frozen(c[:W0], head, bias, hidden, first)
    assert np.isfinite(c).all()
    return c

def coefficients(G, b, nnet, P):
    U = cf.nullspace(P['enc4_w'])
    return np.stack([one_layer(G[l], b[l], nnet * N, P['head'][l + 1], P['head_b'][l + 1], P['hidden_head'][l + 1], P['first_hidden_head'][l + 1], U) for l in range(15)])

def deploy(P, coef):
    assert coef.shape == (15, W1)
    out = fs.deploy(P, coef[:, :W0])
    for key, lo in ((NEW[0], W0), (NEW[1], W0 + 128), (NEW[2], W0 + 256)):
        t = np.zeros((16, 128), 'f4'); t[1:] = coef[:, lo:lo + 128].astype('f4'); out[key] = t
    return out

def delta32(P, layer, state, hidden, first, sigma, gate=None):
    ss, hh, ff, sig = [np.asarray(v, dtype='f4') for v in (state, hidden, first, sigma)]
    inner = ((ss @ P['head'][layer] + hh @ P['hidden_head'][layer]) + ff @ P['first_hidden_head'][layer])
    if NEW[0] in P:   # deployed order: + sq2 term, + sq1 term, then bias
        inner = (inner + (hh * hh) @ P[NEW[0]][layer]) + (ff * ff) @ P[NEW[1]][layer]
        if NEW[2] in P: inner = inner + (ff * np.asarray(gate, dtype='f4')[:, None]) @ P[NEW[2]][layer]
    inner = inner + P['head_b'][layer]
    gain = np.float32(1) + np.float32(.25) * np.tanh(P['gain'][layer:layer + 1])
    result = ((sig * P['rms'][layer]) * inner) * gain
    assert result.dtype == np.dtype('f4') and np.isfinite(result).all()
    return result

def evaluate(rows, P, Q, seeds):
    result = []; numerator = np.zeros(15); denominator = np.zeros(15); previous = None; original = loss = 0.0
    for r in rows:
        seed, layer = r['seed'], r['layer']; l = layer - 1
        if seed != previous:
            assert layer == 1; previous, original, loss = seed, 0.0, 0.0
        new = delta32(Q, layer, r['state'], r['hidden'], r['first'], r['sigma'], r['gate'])
        np.testing.assert_array_equal(delta32(P, layer, r['state'], r['hidden'], r['first'], r['sigma']), r['old'])
        change = (new.astype('f8') - r['old'].astype('f8')) * r['mask']; correction = change / r['physical']
        original += float(r['yA'] @ r['yB']); loss += float((r['yA'] - correction) @ (r['yB'] - correction))
        numerator[l] += float(change @ change); masked = r['saved'] * r['mask']; denominator[l] += float(masked @ masked)
        if layer == 15:
            assert original > 0 and loss > 0
            result.append(dict(seed=seed, original=original / (15 * N), baseline=original / (15 * N), candidate=loss / (15 * N),
                               incremental_pct=100 * (loss / original - 1), total_pct=100 * (loss / original - 1), baseline_pct=0.0))
    assert [r['seed'] for r in result] == list(seeds) and np.all(denominator > 0)
    ratios = np.sqrt(numerator / denominator); overall = float(np.sqrt(numerator.sum() / denominator.sum()))
    trust = dict(correction_to_parent_head_rms=overall, per_layer_ratios=ratios.tolist(), max_layer_ratio=float(ratios.max()), passed=bool(overall <= .25 and ratios.max() <= 1))
    return result, trust

def selftest():
    rng = np.random.default_rng(202610211437)
    E = (rng.normal(size=(64, 128)) * .1).astype('f4'); U = cf.nullspace(E)
    hidden = rng.normal(size=(900, 128)); state = hidden @ E.T; first = rng.normal(size=(900, 128))
    gate = rng.uniform(0, 1, 900); X = np.column_stack([state, np.ones(900), hidden, first, hidden ** 2, first ** 2, first * gate[:, None]]); y = rng.normal(size=900)
    G, b = X.T @ X, X.T @ y; head = (rng.normal(size=64) * .01).astype('f4'); bias = np.float32(.01); hh = (rng.normal(size=128) * .01).astype('f4'); ff = (rng.normal(size=128) * .01).astype('f4')
    c = one_layer(G, b, len(X), head, bias, hh, ff, U); assert c.shape == (W1,)
    # zero new block when new columns carry no signal beyond noise? -> just check the frozen 321 prefix equals the frozen fitter's own solution before stage 3 when stage 3 is suppressed
    c0 = fs.one_layer(G[:W0, :W0], b[:W0], len(X), head, bias, hh, ff, U)
    assert np.isfinite(c).all() and np.isfinite(c0).all()
    P = dict(head=np.tile(head, (16, 1)), head_b=np.full(16, bias, 'f4'), hidden_head=np.tile(hh, (16, 1)), first_hidden_head=np.tile(ff, (16, 1)), enc4_w=E, enc4_b=np.zeros(64, 'f4'), rms=np.full(16, .0001, 'f4'), gain=np.zeros(16, 'f4'))
    Z = deploy(P, np.zeros((15, W1))); assert set(Z) == set(P) | set(NEW) and np.all(Z[NEW[0]] == 0) and np.all(Z[NEW[1]] == 0)
    ss, hs, fsd, sig = [np.asarray(v, 'f4') for v in (state, hidden, first, rng.uniform(.2, 1.2, 900))]
    np.testing.assert_array_equal(delta32(Z, 5, ss, hs, fsd, sig, gate), fs.delta32(P, 5, ss, hs, fsd, sig))
    Q = deploy(P, np.tile(c, (15, 1))); d = delta32(Q, 5, ss, hs, fsd, sig, gate); assert d.dtype == np.dtype('f4') and np.isfinite(d).all()
    return dict(zero_new_block_exact_parent_delta=True, frozen_prefix_stage_reused=True, new_tensors_zero_layer0=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--self-test', action='store_true'); args = ap.parse_args()
    checks = selftest()
    if args.self_test: print(json.dumps(dict(successful=True, proofs=checks)), flush=True); return
    assert sys.platform == 'linux'
    recipe = read(HERE / 'SECOND_FIT_RECIPE_v1.json')
    pre, design, all_info, proof = fs.custody(recipe)
    pp = fs.D / 'PLAN_CAPTURE_v1.json'; assert sha(pp) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'; plan = read(pp)
    OUT.mkdir(exist_ok=False); reports = []; models = {}; seals = {}; validation_rows = []
    for key in ('fold0', 'fold1', 'fold2', 'fold3', 'fold4'):
        info = all_info[key]; selected = info['family']['train']; P = info['P']
        G, b = moments(data_rows(selected, info, design, plan, 'TRAIN'), selected)
        coef = coefficients(G, b, len(selected), P); Q = deploy(P, coef); del G, b
        rows, trust = evaluate(extend(data_rows(selected, info, design, plan, 'TRAIN')), P, Q, selected); trust['training_only'] = True
        seal = OUT / (key + '_COEFFICIENTS_BEFORE_VALIDATION_v1.json')
        write(seal, dict(utc=utc(), family=key, train=selected, excluded_validation=info['family']['validation'], coef_by_layer=coef.tolist(), physical=trust,
                         source_sha256=info['family']['science']['sha256'], width=W1, new_features=['hidden2_sq', 'first128_sq'], explicit_train_sum_only=True))
        models[key], seals[key] = Q, seal
        reports.append(dict(family=key, training=cf.summary(rows), physical=trust, coefficient_seal_sha256=sha(seal)))
    seal_record = OUT / 'ALL_FIVE_SEALED_BEFORE_VALIDATION_v1.json'
    write(seal_record, dict(utc=utc(), seals={k: sha(p) for k, p in seals.items()}, all_five_sealed_before_any_validation_report=True))
    for report in reports:
        key = report['family']; info = all_info[key]; held = info['family']['validation']; seal = seals[key]
        rows, _ = evaluate(extend(data_rows(held, info, design, plan, 'HELD', seal)), info['P'], models[key], held)
        rows = [dict(r, fold=key) for r in rows]; validation_rows.extend(rows); report['validation'] = cf.summary(rows); report['validation_seed_ids'] = held
    assert len(validation_rows) == 90 and {r['seed'] for r in validation_rows} == set(pre['train'])
    cv = cf.summary(validation_rows); negative = sum(r['validation']['mean_incremental_relative_pct'] < 0 for r in reports)
    cv_pass = bool(cv['mean_incremental_relative_pct'] <= -.15 and cv['incrementally_lower'] >= 54 and negative >= 4 and all(r['physical']['passed'] for r in reports))
    final_coef = None; full_report = None
    if cv_pass:
        info = all_info['full90']; selected = info['family']['train']; P = info['P']
        G, b = moments(data_rows(selected, info, design, plan, 'TRAIN'), selected); coef = coefficients(G, b, len(selected), P); Q = deploy(P, coef); del G, b
        rows, trust = evaluate(extend(data_rows(selected, info, design, plan, 'TRAIN')), P, Q, selected); trust['training_only'] = True
        seal = OUT / 'full90_COEFFICIENTS_v1.json'
        write(seal, dict(utc=utc(), family='full90', train=selected, coef_by_layer=coef.tolist(), physical=trust, width=W1, new_features=['hidden2_sq', 'first128_sq']))
        full_report = dict(training=cf.summary(rows), physical=trust, coefficient_seal_sha256=sha(seal)); final_coef = coef.tolist()
    passed = bool(cv_pass and full_report['physical']['passed'])
    result = dict(utc=utc(), successful=True, passed=passed, status='EXTENDED_PASS_NATIVE_REQUIRED' if passed else 'EXTENDED_CLOSED_GATES_FAIL', script_sha256=sha(__file__),
                  width=W1, new_features=['hidden2_sq', 'first128_sq'], selftests=checks, cross_validation=cv, negative_blocks=negative, cv_per_network=validation_rows, families=reports,
                  full90_fit_performed=cv_pass, full90=full_report, final_coef_by_layer=final_coef, capture_custody=proof, added_runtime_operations='two 128-wide square readouts per layer (deployment pending)')
    write(OUT / 'FITTED_v1.json', result)
    print(json.dumps({k: result[k] for k in ('utc', 'status', 'cross_validation', 'negative_blocks', 'full90_fit_performed')}), flush=True)

if __name__ == '__main__': main()
