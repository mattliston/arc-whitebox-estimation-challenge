"""Frozen clean five-block on-policy second head step; no export/native launch.

Each fold first fits its correction on TRAIN72 only, seals the deployed float32
correction, then opens that fold's 18 validation truths once for reporting.
All 540 capture/source/guard/file records are checked before any truth array.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, gzip, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

RECOVERY = Path(__file__).resolve().parent
HERE = RECOVERY.parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
OUT = RUNS/'on_policy_head90_second_fit_recovery_v1'
D = HERE.parent/'truth_response95_v1'
V = HERE.parent.parent/'astra_oct20_vm_frontier_v1'
sys.path[:0] = [str(HERE.parent/'hidden_complement90_v1'), str(V)]
import fit_hidden_complement_v1 as cf
import fit_frontier_v1 as f
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
PREREG = '0c94b0dc84bd17f0542df28e9ae56d7486c8944fe3cd08db74e6c91a76c24443'
MANIFEST = 'd6eaeb233f5f8109247d46cc68672848fb10477bd18a4076829fd267a4613136'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
N = 1024
WIDTH = 193

def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')

def guardcheck(path, threads):
    g = read(path)
    assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
    assert g['guard_sha256'] == GUARD and g['thread_provenance']['blas_threads'] == threads
    assert g['cap_bytes'] == 6500000000 and g['reserve_bytes'] == 19000000000 and g['max_active'] == 36
    assert g['max_owned_swap_bytes'] == 0
    for name in ('oom', 'oom_kill', 'oom_group_kill'):
        assert g['cgroup_events_initial'][name] == g['cgroup_events_final'][name]
    return sha(path)

def frozen(c, head, bias, hidden):
    q = c.copy()
    q[:64] = (head.astype('f8')+q[:64]).astype('f4').astype('f8')-head.astype('f8')
    q[64] = float(np.float32(float(bias)+q[64]))-float(bias)
    # The parent already has a hidden readout: deploy an addition, never overwrite.
    q[65:] = (hidden.astype('f8')+q[65:]).astype('f4').astype('f8')-hidden.astype('f8')
    assert np.isfinite(q).all()
    return q

def one_layer(G, b, n, head, bias, hidden, U):
    baseline = np.zeros(WIDTH)
    baseline[:64] = cf.solve(G[:64, :64], b[:64], n)
    baseline = frozen(baseline, head, bias, hidden)
    T = np.zeros((WIDTH, 64))
    T[65:] = U
    gg = G[:65, :65]
    cross = G[:65]@T
    scale = np.maximum(np.sqrt(np.maximum(np.diag(gg), 0)/n), 1e-8)
    A = np.linalg.solve(gg/(scale[:, None]*scale[None, :])+1e-10*n*np.eye(65), cross/scale[:, None])/scale[:, None]
    R = T.copy()
    R[:65] -= A
    gres = R.T@G@R
    gres = .5*(gres+gres.T)
    k = cf.solve(gres, R.T@(b-G@baseline), n)
    return frozen(baseline+R@k, head, bias, hidden)

def coefficients(records, P, U):
    # Explicitly sum only selected TRAIN records. No full-minus-validation moments.
    G = sum(r['G'] for r in records)
    b = sum(r['b'] for r in records)
    return np.stack([one_layer(G[l], b[l], len(records)*N, P['head'][l+1],
                              P['head_b'][l+1], P['hidden_head'][l+1], U) for l in range(15)])

def metric(r, c):
    original = float(np.sum(r['u0_sum']))
    loss = float(original-2*np.sum(c*r['b'])+np.einsum('li,lij,lj->', c, r['G'], c))
    assert original > 0 and loss > 0
    return dict(seed=r['seed'], original=original/(15*N), baseline=original/(15*N),
                candidate=loss/(15*N), incremental_pct=100*(loss/original-1),
                total_pct=100*(loss/original-1), baseline_pct=0.0)

def physical(records, c):
    numerator = np.sum([np.einsum('li,lij,lj->l', c, r['physical_gram'], c) for r in records], axis=0)
    denominator = np.sum([r['head_rms2'] for r in records], axis=0)
    assert np.all(denominator > 0) and np.all(numerator >= -1e-15)
    ratios = np.sqrt(np.maximum(numerator, 0)/denominator)
    overall = float(np.sqrt(max(float(numerator.sum()), 0)/float(denominator.sum())))
    return dict(correction_to_parent_head_rms=overall, per_layer_ratios=ratios.tolist(),
                max_layer_ratio=float(ratios.max()), training_only=True,
                passed=bool(overall <= .25 and ratios.max() <= 1))

def selftest():
    assert sha(cf.__file__) == 'e6f33d68de379965c887cac3e43fbc31766625e66dead02d8bf0308cea2e5021'
    assert sha(f.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    cf.selftest()
    rng = np.random.default_rng(202610210616)
    E = rng.normal(size=(64, 128))
    U = cf.nullspace(E)
    h = rng.normal(size=(503, 128))
    state = h@E.T+rng.normal(size=64)
    X = np.column_stack([state, np.ones(503), h])
    A = rng.normal(size=503)
    B = A+rng.normal(size=503)*.1
    head = rng.normal(size=64).astype('f4')
    bias = np.float32(.1)
    hidden = rng.normal(size=128).astype('f4')
    G, b = X.T@X, X.T@(.5*(A+B))
    zero = np.zeros(128, dtype='f4')
    expected = cf.one_layer(G, b, len(X), head, bias, U)[1]
    np.testing.assert_array_equal(one_layer(G, b, len(X), head, bias, zero, U), expected)
    c = one_layer(G, b, len(X), head, bias, hidden, U)
    np.testing.assert_array_equal(c[65:], (hidden.astype('f8')+c[65:]).astype('f4').astype('f8')-hidden.astype('f8'))
    direct = (A-X@c)@(B-X@c)
    gram = A@B-2*c@b+c@G@c
    np.testing.assert_allclose(direct, gram, rtol=1e-10, atol=1e-10)
    old = state@head+float(bias)+h@hidden
    new = state@(head.astype('f8')+c[:64])+float(bias)+c[64]+h@(hidden.astype('f8')+c[65:])
    np.testing.assert_allclose(new-old, X@c, rtol=1e-9, atol=1e-10)
    selected = np.arange(0, 503, 5)
    complement = np.asarray([i for i in range(503) if i not in selected])
    assert not set(selected)&set(complement)
    before = one_layer(X[selected].T@X[selected], X[selected].T@(.5*(A+B))[selected], len(selected), head, bias, hidden, U)
    A[complement] += 1e9
    after = one_layer(X[selected].T@X[selected], X[selected].T@(.5*(A+B))[selected], len(selected), head, bias, hidden, U)
    np.testing.assert_array_equal(before, after)
    return dict(zero_hidden_matches_frozen_original_algorithm=True, additive_hidden_float32_deployment=True,
                cross_half_gram_metric=True, excluded_label_mutation_cannot_change_fit=True)

def custody(recipe):
    assert sha(HERE/'PREREG_v1.json') == PREREG and sha(HERE/'PARENT_SOURCES_v1.json') == MANIFEST
    pre = read(HERE/'PREREG_v1.json')
    ready_path = RECOVERY/'RECOVERY_READY_v1.json'
    assert sha(ready_path) == recipe['capture_ready_sha256']
    ready = read(ready_path)
    final_path = RECOVERY/'CAPTURE_FINAL_v1.json'
    final = read(final_path)
    assert final['successful'] and final['captured_predictions'] == 540 and final['unhooked_controls'] == 18
    assert final['ready_sha256'] == sha(ready_path) and final['prereg_sha256'] == PREREG
    assert final['source_manifest_sha256'] == MANIFEST and final['no_truth_arrays_read']
    assert final['controller_sha256'] == recipe['capture_controller_sha256']
    assert final['root_review_sha256'] == sha(RECOVERY/'ROOT_RECOVERY_REVIEW_v1.json')
    assert sha(D/'DESIGN_LOCAL_v1.json') == pre['design_sha256']
    design = read(D/'DESIGN_LOCAL_v1.json')
    assert design['train'] == pre['train'] and design['weight_hash_disjointness']
    assert not set(pre['train'])&set(design['protected_seed_ids'])
    manifest = read(HERE/'PARENT_SOURCES_v1.json')
    records = {}
    proofs = []
    for family, capture in zip(manifest['families'], ready['outputs']):
        key = family['key']
        assert capture['key'] == key and capture['teacher'] == family['science']
        source = ROOT/family['science']['path']
        assert sha(source) == family['science']['sha256']
        stages = []
        files, predictions = {}, {}
        for stage in ('anchor', 'remaining'):
            item = capture['outputs'][stage]
            folder = Path(item['out'])
            audit = folder/'AUDIT_v1.json'
            proof = final['families'][key][stage]
            assert sha(audit) == proof['audit_sha256']
            assert guardcheck(Path(item['audit_guard_path']), 1) == proof['audit_guard_sha256']
            ar = read(audit)
            assert ar['successful'] and ar['family'] == key and ar['stage'] == stage
            expected_ready = ready['recovery']['original_ready_sha256'] if stage == 'anchor' else sha(ready_path)
            expected_auditor = recipe['original_capture_auditor_sha256'] if stage == 'anchor' else recipe['capture_auditor_sha256']
            assert ar['source_sha256'] == family['science']['sha256'] and ar['ready_sha256'] == expected_ready
            assert ar['source_manifest_sha256'] == MANIFEST and ar['prereg_sha256'] == PREREG
            assert ar['auditor_sha256'] == expected_auditor and ar['no_truth_arrays_read']
            assert ar['all_schema_hash_finite_checks_pass'] and ar['current_hidden_head_in_replay']
            if stage == 'anchor':
                assert ar['all3_anchor_predictions_byte_identical'] and len(ar['prediction_pairs']) == 3
                oldproof = ready['recovery']['anchor_proofs'][key]
                assert sha(audit) == oldproof['audit_sha256'] and proof['audit_guard_sha256'] == oldproof['guard_sha256']
            else:
                assert ar['remaining_uses_same_source_anchor_proof']
            for name in ('plan', 'recipe'):
                assert sha(ROOT/item[name]['path']) == item[name]['sha256'] == ar[name+'_sha256']
            for g in ar['guards']:
                assert guardcheck(Path(g['path']), 4) == g['sha256']
            for entry in ar['files']:
                p = Path(entry['path'])
                assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
                k = (entry['seed'], entry['layer'])
                assert k not in files
                files[k] = entry
            for entry in ar['predictions']:
                p = Path(entry['path'])
                assert p.stat().st_size == 65536 and sha(p) == entry['sha256']
                assert entry['seed'] not in predictions
                predictions[entry['seed']] = entry
            stages.append(dict(stage=stage, audit_sha256=sha(audit), guard_sha256=proof['audit_guard_sha256']))
        assert set(predictions) == set(pre['train']) and len(files) == 1350
        assert set(files) == {(s, l) for s in pre['train'] for l in range(1, 16)}
        lit, _ = f.literals_and_functions(source.read_text())
        P = f.decode(lit['_GNN_LITERAL'])
        records[key] = dict(family=family, files=files, predictions=predictions, P=P)
        proofs.append(dict(family=key, source_sha256=sha(source), stages=stages, all90_files_verified=True))
    assert len(records) == 6
    return pre, design, records, dict(capture_final_sha256=sha(final_path), families=proofs,
                                      all540_before_truth_array_open=True, capture_ready_sha256=sha(ready_path))

def load_records(seeds, info, design, original_plan, allowed):
    assert list(seeds) == list(allowed) and not set(seeds)&set(design['protected_seed_ids'])
    P = info['P']
    result, provenance = [], []
    for seed in seeds:
        row = design['rows'][str(seed)]
        wr = original_plan['rows'][str(seed)]['weights']
        wp = ROOT/wr['path']
        tp = ROOT/row['truth']['path']
        assert wp.stat().st_size == wr['bytes'] and sha(wp) == wr['sha256']
        assert row['raw_float32_sha256'] == wr['raw_float32_sha256']
        assert wr['raw_float32_sha256'] not in design['protected_raw_weight_hashes']
        assert tp.stat().st_size == row['truth']['bytes'] and sha(tp) == row['truth']['sha256']
        W = np.load(wp, mmap_mode='r', allow_pickle=False)
        assert W.shape == (16, N, N)
        pred = np.fromfile(info['predictions'][seed]['path'], dtype='<f4').reshape(16, N).astype('f8')
        with np.load(tp, allow_pickle=False) as z:
            assert int(z['seed']) == seed and str(z['weights_sha256']) == wr['raw_float32_sha256']
            A = z['mc_post_A'].astype('f8')-pred
            B = z['mc_post_B'].astype('f8')-pred
        assert A.shape == B.shape == (16, N) and np.isfinite(A).all() and np.isfinite(B).all()
        r = dict(seed=seed, G=[], b=[], u0_sum=[], physical_gram=[], head_rms2=[])
        for layer in range(1, 16):
            with gzip.open(info['files'][seed, layer]['path'], 'rt') as stream:
                cap = json.load(stream)
            assert cap['rank'] == seed and cap['layer'] == layer and cap['phase'] == 'scored'
            state, hidden, sigma, ng, saved = [cf.arr(cap, k) for k in ('state', 'hidden2', 'sigma', 'gate_feature', 'delta')]
            assert state.shape == (N, 64) and hidden.shape == (N, 128)
            assert sigma.shape == ng.shape == saved.shape == (N,) and np.max(np.abs(ng)) < 7.999
            gate = ng*float(P['scales'][layer, 5])+float(P['centers'][layer, 5])
            assert gate.min() > -2e-6 and gate.max() < 1+2e-6
            gate = np.clip(gate, 0, 1)
            physical_scale = sigma*float(P['rms'][layer])*(1+.25*np.tanh(float(P['gain'][layer])))
            assert physical_scale.min() > 1e-12
            replay = physical_scale*(state@P['head'][layer].astype('f8')+float(P['head_b'][layer])+hidden@P['hidden_head'][layer].astype('f8'))
            assert np.max(np.abs(replay-saved)) < 1e-8+2e-5*float(np.max(np.abs(saved)))
            mask = (pred[layer] > 1e-12).astype('f8') if layer < 15 else np.ones(N)
            X = np.column_stack([state, np.ones(N), hidden])*mask[:, None]
            inherited = np.asarray(W[layer], dtype='f8').T@np.stack([A[layer-1], B[layer-1]], axis=1)
            yA = (A[layer]-gate*inherited[:, 0])/physical_scale
            yB = (B[layer]-gate*inherited[:, 1])/physical_scale
            r['G'].append(X.T@X)
            r['b'].append(X.T@(.5*(yA+yB)))
            r['u0_sum'].append(float(yA@yB))
            direct = physical_scale[:, None]*X
            r['physical_gram'].append(direct.T@direct/N)
            r['head_rms2'].append(float(np.mean((saved*mask)**2)))
        for k in ('G', 'b', 'u0_sum', 'physical_gram', 'head_rms2'):
            r[k] = np.asarray(r[k])
        assert r['u0_sum'].sum() > 0
        result.append(r)
        provenance.append(dict(seed=seed, truth_sha256=sha(tp), weights_sha256=sha(wp),
                               prediction_sha256=info['predictions'][seed]['sha256']))
        print(json.dumps(dict(utc=utc(), family=info['family']['key'], loaded_seed=seed)), flush=True)
    return result, provenance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    checks = selftest()
    if args.self_test:
        print(json.dumps(dict(successful=True, selftests=checks)), flush=True)
        return
    recipe = read(RECOVERY/'SECOND_FIT_RECIPE_v1.json')
    review = read(RECOVERY/'ROOT_RECOVERY_REVIEW_v1.json')
    assert review['successful'] and review['fixed_fit_authorized'] and review['fitter_sha256'] == sha(__file__)
    assert review['fit_recipe_sha256'] == sha(RECOVERY/'SECOND_FIT_RECIPE_v1.json')
    assert sha(__file__) == recipe['fitter_sha256']
    assert sys.platform == 'linux'
    pre, design, all_info, capture_proof = custody(recipe)
    plan_path = D/'PLAN_CAPTURE_v1.json'
    assert sha(plan_path) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'
    original_plan = read(plan_path)
    OUT.mkdir(exist_ok=False)
    reports, validation_rows, provenance = [], [], []
    final_coef = None
    for key in ('fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'full90'):
        info = all_info[key]
        family = info['family']
        selected = family['train']
        held = family['validation']
        assert not set(selected)&set(held)
        train_records, train_provenance = load_records(selected, info, design, original_plan, selected)
        U = cf.nullspace(info['P']['enc4_w'])
        coef = coefficients(train_records, info['P'], U)
        trust = physical(train_records, coef)
        seal = OUT/(key+'_COEFFICIENTS_BEFORE_VALIDATION_v1.json')
        write(seal, dict(utc=utc(), family=key, train=selected, excluded_validation=held,
                         coef_by_layer=coef.tolist(), physical=trust, truth_read_seed_ids=selected,
                         explicit_train_sum_only=True, no_validation_truth_array_yet=True,
                         source_sha256=family['science']['sha256'], prereg_sha256=PREREG))
        training = cf.summary([metric(r, coef) for r in train_records])
        del train_records
        if held:
            held_records, held_provenance = load_records(held, info, design, original_plan, held)
            rows = [dict(metric(r, coef), fold=key, sealed_coef_sha256=sha(seal)) for r in held_records]
            validation_rows += rows
            cv = cf.summary(rows)
            del held_records
        else:
            held_provenance, cv = [], None
            final_coef = coef.copy()
        provenance.append(dict(family=key, training=train_provenance, validation=held_provenance,
                               coefficient_seal_sha256=sha(seal), validation_opened_only_after_coefficient_seal=True))
        reports.append(dict(family=key, training=training, validation=cv, physical=trust,
                            train=selected, validation_seed_ids=held, coefficient_seal_sha256=sha(seal)))
    assert len(validation_rows) == 90 and {r['seed'] for r in validation_rows} == set(pre['train'])
    assert len({r['seed'] for r in validation_rows}) == 90
    cv = cf.summary(validation_rows)
    negative = sum(r['validation']['mean_incremental_relative_pct'] < 0 for r in reports[:5])
    passed = bool(cv['mean_incremental_relative_pct'] <= -.15 and cv['incrementally_lower'] >= 54
                  and negative >= 4 and all(r['physical']['passed'] for r in reports))
    result = dict(utc=utc(), successful=True, passed=passed,
                  status='CLEAN_FIVE_BLOCK_PROXY_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',
                  script_sha256=sha(__file__), prereg_sha256=PREREG, source_manifest_sha256=MANIFEST,
                  selftests=checks, capture_custody=capture_proof, cross_validation=cv,
                  cv_kind='five_block_clean_two_stage_parent_and_correction_train72_exclusion',
                  cv_per_network=validation_rows, negative_blocks=negative, families=reports,
                  final_coef_by_layer=final_coef.tolist(), provenance=provenance,
                  parent_science_sha256=pre['parent_science_sha256'], parent_release_sha256=pre['parent_release_sha256'],
                  added_runtime_operations=0, no_protected_truth_read=True, no_adjusted_gain_claim=True,
                  no_candidate_export=True, no_native_launch=True, no_grid_or_repeated_step=True)
    write(OUT/'FITTED_v1.json', result)
    print(json.dumps({k: result[k] for k in ('utc', 'status', 'cross_validation', 'negative_blocks')}), flush=True)

if __name__ == '__main__':
    main()
