"""Root-only TRAIN grouped-CV diagnostic; no full-data fit, candidate, or model export."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, resource, sys, time

D = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    assert pin(record['path']) == record, record['path']
    return Path(record['path']).read_bytes()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args(); started = time.monotonic()
    pp = pin(D/'protocol.json'); assert pp['sha256'] == args.protocol_sha
    p = json.loads(checked(pp))
    assert p['schema'] == 'rrpi-pooled-head-grouped-cv-v1' and p['root_diagnostic_authorized']
    assert p['ridge'] == .001 and p['folds'] == 5 and p['features'] == [65, 129]
    assert p['fold_assignment'] == 'roster_index_modulo_5' and p['no_full_data_fit_or_export']
    assert p['gates'] == dict(augmented_raw_pct_max=0., raw_strict=True,
        augmented_adjusted_pct_max=-.2, fold_adjusted_wins_min=4,
        augmented_vs_additive_adjusted_pct_max=-.1, all_finite=True)
    for value in p['pins'].values(): checked(value)
    assert p['pins']['fitter'] == pin(__file__)
    assert Path(sys.executable).resolve() == Path(p['runtime']['path']).resolve(); checked(p['runtime'])
    assert all(os.environ.get(k) == '1' for k in p['thread_env'])
    index = json.loads(checked(p['pins']['data_index']))
    assert not index['validation_targets_included'] and index['parameter_identity']['passed']
    for v in index['authorities'].values(): checked(v)
    authority = json.loads(checked(index['authorities']['train160_protocol']))
    assert index['ranks'] == authority['training_ranks'] and len(set(index['ranks'])) == 160
    assert not set(index['ranks']) & set(authority['validation']['fixed40'])
    assert [r['rank'] for r in index['rows']] == index['ranks']
    assert [r['truth'] for r in index['rows']] == [r['truth'] for r in authority['data']]
    assert index['baseline'] == p['baseline'] and index['cached_source'] == p['cached_source']
    baseline = checked(index['baseline']).decode(); cached = checked(index['cached_source']).decode()
    for v in index['capture_evidence']:
        g = json.loads(checked(v['guard'])); c = json.loads(checked(v['completion']))
        assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0 and g['error'] is None
        assert not g['remaining_rows'] and g['max_swap_growth_bytes'] == 0
        assert not c['targets_accessed'] and c['parent_sha256'] == index['cached_source']['sha256']
    spec = importlib.util.spec_from_file_location('pooled_head_payload_reader', p['pins']['source_editor']['path'])
    editor = importlib.util.module_from_spec(spec); spec.loader.exec_module(editor)
    payload = editor.decoded_payload(baseline)
    assert payload == editor.decoded_payload(cached) and len(payload) == 15

    # Root executes all numerical work under the dedicated single-thread guard.
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(v['num_threads'] == 1 for v in threadpool_info())
    old = np.asarray(editor.terminal_head(baseline), dtype=np.float64)
    assert old.shape == (65,) and np.isfinite(old).all()
    rms = np.frombuffer(payload['rms'][1], dtype='<f4').reshape(payload['rms'][0])[15]
    gain = np.frombuffer(payload['gain'][1], dtype='<f4').reshape(payload['gain'][0])[15]
    gain_factor = (np.float32(1) + np.float32(.25)*np.tanh(np.asarray([gain], dtype=np.float32)))[0]
    packets = []
    for row in index['rows']:
        checked(row['packet']); checked(row['truth'])
        # Only the original two-network smoke independently checked all16 and bill.
        assert row['record']['independent_all16_and_bill_exact'] is (True if row['capture_group'] == 0 else None)
        assert not row['record']['targets_accessed']
        assert row['record']['sha256'] == row['packet']['sha256'] and row['record']['weights_sha256'] == row['weights']['sha256']
        assert row['record']['flops'] == row['flops'] and type(row['flops']) is int and 0 < row['flops'] <= 2**41
        with np.load(row['packet']['path'], allow_pickle=False) as z:
            control, sigma, prediction = z['control'], z['sigma'], z['prediction']
            assert control.shape == (1024, 94) and control.dtype == np.float32
            assert sigma.shape == (1024,) and prediction.shape == (16, 1024)
            assert all(np.isfinite(a).all() for a in (control, sigma, prediction)) and (sigma > 0).all()
            hidden = control[:, 30:].copy()
            physical = (sigma.astype(np.float32)*np.float32(rms)).astype(np.float64)*np.float64(gain_factor)
            parent = prediction[-1].astype(np.float64)
        with np.load(row['truth']['path'], allow_pickle=False) as z:
            target = z['mc_post']; assert target.shape == (16, 1024) and np.isfinite(target).all()
            truth = target[-1].astype(np.float64)
        assert np.isfinite(physical).all()
        packets.append(dict(rank=row['rank'], hidden=hidden, mean=hidden.mean(axis=0, dtype=np.float64),
            physical=physical, parent=parent, truth=truth, y=truth-parent, weight=max(.1, row['flops']/2**41)))
    assert len(packets) == 160

    def basis(row, center):
        h = row['hidden']
        b = row['physical'][:, None]*np.column_stack((h, np.ones(1024), h*(row['mean']-center)))
        assert b.shape == (1024, 129) and np.isfinite(b).all()
        return b

    def solve(gram_sum, rhs_sum, weight_sum):
        # Algebraically the old head160 weighted-RMS ridge, accumulated packetwise.
        moment = gram_sum/weight_sum; scale = np.sqrt(np.diag(moment))
        assert np.isfinite(scale).all() and np.min(scale) > 1e-20
        gram = moment/scale[:, None]/scale[None, :] + p['ridge']*np.eye(len(scale))
        rhs = rhs_sum/weight_sum/scale
        standardized = np.linalg.solve(gram, rhs)
        error = float(np.linalg.norm(gram@standardized-rhs)/max(np.linalg.norm(rhs), 1e-30))
        coef = standardized/scale
        assert np.isfinite(coef).all() and error < 1e-9
        return coef, dict(normal_equation_relative_error=error, condition=float(np.linalg.cond(gram)))

    # Small independent augmented least-squares certificate; no training decisions.
    rng = np.random.Generator(np.random.PCG64DXSM(2026092062))
    tb = rng.normal(size=(113, 7)); ty = rng.normal(size=113); tw = np.linspace(.1, 1, 113)
    got, certificate = solve(tb.T@(tw[:, None]*tb), tb.T@(tw*ty), tw.sum())
    wn = tw/tw.sum(); ts = np.sqrt(np.sum(wn[:, None]*tb*tb, axis=0))
    expected = np.linalg.lstsq(np.vstack((np.sqrt(wn)[:, None]*tb/ts, np.sqrt(p['ridge'])*np.eye(7))),
        np.r_[np.sqrt(wn)*ty, np.zeros(7)], rcond=None)[0]/ts
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-12)

    def comparison(rows, candidate, parent):
        a = np.asarray([r[candidate] for r in rows]); b = np.asarray([r[parent] for r in rows])
        w = np.asarray([r['weight'] for r in rows])
        assert np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()
        return dict(raw_pct=float(100*(a.mean()/b.mean()-1)),
            adjusted_pct=float(100*((a*w).mean()/(b*w).mean()-1)), wins=int(np.sum(a < b)))

    folds = []; oof = []
    for fold in range(5):
        train = [i for i in range(160) if i % 5 != fold]
        test = [i for i in range(160) if i % 5 == fold]
        assert len(train) == 128 and len(test) == 32 and not set(train) & set(test)
        center = np.mean([packets[i]['mean'] for i in train], axis=0, dtype=np.float64)
        gram = np.zeros((129, 129)); rhs = np.zeros(129); weight_sum = 0.
        for i in train:
            row = packets[i]; b = basis(row, center); w = row['weight']
            gram += w*(b.T@b); rhs += w*(b.T@row['y']); weight_sum += 1024*w
        coef65, eq65 = solve(gram[:65, :65], rhs[:65], weight_sum)
        coef129, eq129 = solve(gram, rhs, weight_sum)
        # Same rounded-head/clipped-parent surrogate as old head160; new weights also float32.
        coef65 = (old+coef65).astype(np.float32).astype(np.float64)-old
        coef129[:65] = (old+coef129[:65]).astype(np.float32).astype(np.float64)-old
        coef129[65:] = coef129[65:].astype(np.float32).astype(np.float64)
        assert np.isfinite(coef65).all() and np.isfinite(coef129).all()
        scores = []
        for i in test:
            row = packets[i]; b = basis(row, center)
            additive = np.maximum(row['parent']+b[:, :65]@coef65, 0).astype(np.float32).astype(np.float64)
            augmented = np.maximum(row['parent']+b@coef129, 0).astype(np.float32).astype(np.float64)
            assert np.isfinite(additive).all() and np.isfinite(augmented).all()
            scores.append(dict(rank=row['rank'], roster_index=i, fold=fold, weight=row['weight'],
                parent_mse=float(np.mean((row['parent']-row['truth'])**2)),
                additive_mse=float(np.mean((additive-row['truth'])**2)), augmented_mse=float(np.mean((augmented-row['truth'])**2))))
        oof.extend(scores)
        folds.append(dict(fold=fold, train_ranks=[packets[i]['rank'] for i in train],
            test_ranks=[packets[i]['rank'] for i in test], center=center.tolist(), equations65=eq65, equations129=eq129,
            augmented_vs_parent=comparison(scores, 'augmented_mse', 'parent_mse'),
            additive_vs_parent=comparison(scores, 'additive_mse', 'parent_mse'),
            augmented_vs_additive=comparison(scores, 'augmented_mse', 'additive_mse')))
    oof.sort(key=lambda r:r['roster_index']); assert [r['rank'] for r in oof] == index['ranks']
    aug = comparison(oof, 'augmented_mse', 'parent_mse'); extra = comparison(oof, 'augmented_mse', 'additive_mse')
    wins = sum(f['augmented_vs_parent']['adjusted_pct'] < 0 for f in folds)
    gates = dict(augmented_raw_gain=aug['raw_pct'] < 0, augmented_adjusted_gain=aug['adjusted_pct'] <= -.2,
        fold_adjusted_wins=wins >= 4, added_feature_adjusted_gain=extra['adjusted_pct'] <= -.1, all_finite=True)
    result = dict(schema='rrpi-pooled-head-grouped-cv-result-v1', completed=True, protocol=pp,
        data_index=p['pins']['data_index'], baseline=index['baseline'], cached_source=index['cached_source'],
        ridge=p['ridge'], fold_adjusted_wins=wins, augmented_vs_parent=aug, augmented_vs_additive=extra,
        additive_vs_parent=comparison(oof, 'additive_mse', 'parent_mse'), gates=gates,
        diagnostic_passed=all(gates.values()), full_fit_eligible=all(gates.values()), full_fit_authorized=False,
        folds=folds, rows=oof, independent_solver_max_error=float(np.max(np.abs(got-expected))), solver_certificate=certificate,
        validation_targets_read=False, all15_payload_fields_identical=True, old_s31_feature_trajectory=True,
        independent_all16_and_bill_cache_check_ranks=[r['rank'] for r in index['rows'] if r['capture_group'] == 0],
        clipped_parent_surrogate=True, cached_parent_bill_proxy=True, added_inference_bill_measured=False,
        actual_current_native_score_measured=False, full_data_fit=False, candidate_exported=False, model_exported=False,
        model_executed=False, bank_qualified=False, elapsed_s=time.monotonic()-started,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    with (D/'cv_result.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps({k:result[k] for k in ('completed','diagnostic_passed','augmented_vs_parent','additive_vs_parent',
        'augmented_vs_additive','fold_adjusted_wins','gates','elapsed_s','peak_rss_bytes')}), flush=True)


if __name__ == '__main__': main()
