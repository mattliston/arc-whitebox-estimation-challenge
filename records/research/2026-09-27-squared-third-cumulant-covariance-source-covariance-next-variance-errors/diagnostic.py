"""Passive native source alignment; the hypothetical delta never enters S31."""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import json
import math
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops

D = Path(__file__).resolve().parent
H = D.parent
P2 = H.parent
BANK = H / 'candidate_old_both_tail_trace_bounded_v6.py'
FRAGMENT = H / 'quadratic_third_covariance_fragment_v1.py'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def cross_stats(ea, eb, delta):
    base = float(np.sum(ea * eb, dtype=np.float64))
    dot = float(np.sum(delta * (ea + eb) * .5, dtype=np.float64))
    energy = float(np.sum(delta * delta, dtype=np.float64))
    changed = base + 2 * dot + energy
    return dict(base_cross=base, changed_cross=changed, error_dot_delta=dot,
                delta_energy=energy,
                change_pct=100 * (changed / base - 1) if base > 0 else None,
                error_cosine=dot / math.sqrt(base * energy) if base > 0 and energy > 0 else None,
                half_a_base=float(np.sum(ea * ea)), half_b_base=float(np.sum(eb * eb)),
                half_a_changed=float(np.sum((ea + delta) ** 2)),
                half_b_changed=float(np.sum((eb + delta) ** 2)))


def diagonal_push(matrix, weights):
    return np.sum(weights * (matrix @ weights), axis=0, dtype=np.float64)


def qualify_statistics():
    a = np.array([[.7, .2, -.1], [.2, .9, .3], [-.1, .3, .8]])
    b = np.array([[.6, .1, -.2], [.1, 1.1, .2], [-.2, .2, .7]])
    delta = np.array([[0., -.03, .01], [-.03, 0., -.02], [.01, -.02, 0.]])
    w = np.array([[.7, -.2], [.1, .5], [-.4, .8]])
    direct = np.array([sum(w[i, k] * a[i, j] * w[j, k]
                          for i in range(3) for j in range(3)) for k in range(2)])
    assert np.allclose(diagonal_push(a, w), direct, rtol=0, atol=2e-16)
    stats = cross_stats(a, b, delta)
    expected = sum((a[i, j] + delta[i, j]) * (b[i, j] + delta[i, j])
                   for i in range(3) for j in range(3))
    assert abs(stats['changed_cross'] - expected) < 1e-15
    null = cross_stats(a, b, np.zeros_like(a))
    assert null['base_cross'] == null['changed_cross']
    known = cross_stats(a, a, -a)
    assert abs(known['changed_cross']) < 1e-15
    return dict(passed=True, direct_scalar_congruence=True,
                direct_scalar_changed_cross=True, exact_null=True, known_error_removed=True)


assert sha(BANK) == '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert sha(FRAGMENT) == 'e1d8db8551ffe9fcfeea88b87d161e1788f7fb5ff1b092bdb2b40868bb6a97f0'
assert not (D / 'result.json').exists()
truth_paths = {rank: P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz' for rank in [35, 100]}
source_paths = [BANK, FRAGMENT, D / 'PROTOCOL.md', Path(__file__),
                P2 / 'fable_covhunter_v1/mc_allayers_mps.py', *truth_paths.values()]
pins = {str(p): sha(p) for p in source_paths}
once(D / 'pre_array_source_seal.json', dict(files=pins, covariance_arrays_read=False))
qualification = qualify_statistics()
once(D / 'statistics_qualification.json', qualification)
fragment_tree = ast.parse(FRAGMENT.read_text())
func = next(n for n in fragment_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q3c_delta')
offline = {'fnp': np}
exec(compile(ast.Module(body=[func], type_ignores=[]), '<frozen offline Q3C>', 'exec'), offline)
q3delta = offline['_q3c_delta']

stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('q3_covariance_alignment_bank')
mod.__file__ = str(BANK)
sys.modules[mod.__name__] = mod
exec(compile(BANK.read_text(), str(BANK), 'exec'), mod.__dict__)
oldgn, oldpush, olddiag, oldfinal = (mod._gn_dcov, mod._flms_cov_star,
                                   mod._flms_covariance_diagonal, mod.final_scalar_features)
current = {}
pending_delta = {}
pending_metric = {}
records = []


def finalize(layer, aux):
    if not pending_metric:
        return
    assert layer == pending_metric['layer'] + 1
    weight = np.asarray(aux['pdf'], np.float64) / (2 * np.asarray(aux['sigma'], np.float64))
    assert np.isfinite(weight).all() and (weight >= 0).all()
    stats = cross_stats(pending_metric.pop('va') * weight,
                        pending_metric.pop('vb') * weight,
                        pending_metric.pop('vd') * weight)
    pending_metric['weighted_next_variance'] = stats
    records.append(dict(pending_metric))
    pending_metric.clear()


def gn(layer, mu, cov, third, fourth, aux):
    finalize(layer, aux)
    value = oldgn(layer, mu, cov, third, fourth, aux)
    assert not pending_delta and layer in range(2, 15)
    pr = mod._E2_LAYER['pair']
    assert pr['aux'] is aux and aux['pre_cov'] is cov
    st = mod._e2_pair_state(int(mu.shape[0]))
    # Existing state lookup only; numerical reference below is ordinary NumPy.
    delta = q3delta(np.asarray(st['FT'])[1:6, 0], np.asarray(st['T21']),
                   np.asarray(st['t3']), np.asarray(pr['ss']))
    assert delta.shape == (1024, 1024) and np.isfinite(delta).all()
    assert np.array_equal(delta, delta.T) and np.count_nonzero(np.diag(delta)) == 0
    pending_delta.update(layer=layer, delta=delta)
    return value


def observe(cov, weights):
    if not pending_delta:
        return
    assert not pending_metric
    layer = pending_delta['layer']
    delta = pending_delta['delta']
    c = np.asarray(cov, np.float64)
    w = np.asarray(weights, np.float64)
    mc = current['mc']
    ca = np.asarray(mc[f'cov_post_A_{layer}'], np.float64) * current['half_scale'][0]
    cb = np.asarray(mc[f'cov_post_B_{layer}'], np.float64) * current['half_scale'][1]
    ea, eb = c - ca, c - cb
    va, vb, vd = (diagonal_push(ea, w), diagonal_push(eb, w), diagonal_push(delta, w))
    np.fill_diagonal(ea, 0)
    np.fill_diagonal(eb, 0)
    covstats = cross_stats(ea, eb, delta)
    pending_metric.update(rank=current['rank'], layer=layer,
                          offdiagonal_covariance=covstats, va=va, vb=vb, vd=vd,
                          delta_max_abs=float(np.max(np.abs(delta))),
                          delta_symmetric_exact=True, delta_diagonal_zero=True)
    pending_delta.clear()


def push(w, cov, star):
    observe(cov, w)
    return oldpush(w, cov, star)


def diag(w, cov):
    observe(cov, w)
    return olddiag(w, cov)


def final(layer, mean, aux, w):
    finalize(layer, aux)
    return oldfinal(layer, mean, aux, w)


mod._gn_dcov, mod._flms_cov_star = gn, push
mod._flms_covariance_diagonal, mod.final_scalar_features = diag, final
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
          api_version='2', scratch_dir=None, submission_dir=None, seed=0))
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
weight_rows = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
base_rows = {r['rank']: r for r in map(json.loads, (H / 'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
relief = ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
begin = time.monotonic()
network_rows = []
try:
    for rank in [35, 100]:
        assert not current and not pending_delta and not pending_metric
        wp = store / Path(weight_rows[rank]['path']).parent / 'weights.npy'
        weights = np.load(wp)
        mc = np.load(truth_paths[rank])
        population = int(mc['P'])
        counts = [0, 0]
        remaining = population
        i = 0
        while remaining:
            count = min(65536, remaining)
            counts[i % 2] += count
            remaining -= count
            i += 1
        current.update(rank=rank, mc=mc, half_scale=[n / (n - 1) for n in counts])
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                                   seed=0, name=f'phase2-{rank}')
        start = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            prediction = np.asarray(est.predict(mlp, 2**41), np.float64)
        assert not pending_delta and not pending_metric
        base = np.load(H / 'both_tail_train2_v6' / f'pred_{rank}.npy')
        bill = int(ctx.summary_dict()['flops_used'])
        assert prediction.shape == base.shape and prediction.dtype == base.dtype
        assert prediction.tobytes() == base.tobytes()
        assert bill == base_rows[rank]['flops'], (rank, bill, base_rows[rank]['flops'])
        with (D / f'pred_{rank}.npy').open('xb') as f:
            np.save(f, prediction)
        assert sha(D / f'pred_{rank}.npy') == sha(H / 'both_tail_train2_v6' / f'pred_{rank}.npy')
        mine = [r for r in records if r['rank'] == rank]
        assert [r['layer'] for r in mine] == list(range(2, 15))
        row = dict(rank=rank, flops=bill, all16_file_bytes_exact=True, complete_bill_exact=True,
                   half_counts=counts, weights_sha256=sha(wp), elapsed_s=time.monotonic() - start,
                   covariance_layers=13, final_mean_truth_read=False)
        for key in ['offdiagonal_covariance', 'weighted_next_variance']:
            base_energy = sum(r[key]['base_cross'] for r in mine)
            changed_energy = sum(r[key]['changed_cross'] for r in mine)
            row[key] = dict(base_cross=base_energy, changed_cross=changed_energy,
                ratio=changed_energy / base_energy if base_energy > 0 else None,
                change_pct=100 * (changed_energy / base_energy - 1) if base_energy > 0 else None,
                improved_layers=sum(r[key]['changed_cross'] < r[key]['base_cross'] for r in mine),
                all_base_energies_positive=all(r[key]['base_cross'] > 0 for r in mine))
        row['gates'] = dict(
            positive_cross_energies=all(row[k]['all_base_energies_positive'] for k in ['offdiagonal_covariance', 'weighted_next_variance']),
            covariance_reduction=row['offdiagonal_covariance']['ratio'] is not None and row['offdiagonal_covariance']['ratio'] <= .98,
            weighted_variance_reduction=row['weighted_next_variance']['ratio'] is not None and row['weighted_next_variance']['ratio'] <= .98,
            weighted_variance_layer_wins=row['weighted_next_variance']['improved_layers'] >= 10)
        network_rows.append(row)
        once(D / f'layers_{rank}.json', mine)
        once(D / f'network_{rank}.json', row)
        print(json.dumps(dict(event='network_complete', **row)), flush=True)
        mc.close()
        current.clear()
        del weights, mlp, mc, prediction, base
        gc.collect()
        relief(None, 0)
finally:
    mod._gn_dcov, mod._flms_cov_star = oldgn, oldpush
    mod._flms_covariance_diagonal, mod.final_scalar_features = olddiag, oldfinal

result = dict(networks=network_rows,
              passes_source_gate=all(all(row['gates'].values()) for row in network_rows),
              scope='Frozen S31 local covariance action at coefficient 1, never fed back into inference',
              covariance_reference='Independent even/odd-batch P32M halves, unbiased sample covariance denominators',
              limitations='Cross-half energy removes independent sample noise, not shared FP32/MPS numerical bias; reused TRAIN, two networks, no endpoint gain implied',
              source_hashes=pins, qualification=qualification,
              covariance_targets_accessed=True, final_mean_truth_read=False, valid40_accessed=False,
              fits_run=False, changed_candidate_built=False, board_submitted=False,
              original_failed_candidate_unchanged=True,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              elapsed_s=time.monotonic() - begin)
once(D / 'result.json', result)
assert sum(p.stat().st_size for p in D.glob('*') if p.is_file()) < 5_000_000
print(json.dumps(dict(event='complete', passes=result['passes_source_gate'],
                     elapsed_s=result['elapsed_s'], peak_rss_bytes=result['peak_rss_bytes'])), flush=True)
