"""One fixed native five-knob fit; root executes after all six panels retire."""
from pathlib import Path
import hashlib
import io
import json
import math

import numpy as np
from scipy.optimize import lsq_linear


HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
RANKS = [3, 12, 40, 41, 43, 44]
KNOBS = ['iso', 'k21n', 'k31', 'edge', 'rc']


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked_bytes(record):
    raw = Path(record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
    assert 'bytes' not in record or len(raw) == record['bytes'], record['path']
    return raw


def json_file(path):
    record = pin(path)
    return json.loads(checked_bytes(record)), record


def panel(tag, source, guard_path, expected_ranks):
    """Validate a complete native run before opening any prediction array."""
    checked_bytes(source)
    folder = H2 / tag
    manifest, manifest_pin = json_file(folder / 'manifest.json')
    guard, guard_pin = json_file(guard_path)
    assert manifest['candidate'] == source['path']
    assert manifest['sha256'] == source['sha256']
    assert manifest['ranks'] == expected_ranks
    assert manifest['numpy'] == '2.4.6'
    assert all(value == '1' for value in manifest['thread_env'].values())
    assert guard['completed'] is True and guard['exit_code'] == 0
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert guard['error'] is None and guard['max_swap_growth_bytes'] == 0
    assert guard['max_rss_bytes'] <= 7_500_000_000 and guard['elapsed_s'] <= 300
    assert guard['command'][-4:] == [str(H2 / 'native_eval.py'), source['path'],
                                   ','.join(map(str, expected_ranks)), tag]
    rows_pin = pin(folder / 'rows.jsonl')
    rows = [json.loads(line) for line in checked_bytes(rows_pin).decode().splitlines()]
    assert [row['rank'] for row in rows] == expected_ranks
    for row in rows:
        assert all(math.isfinite(row[key]) for key in ('mse', 'util', 'adjusted'))
        assert row['mse'] >= 0 and row['flops'] > 0
        assert row['util'] == row['flops'] / 2**41
        assert math.isclose(row['adjusted'], row['mse'] * max(.1, row['util']), rel_tol=1e-14)
    return dict(tag=tag, source=source, manifest=manifest_pin, rows=rows_pin,
                guard=guard_pin, selected_rows=rows[:6])


def solve(design, error, weights):
    """Normalized-row objective plus .01*MSE0*||z||², with fixed box bounds."""
    row_weights = weights / (weights.sum() * error.shape[1])
    mse0 = float(np.sum(row_weights[:, None] * error**2))
    assert np.isfinite(mse0) and mse0 > 0
    # Scale the entire objective by MSE0: identical optimum, stable tolerances.
    scale = np.sqrt(row_weights / mse0)
    matrix = (design * scale[:, None, None]).reshape(-1, len(KNOBS))
    rhs = -(error * scale[:, None]).reshape(-1)
    result = lsq_linear(np.vstack((matrix, np.sqrt(.01) * np.eye(len(KNOBS)))),
                        np.concatenate((rhs, np.zeros(len(KNOBS)))), bounds=(-2., 2.))
    assert result.success and np.isfinite(result.x).all(), result.message
    assert np.all(result.x >= -2.) and np.all(result.x <= 2.)
    return result.x, dict(baseline_weighted_mse=mse0, ridge_lambda=.01*mse0,
        objective_scale=mse0, status=int(result.status), message=result.message,
        iterations=int(result.nit), optimality=float(result.optimality),
        scaled_augmented_cost=float(result.cost), active_mask=result.active_mask.tolist())


def weighted_mse(error, weights):
    return float(np.dot(weights, np.mean(error**2, axis=1)) / weights.sum())


def main():
    assert not (HERE / 'fit_result.json').exists()
    assert not (HERE / 'candidate.py').exists()
    protocol, protocol_pin = json_file(HERE / 'protocol.json')
    assert protocol['fit_ranks'] == RANKS and protocol['knobs'] == KNOBS
    assert not set(RANKS) & set(protocol['actual_score_ranks'])
    assert protocol['base'] == dict(iso=-.094, k21n=-.028, k31=-.212, edge=.164, rc=-.08)
    assert protocol['steps'] == dict(iso=.05, k21n=.05, k31=.05, edge=.1, rc=.1)
    method = protocol['fit_method']
    assert method['ridge_fraction_of_baseline_weighted_mse'] == .01
    assert method['bounds_each_normalized_coordinate'] == [-2, 2] and method['one_fit_only']
    gate = protocol['fit_admission']
    assert gate == dict(predicted_pooled_weighted_gain_pct_min=.1,
        leave_one_network_out_pooled_gain_pct_min=.1, leave_one_network_out_wins_min=4)
    parent_bytes = checked_bytes(protocol['parent'])
    panels = [panel(protocol['baseline_tag'], protocol['parent'],
        H2 / 'hill_climb_star_projection32_v1/extra8_candidate_guard.json', RANKS+[57, 68])]
    for knob in KNOBS:
        source = protocol['variants'][knob]
        assert source['step'] == protocol['steps'][knob]
        expected = dict(protocol['base'])
        expected[knob] += protocol['steps'][knob]
        assert source['values'] == expected
        panels.append(panel('hill_climb_native_five_knobs_'+knob+'_v1', source,
                            HERE / (knob+'_guard.json'), RANKS))

    # All guard/source/roster checks above precede hashes and array decoding.
    # Hash each selected file now; decode only the bytes that match that hash.
    for item in panels:
        item['predictions'] = [pin(H2/item['tag']/f'pred_{rank}.npy') for rank in RANKS]
    truths = [pin(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') for rank in RANKS]
    script_pin = pin(__file__)
    evaluator_pin = pin(H2 / 'native_eval.py')
    predictions = []
    for item in panels:
        rows = []
        for record in item['predictions']:
            array = np.load(io.BytesIO(checked_bytes(record)), allow_pickle=False)
            assert array.shape == (16, 1024) and array.dtype == np.dtype('float64')
            final = array[15].copy()
            assert np.isfinite(final).all()
            rows.append(final)
        predictions.append(np.stack(rows))
    target_rows = []
    for record in truths:
        with np.load(io.BytesIO(checked_bytes(record)), allow_pickle=False) as packet:
            means = packet['mc_post']
            assert means.shape == (16, 1024) and means.dtype.kind == 'f'
            final = means[15].astype(np.float64)
            assert np.isfinite(final).all()
            target_rows.append(final)
    target = np.stack(target_rows)
    for item, prediction in zip(panels, predictions):
        measured = np.mean((prediction-target)**2, axis=1)
        recorded = np.array([row['mse'] for row in item['selected_rows']])
        assert np.allclose(measured, recorded, rtol=1e-12, atol=0), item['tag']

    error = predictions[0] - target
    design = np.stack([prediction-predictions[0] for prediction in predictions[1:]], axis=-1)
    weights = np.array([max(.1, row['util']) for row in panels[0]['selected_rows']])
    z, solver = solve(design, error, weights)
    linear_error = error + design @ z
    baseline_mse = weighted_mse(error, weights)
    predicted_mse = weighted_mse(linear_error, weights)
    loo_error = np.empty_like(error)
    loo = []
    for index, rank in enumerate(RANKS):
        keep = np.arange(len(RANKS)) != index
        fold_z, fold_solver = solve(design[keep], error[keep], weights[keep])
        loo_error[index] = error[index] + design[index] @ fold_z
        base = float(np.mean(error[index]**2))
        fitted = float(np.mean(loo_error[index]**2))
        loo.append(dict(rank=rank, z=fold_z.tolist(), baseline_mse=base,
            held_out_linear_mse=fitted, gain_pct=100*(1-fitted/base),
            win=bool(fitted < base), solver=fold_solver))
    loo_mse = weighted_mse(loo_error, weights)
    predicted_gain = 100*(1-predicted_mse/baseline_mse)
    loo_gain = 100*(1-loo_mse/baseline_mse)
    loo_wins = sum(row['win'] for row in loo)
    admitted = bool(predicted_gain >= .1 and loo_gain >= .1 and loo_wins >= 4)
    values = {knob: float(protocol['base'][knob]+protocol['steps'][knob]*z[i])
              for i, knob in enumerate(KNOBS)}
    candidate = None
    if admitted:
        suffix = ('\n# One fixed native five-knob ridge fit, normalized offsets bounded to [-2, 2].\n'
                  '_GN_KNOBS = '+repr(values)+'\n'
                  "_GN_EDGE = {l: _GN_KNOBS['edge'] for l in range(8, 15)}\n"
                  "_GN_RC = {l: _GN_KNOBS['rc'] for l in range(1, 15)}\n").encode()
        source = parent_bytes + suffix
        compile(source, str(HERE/'candidate.py'), 'exec')
        with (HERE/'candidate.py').open('xb') as output:
            output.write(source)
        candidate = pin(HERE/'candidate.py')
        assert checked_bytes(candidate) == parent_bytes + suffix
    result = dict(passed=True, admitted=admitted, ranks=RANKS, knobs=KNOBS,
        protocol=protocol_pin, fit_source=script_pin, native_evaluator=evaluator_pin,
        panels=panels, truths=truths, solver=solver, z=z.tolist(), values=values,
        baseline_network_weights=weights.tolist(),
        normalized_per_neuron_row_weights=(weights/(1024*weights.sum())).tolist(),
        baseline_weighted_mse=baseline_mse, predicted_weighted_mse=predicted_mse,
        predicted_pooled_weighted_gain_pct=predicted_gain,
        leave_one_network_out_pooled_gain_pct=loo_gain,
        leave_one_network_out_wins=loo_wins, leave_one_network_out=loo,
        fit_admission=gate, candidate=candidate, full_data_fits=1, diagnostic_loo_fits=6,
        predictions_are_linear=True, actual_candidate_score_evaluated=False,
        target_ranks_read=RANKS, no_sweep=True)
    with (HERE/'fit_result.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'admitted',
        'predicted_pooled_weighted_gain_pct', 'leave_one_network_out_pooled_gain_pct',
        'leave_one_network_out_wins', 'values', 'candidate')}))


if __name__ == '__main__':
    main()
