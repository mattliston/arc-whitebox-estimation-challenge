"""Read-only matched-cold pair analysis, executable only after both retire."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    directory = Path(__file__).resolve().parent
    plan = json.loads((directory / 'PLAN_v1.json').read_text())
    expected = plan['candidate_sha256']
    complete = json.loads((args.input / 'COMPLETE_v1.json').read_text())
    assert complete == dict(ranks=[35, 200], candidate_sha256=expected,
                            all_groups_retired=True)
    O = directory.parents[4]
    baseline = 1.4468565569762618e-9
    records, guards = [], {}

    def row(path, rank):
        rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
        hits = [value for value in rows if value.get('rank') == rank]
        assert len(hits) == 1
        return hits[0]

    def predictions(path, rank):
        values = np.fromfile(path / f'pred_{rank}.bin', dtype='<f4')
        assert values.size == 16 * 1024 and np.isfinite(values).all()
        return values.reshape(16, 1024).astype('d')

    for rank in (35, 200):
        candidate = args.input / f'rank{rank}'
        control = Path(plan['controls'][str(rank)])
        native_plan = json.loads((candidate / 'PLAN_v1.json').read_text())
        assert native_plan['sha256'] == expected
        for values in native_plan['runtime'].values():
            assert values['version'].split('+')[0] == '0.12.3'
        guard = json.loads((candidate / 'GUARD_v1.json').read_text())
        assert guard['completed'] and guard['whole_group_retired'] and not guard['remaining_rows']
        assert guard['max_rss_bytes'] < 7_500_000_000
        assert guard['max_footprint_bytes'] < 7_500_000_000
        assert guard['min_pressure_free_pct'] >= 50
        guards[str(rank)] = {key: guard.get(key) for key in (
            'completed', 'whole_group_retired', 'max_rss_bytes',
            'max_footprint_bytes', 'min_pressure_free_pct', 'exit_code')}
        a, b = row(candidate, rank), row(control, rank)
        p, c = predictions(candidate, rank), predictions(control, rank)
        truths = []
        for tag in ('A', 'B'):
            path = O / 'analyst/fresh/pu' / f'mps_truth_rank{rank:04d}_{tag}32M.npz'
            with np.load(path, allow_pickle=False) as data:
                truth = data['mc_post'].astype('d')
            assert truth.shape == (16, 1024) and np.isfinite(truth).all()
            truths.append(truth)
        u, v = truths
        # Independent A/B cross-residual removes the MC target-noise floor.
        candidate_nc = np.mean((p - u) * (p - v), axis=1)
        control_nc = np.mean((c - u) * (c - v), axis=1)
        ordinary_candidate = .5 * np.mean((p - u) ** 2 + (p - v) ** 2, axis=1)
        ordinary_control = .5 * np.mean((c - u) ** 2 + (c - v) ** 2, axis=1)
        assert candidate_nc[-1] > 0 and control_nc[-1] > 0
        raw_ratio = float(candidate_nc[-1] / control_nc[-1])
        bill_ratio = a['flops'] / b['flops']
        adjusted_ratio = raw_ratio * bill_ratio
        rescue = a.get('fast_rescue_stats') or {}
        assert a['nonfinite'] == 0 and rescue.get('calls', 0) == 0
        records.append(dict(
            rank=rank, control=str(control), raw_noise_corrected_pct=100 * (raw_ratio - 1),
            raw_ordinary_mc_pct=100 * (ordinary_candidate[-1] / ordinary_control[-1] - 1),
            candidate_noise_corrected_mse=float(candidate_nc[-1]),
            control_noise_corrected_mse=float(control_nc[-1]),
            bill_pct=100 * (bill_ratio - 1), adjusted_pct=100 * (adjusted_ratio - 1),
            conditional_absolute_score=baseline * adjusted_ratio,
            flops=a['flops'], control_flops=b['flops'], extra_flops=a['flops'] - b['flops'],
            requests=a['requests'], control_requests=b['requests'],
            requests_delta=a['requests'] - b['requests'],
            wall_s=a['wall_s'], control_wall_s=b['wall_s'],
            residual_instructions=a['residual_instructions'],
            control_residual_instructions=b['residual_instructions'],
            prediction_rms=float(np.sqrt(np.mean((p - c) ** 2))),
            final_prediction_rms=float(np.sqrt(np.mean((p[-1] - c[-1]) ** 2))),
            layer_noise_corrected_raw_pct=(100 * (candidate_nc / control_nc - 1)).tolist(),
            rescue=rescue, nonfinite=a['nonfinite'],
            cpcenter_state=a.get('oct18_cpcenter_state'),
            strip_state=a.get('oct18_margin_state'),
            candidate_prediction_sha256=hashlib.sha256((candidate / f'pred_{rank}.bin').read_bytes()).hexdigest()))
    mean = {key: statistics.mean(value[key] for value in records) for key in (
        'raw_noise_corrected_pct', 'raw_ordinary_mc_pct', 'bill_pct',
        'adjusted_pct', 'requests_delta')}
    geo = 100 * (math.exp(statistics.mean(math.log1p(value['adjusted_pct'] / 100)
                                        for value in records)) - 1)
    result = dict(status='FIXED_TWO_MATCHED_COLD_DIAGNOSTIC',
                  candidate_sha256=expected, baseline_submission=336935,
                  baseline_adjusted=baseline, rows=records, mean=mean,
                  geometric_adjusted_pct=geo,
                  conditional_projected_adjusted=baseline * (1 + mean['adjusted_pct'] / 100),
                  guards=guards, controls=plan['controls'],
                  caveats=['Both ranks use current95c fresh/cold controls; no cold200-versus-warm bias.',
                           'Raw means error before utilization; adjusted uses noise-corrected MSE times billed FLOPs.',
                           'Two public networks are an integration screen, not population confirmation or runtime qualification.',
                           'Current candidate includes fixedL1-3 CP-affine-centering arithmetic and request costs.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(mean=mean, geometric_adjusted_pct=geo,
                          conditional_projected_adjusted=result['conditional_projected_adjusted'],
                          guards=guards)))


if __name__ == '__main__':
    main()
