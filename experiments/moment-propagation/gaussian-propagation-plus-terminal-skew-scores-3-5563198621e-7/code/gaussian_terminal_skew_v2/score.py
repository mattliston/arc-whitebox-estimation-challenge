"""Judge the two fixed, sealed native acquisitions; never fit or alter them."""
import argparse
import json
from pathlib import Path
import traceback
import common as h

TRUTH_PINS = {
    35: 'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',
    100: '637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c',
}


def authenticate(role):
    folder = h.HERE / ('native_' + role)
    path, completion = folder / 'result.json', folder / 'completion.json'
    result = json.loads(path.read_text())
    seal = json.loads(completion.read_text())
    h.require(seal['passed'] and seal['completed'] and seal['result_sha256'] == h.sha(path),
              'Native acquisition completion seal')
    h.require(result['passed'] and result['completed'] and result['all_bindings_unchanged'],
              'Qualified native acquisition')
    h.require(result['role'] == role and result['sequence'] == [35, 100, 35], 'Fixed role/sequence')
    h.require(result['official_fnp_input'] and not result['input_preparation_in_participant_bill'],
              'Official input ownership')
    h.require(not result['truth_accessed'] and result['initialization_once'] and
              result['repeat_excluded_from_score'] and result['initialization_flops'] == 0,
              'Acquisition and initialization scope')
    h.verify(result['source_bindings'])
    h.verify(result['artifacts'])
    h.require(len(result['trials']) == 3, 'Three completed acquisition calls')
    for trial, rank in zip(result['trials'], (35, 100, 35)):
        h.guard(trial)
        h.require(trial['rank'] == rank and trial['error'] is None and trial['output_contract'] and
                  trial['served_materialized_byte_exact'], 'Qualified prediction')
        h.require(h.sha(trial['output']) == trial['output_sha256'], 'Sealed prediction bytes')
    first, repeat = result['trials'][0], result['trials'][2]
    h.require(first['flops'] == repeat['flops'] and
              first['output_sha256'] == repeat['output_sha256'], 'Exact repeat output/bill')
    return result, {str(path): h.sha(path), str(completion): h.sha(completion)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    roles, bindings = {}, {str(Path(__file__).resolve()): h.sha(__file__)}
    # Authenticate ALL predictions/bills before importing NumPy or opening truths.
    for role in ('gaussian', 'skew'):
        roles[role], pins = authenticate(role)
        bindings.update(pins)
    truths = {}
    for rank, expected in TRUTH_PINS.items():
        path = h.P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
        h.require(h.sha(path) == expected, 'Fixed development truth seal')
        truths[rank] = path
        bindings[str(path)] = expected
    out = h.HERE / 'score'
    out.mkdir()
    result = dict(passed=False, completed=False, official_score=False, independent14=False,
                  reference='Fixed development ranks35/100, existing1e9 truthpost mc_post means',
                  aggregation='Mean of per-net FINAL raw*max(.1,own complete F/2^41); repeat excluded',
                  source_bindings=bindings, roles={}, started_utc=h.utc(),
                  target_use='Judgment only after all native outputs/bills sealed; no refit')
    h.write(out / 'started.json', result)
    try:
        import numpy as np
        values = {}
        for rank, path in truths.items():
            with np.load(path, allow_pickle=False) as data:
                value = np.asarray(data['mc_post'], dtype=np.float64)
            h.require(value.shape == (16, 1024) and np.isfinite(value).all(), 'Truth shape/finite')
            values[rank] = value
        for role, acquisition in roles.items():
            rows = []
            for trial in acquisition['trials'][:2]:
                prediction = np.load(trial['output'], allow_pickle=False)
                h.require(prediction.shape == (16, 1024) and prediction.dtype == np.float32 and
                          np.isfinite(prediction).all() and (prediction >= 0).all(), 'Prediction contract')
                mse = np.mean((prediction.astype(np.float64) - values[trial['rank']])**2, axis=1)
                multiplier = max(.1, trial['flops'] / h.BUDGET)
                rows.append(dict(rank=trial['rank'], raw=float(mse[-1]), layer_mse=mse.tolist(),
                                 flops=trial['flops'], multiplier=multiplier,
                                 adjusted=float(mse[-1]) * multiplier,
                                 wall_s=trial['wall_s'], peak_rss_bytes=trial['peak_rss_bytes']))
            result['roles'][role] = dict(per_net=rows,
                mean_raw=float(np.mean([x['raw'] for x in rows])),
                mean_flops=float(np.mean([x['flops'] for x in rows])),
                mean_adjusted=float(np.mean([x['adjusted'] for x in rows])))
        candidate, control = result['roles']['skew'], result['roles']['gaussian']
        result['adjusted_ratio_vs_gaussian'] = candidate['mean_adjusted'] / control['mean_adjusted']
        result['prospective_both_net_thresholds'] = {
            str(threshold): all(x['adjusted'] < threshold for x in candidate['per_net'])
            for threshold in (4e-9, 3e-9, 1e-9)}
        result['advances_to_independent14'] = result['prospective_both_net_thresholds']['4e-09']
        h.verify(bindings)
        for acquisition in roles.values():
            h.verify(acquisition['artifacts'])
        result.update(passed=True, completed=True, hashes_unchanged=True,
                      milestone_bank_achieved=False, production_residual_validated=False)
    except BaseException:
        result['error'] = traceback.format_exc()
    h.finish(out, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
