"""Root-only assessment of the disclosed fixed-six extension; no model or targets.

CLI: assess_extension.py 8, under NATIVE BLAS1 / 120s / 7.5GB / zero-swap guard.
The failed TRAIN2 receipt remains failed. This script writes a distinct result.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

D = Path(__file__).resolve().parent
BASE_SHA = '530a750f819b6e49e0c4e6901ab56b7d1af7b6cce666d07a602926aa292bd4a5'
EXTENSION_SHA = '1613ec5ad4e06678164b5b80b8924ae8f8a9c53fdc805ba99a4bedef4fd0d404'


def main():
    assert sys.argv[1:] == ['8']
    base_path = D/'assess.py'
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location('frozen_precision_assessor', base_path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)  # Pinned helpers only; its main is not invoked.
    assert Path(sys.executable).absolute() == q.NATIVE and q.np.__version__ == '2.4.6'
    assert all(os.environ.get(name) == '1' for name in q.THREADS)
    target = D/'native8_extension_assessment.json'
    assert not target.exists()
    extension_pin = q.pin(D/'extension_protocol.json')
    assert extension_pin['sha256'] == EXTENSION_SHA
    extension = q.load(D/'extension_protocol.json')
    own_pin = q.pin(Path(__file__))
    q.checked(own_pin)
    for key in ('candidate', 'parent', 'original_protocol', 'original_assessor',
                'prior_assessment', 'prior_assessment_guard', 'prior_native_guard',
                'build', 'root_admission'):
        q.checked(extension[key])
    assert extension['train8_gates'] == dict(adjusted_pct_at_most=-1.0, minimum_wins=6,
                                           raw_pct_at_most=.25, ci_upper_less_than=0.0)
    assert extension['bootstrap'] == dict(resamples=50000, seed=20260926)
    assert extension['additional_ranks'] == list(q.BLOCKS[1])
    assert extension['combined_ranks'] == list(q.BLOCKS[0] + q.BLOCKS[1])
    assert extension['original_assessor'] == q.pin(base_path)
    child = q.build_check()
    assert extension['candidate'] == child
    assert extension['original_protocol'] == q.pin(D/'PROTOCOL.md')
    admission = q.load(extension['root_admission']['path'])
    for key, value in (('candidate', child), ('original_protocol', extension['original_protocol']),
                       ('failed_native2', extension['prior_assessment']),
                       ('original_assessor', extension['original_assessor']),
                       ('assessor_guard', extension['prior_assessment_guard'])):
        assert admission[key] == {field: value[field] for field in ('path', 'sha256')}
    assert admission['next_ranks'] == list(q.BLOCKS[1])
    assert admission['next_tag'] == extension['native_driver_tag']
    assert admission['resources'] == dict(seconds=300, rss_limit=7500000000,
                                           max_added_swap=0, whole_group_retired=True)
    assert admission['train8_gates'] == dict(adjusted_gain_at_least_pct=1, wins_at_least=6,
        raw_regression_at_most_pct=.25, bootstrap_upper_less_than=0, resamples=50000, seed=20260926)
    assert admission['source_changes'] is False and admission['new_variants'] is False
    assert admission['original_failed_receipt_preserved'] is True
    assert admission['extension_outcomes_seen'] is False
    prior = q.load(extension['prior_assessment']['path'])
    assert prior['passed_integrity'] and prior['admitted'] is False and prior['n'] == 2
    assert prior['candidate'] == child and prior['ranks'] == list(q.BLOCKS[0])
    assert prior['protocol'] == extension['original_protocol']
    assert prior['assessor'] == extension['original_assessor']
    assert prior['gates'] == dict(adjusted_gain=False, wins=True, raw_regression=True)
    assert extension['prior_observations'] == {key: prior[key] for key in extension['prior_observations']}
    for record in prior['evidence']:
        q.checked(record)
    q.guard(D/'native2_assess_guard.json', 120,
            [str(q.NATIVE), '-B', str(base_path), '2'])
    base, current, peaks = [], [], []
    for size in (2, 6):
        for flag, output in ((False, base), (True, current)):
            rows, peak = q.panel(size, flag, child)
            output.extend(rows)
            peaks.append(peak)
    b, c = ({row['rank']: row for row in rows} for rows in (base, current))
    ranks = sorted(b)
    assert ranks == sorted(c) == sorted(extension['combined_ranks']) and len(ranks) == 8
    delta = lambda key: 100*(sum(c[r][key] for r in ranks)/sum(b[r][key] for r in ranks)-1)
    adjusted, raw, bill = delta('adjusted'), delta('mse'), delta('flops')
    wins = sum(c[r]['adjusted'] < b[r]['adjusted'] for r in ranks)
    bb = q.np.array([b[r]['adjusted'] for r in ranks])
    cc = q.np.array([c[r]['adjusted'] for r in ranks])
    idx = q.np.random.default_rng(20260926).integers(0, 8, size=(50000, 8))
    ci = (100*q.np.quantile(cc[idx].mean(1)/bb[idx].mean(1)-1, [.025, .975])).tolist()
    gates = dict(adjusted_gain=adjusted <= -1., wins=wins >= 6,
                 raw_regression=raw <= .25, ci_upper_negative=ci[1] < 0)
    result = dict(passed_integrity=True, admitted=all(gates.values()), n=8, ranks=ranks,
        candidate=child, protocol=q.pin(D/'PROTOCOL.md'), extension_protocol=extension_pin,
        root_admission=extension['root_admission'], assessor=own_pin,
        original_assessor=extension['original_assessor'], prior_native2=extension['prior_assessment'],
        original_native2_admitted=False, selection_deviation=extension['selection_deviation'],
        adjusted_pct=adjusted, raw_pct=raw, bill_pct=bill, adjusted_wins=wins,
        adjusted_ci95_pct=ci, bootstrap=dict(resamples=50000, seed=20260926), gates=gates,
        maximum_guard_rss_bytes=max(peaks), evidence=list(q.EVIDENCE.values()),
        rows=[dict(rank=r, adjusted_pct=100*(c[r]['adjusted']/b[r]['adjusted']-1),
                   raw_pct=100*(c[r]['mse']/b[r]['mse']-1),
                   bill_pct=100*(c[r]['flops']/b[r]['flops']-1)) for r in ranks],
        targets_read=False, model_executed=False, serving_qualified=False, submitted=False,
        bank_qualified=False, next_stage='Original VALID40 only if admitted; full serving still required.')
    for record in list(q.EVIDENCE.values()):
        q.checked(record)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('admitted', 'n', 'adjusted_pct', 'raw_pct',
        'bill_pct', 'adjusted_wins', 'adjusted_ci95_pct', 'gates')}))


if __name__ == '__main__':
    main()
