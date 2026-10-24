"""Direct native-score test of one sealed proposal; never post-check tuning."""
import argparse
import importlib.util
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in ('admission', 'qualification-result', 'baseline-result', 'proposal-result',
                 'plus-result', 'minus-result', 'interpolation-result', 'candidate-result'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    if any(not isinstance(v, str) or len(v) != 64 for k, v in vars(args).items() if k != 'run'):
        parser.error('All reviewed stage digests are required')
    spec = importlib.util.spec_from_file_location('nd_common_judge', HERE / 'common.py')
    c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c
    spec.loader.exec_module(c)
    s = c.Stage(__file__, args.admission_sha256, 'judge', 'judgment_v1')
    try:
        s.environment('AP')
        q = s.receipt('qualification_v1', args.qualification_result_sha256, 'qualify')
        p = s.receipt('proposal_v1', args.proposal_result_sha256, 'propose')
        inter = s.receipt('interpolation_v1', args.interpolation_result_sha256, 'interpolate')
        basic = dict(qualification_result_sha256=args.qualification_result_sha256)
        both = basic | dict(baseline_result_sha256=args.baseline_result_sha256)
        direction = both | dict(proposal_result_sha256=args.proposal_result_sha256)
        selected = direction | dict(plus_result_sha256=args.plus_result_sha256,
                                    minus_result_sha256=args.minus_result_sha256)
        c.require(p['prerequisites'] == both and inter['prerequisites'] == selected
            and inter['candidate_sealed'] is True and inter['curve_h'] == 5e-4
            and inter['check_targets_accessed'] is False and inter['validation_targets_accessed'] is False,
            'Candidate sealed before any proposal-check target access')
        baseline = c.native_prerequisite(s, 'native_baseline_v1', args.baseline_result_sha256,
                                        'baseline', q['payloads']['baseline'], basic)
        for case in ('plus', 'minus'):
            c.native_prerequisite(s, 'native_' + case + '_v1', getattr(args, case + '_result_sha256'),
                                  case, p['payloads'][case], direction)
        candidate = c.native_prerequisite(s, 'native_candidate_v1', args.candidate_result_sha256,
            'candidate', inter['payloads']['candidate'], direction | dict(interpolation_result_sha256=args.interpolation_result_sha256))
        s.record.update(prerequisites=selected | dict(interpolation_result_sha256=args.interpolation_result_sha256,
            candidate_result_sha256=args.candidate_result_sha256), candidate_payload=inter['payloads']['candidate'],
            fit_ranks=list(c.FIT), check_ranks=list(c.CHECK), native_score_measured=True,
            validation_score_measured=False, independent14_score_measured=False, board_qualified=False,
            parameter_updates=0, no_post_check_tuning=True, target_scope='fixed TRAIN8 only')
        import numpy as np
        c.require(np.__version__ == '1.26.4' and 'torch' not in sys.modules, 'Pinned native-score-only environment')
        s.record['versions'] = dict(numpy=np.__version__)
        # This is the first check-target decoder: all nine candidate calls,
        # their complete bills, repeat/domain/source checks and seal precede it.
        targets = c.truths(np, s, c.FIT + c.CHECK)
        comparisons = {}
        for name, ranks in (('fit', c.FIT), ('check', c.CHECK)):
            old = c.score(np, s, baseline, targets, ranks)
            new = c.score(np, s, candidate, targets, ranks)
            if name == 'fit':
                c.require(old == inter['scores']['baseline'], 'Exact replay of fit native baseline objective')
            c.require(old['mean_adjusted'] > 0 and all(a['adjusted'] > 0 for a in old['per_net']),
                      'Positive baseline mean and per-net denominators')
            ratios = [b['adjusted'] / a['adjusted'] for a, b in zip(old['per_net'], new['per_net'])]
            c.require(all(np.isfinite(x) and x >= 0 for x in ratios), 'Nonnegative finite paired individual scores')
            wins = sum(b['adjusted'] < a['adjusted'] for a, b in zip(old['per_net'], new['per_net']))
            ratio = new['mean_adjusted'] / old['mean_adjusted']
            comparisons[name] = dict(baseline=old, candidate=new, mean_adjusted_ratio=ratio,
                individual_adjusted_ratios=ratios, wins=int(wins), passed=bool(ratio <= .98 and wins >= 3),
                mean_raw_ratio=new['mean_raw_final_mse'] / old['mean_raw_final_mse'],
                mean_complete_flops_ratio=new['mean_complete_flops'] / old['mean_complete_flops'])
        passed = all(row['passed'] for row in comparisons.values())
        s.record.update(comparisons=comparisons, advancement_gate_passed=passed,
            numerical_gates_passed=True, scientific_verdict_only=True,
            actual_native_candidate_verified=True, next_permission='consider separate larger evaluation' if passed
                else 'no advance or tuning; preserve receipts and any smaller component',
            decision='PASS_fixed_native_step' if passed else 'FAIL_fixed_advancement_gate')
        s.attempt('paired_native_verdict.json', dict(comparisons=comparisons,
            advancement_gate_passed=passed, numerical_gates_passed=True,
            independent14_score_measured=False, board_qualified=False))
        s.finish(passed=passed, scientific=True)
    except BaseException:
        s.record.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        return s.close()


if __name__ == '__main__':
    raise SystemExit(main())
