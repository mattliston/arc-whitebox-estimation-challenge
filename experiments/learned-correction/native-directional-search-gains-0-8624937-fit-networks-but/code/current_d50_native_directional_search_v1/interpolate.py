"""One bounded native-objective parabola proposal; no check-target access."""
import argparse
import importlib.util
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE = ('head', 'head_b', 'gain')
H = 5e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in ('admission', 'qualification-result', 'baseline-result', 'proposal-result',
                 'plus-result', 'minus-result'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    if any(not isinstance(v, str) or len(v) != 64 for k, v in vars(args).items() if k != 'run'):
        parser.error('Every fixed reviewed admission and prerequisite digest is required')
    spec = importlib.util.spec_from_file_location('nd_common_interpolate', HERE / 'common.py')
    c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c
    spec.loader.exec_module(c)
    s = c.Stage(__file__, args.admission_sha256, 'interpolate', 'interpolation_v1')
    try:
        s.environment('AP')
        q = s.receipt('qualification_v1', args.qualification_result_sha256, 'qualify')
        p = s.receipt('proposal_v1', args.proposal_result_sha256, 'propose')
        basic = dict(qualification_result_sha256=args.qualification_result_sha256)
        both = basic | dict(baseline_result_sha256=args.baseline_result_sha256)
        direction = both | dict(proposal_result_sha256=args.proposal_result_sha256)
        c.require(p['prerequisites'] == both and p['parameter_updates'] == p['adam_steps'] == 1
            and p['curve_h'] == H and p['curve_endpoint_byte_exact'] is True
            and p['frozen_payload_fields_byte_exact'] is True and p['proposal_sealed'] is True
            and p['surrogate_only'] is True and p['no_native_step_selection'] is True
            and p['check_targets_accessed'] is False and p['validation_targets_accessed'] is False,
            'One fixed qualified CPU-function proposal, not a native derivative')
        native = dict(
            baseline=c.native_prerequisite(s, 'native_baseline_v1', args.baseline_result_sha256,
                'baseline', q['payloads']['baseline'], basic),
            plus=c.native_prerequisite(s, 'native_plus_v1', args.plus_result_sha256,
                'plus', p['payloads']['plus'], direction),
            minus=c.native_prerequisite(s, 'native_minus_v1', args.minus_result_sha256,
                'minus', p['payloads']['minus'], direction))
        s.record.update(prerequisites=direction | dict(plus_result_sha256=args.plus_result_sha256,
            minus_result_sha256=args.minus_result_sha256), candidate_sealed=False,
            native_score_measured=True, interpolation_is_not_descent_proof=True, curve_h=H,
            payloads={}, fit_ranks=list(c.FIT), check_ranks=list(c.CHECK), parameter_updates=0)
        import numpy as np
        c.require(np.__version__ == '1.26.4' and 'torch' not in sys.modules, 'Pinned CPU score-only environment')
        s.record['versions'] = dict(numpy=np.__version__)
        baseline = s.load_bundle(np, p['payloads']['baseline'])
        qualified = s.load_bundle(np, q['payloads']['baseline'])
        plus = s.load_bundle(np, p['payloads']['plus'])
        minus = s.load_bundle(np, p['payloads']['minus'])
        difference = s.load_bundle(np, p['difference64'])

        def equal(a, b):
            return set(a) == set(b) and all(c.array_meta(a[k]) == c.array_meta(b[k]) for k in a)

        c.require(equal(baseline, qualified) and set(baseline) == set(plus) == set(minus) == set(difference),
                  'Exact qualified baseline and full curve key roster')
        c.require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in baseline.values())
            and all(difference[k].dtype == np.float64 and difference[k].shape == baseline[k].shape
                    and np.array_equal(difference[k], plus[k].astype(np.float64) - baseline[k].astype(np.float64))
                    for k in baseline), 'Exact frozen FP64 endpoint difference')
        c.require(all((difference[k] == 0).all() for k in baseline if k not in PKEYS)
            and all((difference[k][0] == 0).all() for k in INACTIVE), 'Frozen/inactive curve coordinates')

        def curve(t):
            result = {k: (a.astype(np.float64) + (t / H) * difference[k]).astype(np.float32)
                      if k in PKEYS else a.copy() for k, a in baseline.items()}
            for key in INACTIVE:
                result[key][0] = baseline[key][0]
            return result

        c.require(equal(curve(0.), baseline) and equal(curve(H), plus) and equal(curve(-H), minus),
                  'Exact declared curve through all three native-scored payloads')
        # All payloads and actual native fitting predictions/bills were admitted
        # before this first numerical truth access.
        target = c.truths(np, s, c.FIT)
        scores = {name: c.score(np, s, record, target, c.FIT) for name, record in native.items()}
        s.record['scores'] = scores
        j0, jm, jp = (scores[k]['mean_adjusted'] for k in ('baseline', 'minus', 'plus'))
        c.require(j0 > 0 and all(r['adjusted'] > 0 for r in scores['baseline']['per_net']),
                  'Positive baseline score denominators')
        curvature = jm + jp - 2 * j0
        s.record['curvature_denominator'] = curvature
        s.attempt('native_three_point_scores.json', dict(scores=scores, curvature_denominator=curvature))
        if not np.isfinite(curvature) or curvature <= 0:
            s.record['decision'] = 'STOP_nonpositive_or_nonfinite_curvature'
            s.finish(passed=False, scientific=True)
        else:
            step = float(np.clip(H * (jm - jp) / (2 * curvature), -H, H))
            c.require(np.isfinite(step) and -H <= step <= H, 'Bounded finite fixed parabolic proposal')
            candidate = curve(step)
            c.require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in candidate.values()),
                      'Finite exact candidate payload')
            s.record.update(tstar=step, step_fraction=step / H,
                parabola_predicted_adjusted=j0 + (jp - jm) * step / (2 * H)
                    + curvature * step**2 / (2 * H**2),
                candidate_equals_plus=equal(candidate, plus), candidate_equals_minus=equal(candidate, minus))
            if equal(candidate, baseline):
                s.record['decision'] = 'STOP_candidate_byte_identical_to_baseline'
                s.finish(passed=False, scientific=True)
            else:
                s.record['payloads']['candidate'] = s.bundle(np, 'payload_candidate.npz', candidate)
                s.record.update(candidate_sealed=True, decision='ACQUIRE_fixed_native_candidate',
                                frozen_payload_fields_byte_exact=True)
                s.attempt('candidate_seal.json', dict(payloads=s.record['payloads'], tstar=step,
                    curve_h=H, prerequisites=s.record['prerequisites'],
                    check_targets_accessed=False, validation_targets_accessed=False,
                    native_candidate_score_measured=False))
                s.finish(passed=True, scientific=False)
    except BaseException:
        s.record.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        return s.close()


if __name__ == '__main__':
    raise SystemExit(main())
