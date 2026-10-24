"""Freeze one native version of the inherited physical-dial GN experiment.

Source/metadata only. No predictor import, arrays, targets, or numerical jobs.
"""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
ROOT = P2.parents[3]
PARENT = H2 / 'hill_climb_star_projection32_v1/candidate.py'
PARENT_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
BASE = dict(X1=.5565819405586057, X0=.024419009444819114,
            X2=.3928502699992539, X3=.15153807567151212,
            bscale=1.0383394097164294, s_carry=1.0032287195131746,
            s_new=.9932375877606482, c0=.5341766552087044,
            c1=.5026934507950427, k4a=.9830946248325227,
            k4b=.9659721622805341, k4c=.9595022755753744)
STEPS = dict(X1=.05, X0=.08, X2=.08, bscale=.02, s_carry=.005,
             s_new=.02, c0=.05, c1=.1, k4a=.1, k4b=.1, k4c=.1)
SCORE = [35, 100, 36, 45, 52, 58, 64, 70]


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def suffix(d):
    assert set(d) == set(BASE)
    assert d['X3'] == BASE['X3']
    return ('\n\n# Fixed native physical-dial calibration; no learned literal changes.\n'
            f"_BD_J1_PHYS = {d['X1']!r}\n"
            f"_S19_BD_ADD = {(d['X0'], 0.0, d['X2'], d['X3'])!r}\n"
            f"_S19_BSCALE = {d['bscale']!r}\n"
            f"_S19_LAM = {(d['s_carry'], d['s_new'])!r}\n"
            f"_S19_K4BAND = {(d['k4a'], d['k4b'], d['k4c'])!r}\n"
            f"_K31R_TABLE = {{l: {(d['c0'], d['c1'])!r} for l in range(1, 15)}}\n"
            'assert _S19_GN_TORCH is False\n')


def write_candidate(path, dials):
    raw = PARENT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    ending = suffix(dials).encode()
    result = raw + ending
    ast.parse(result)
    with path.open('xb') as f:
        f.write(result)
    assert path.read_bytes()[:-len(ending)] == raw
    return dict(**pin(path), dials=dials, appended_bytes=len(ending),
                exact_parent_prefix=True, recovered_parent_sha256=PARENT_SHA)


def main():
    inputs_path = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json'
    inputs = json.loads(inputs_path.read_text())
    fit = [r for r in inputs['train'] if r not in SCORE][:12]
    assert fit == [38, 40, 41, 43, 44, 46, 47, 49, 50, 51, 53, 54]
    assert not set(fit) & (set(SCORE) | set(inputs['valid']))
    provenance = json.loads((H2 / 'teammate_s32_physical_dials_v1.json').read_text())
    assert provenance['current_values'] == BASE
    variants = {}
    for name, step in STEPS.items():
        d = dict(BASE)
        d[name] += step
        variants[name] = dict(**write_candidate(HERE / f'perturb_{name}.py', d),
                              step=step, tag=f'hill_climb_physical11_fd_{name}_v1')
    protocol = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Small joint calibration of 11 active existing physical dials on actual S32 native arithmetic; inherited Opus forward-difference GN process.',
        prior_scope='Ten dials were already jointly trained in LoopR8s. c1 was frozen for weak identification. X3 is inert and excluded. Earlier five-native-GN-knob fit failed on different coordinates. No claim of positive headroom before measurement.',
        parent=pin(PARENT), base_dials=BASE, steps=STEPS, knobs=list(STEPS),
        variants=variants, baseline_tag='hill_climb_physical11_base_fit12_v1',
        fit_ranks=fit, selection='First 12 TRAIN160 ranks excluding fixed score8; no target-dependent selection.',
        train160_reused=True, pristine_holdout_claim=False,
        actual_score_ranks=SCORE, broad_ranks=inputs['valid'],
        fit_method=dict(derivative='Forward secant delta at inherited Opus step H; native F32-promoted final outputs.',
                        objective='Sum of baseline-utilization-weighted final squared residuals; fixed weights from baseline bill.',
                        normalized_coordinates='z=delta/H', bounds=[-2., 2.],
                        damping=[0., .01, .1, 1.], damping_matrix='diag(A.T@A) in normalized coordinates',
                        damping_selection='Lowest pooled leave-one-network-out predicted loss; ties choose larger damping.',
                        solver='scipy.optimize.lsq_linear augmented system; zero-response coordinates fixed at zero.',
                        no_step_size_sweep=True),
        fit_admission=dict(predicted_gain_pct_min=.25, loo_gain_pct_min=.25, loo_wins_min=8),
        actual_fit_verification=dict(scales=[1., .5], freeze_both_before_scoring=True,
                                     select='Lowest actual pooled adjusted ratio on the fixed FIT12; tie chooses half step.',
                                     gain_pct_min=.1, wins_min=8,
                                     no_other_scale_epoch_or_coordinate_selection=True),
        separate_score_gate=dict(train2_adjusted_pct_max=-.1, train2_wins_min=2,
                                 train8_adjusted_pct_max=-.1, train8_wins_min=5,
                                 train8_ci_upper_pct_max=0., train6_additional_gain_required=True,
                                 max_individual_regression_pct=5.),
        release='Only a fixed candidate or prospective exact-cost bundle with >=1% measured VALID40 adjusted gain, negative paired CI, >=28/40 wins, bounded regressions, and full serving qualification may be submitted. No automatic upload.',
        resource_limits=dict(seconds_each_native12=450, rss_bytes=7500000000,
                             added_swap_bytes=0, blas_threads=1, nice=10, serial=True),
        native_calls_acquisition=144,
        preparation_only=True, admission_required=True,
        sources=[pin(inputs_path), pin(P2/'fable_covhunter_v1/gn_joint_fit.py'),
                 pin(H2/'teammate_s32_physical_dials_v1.json'), pin(H2/'native_eval.py'),
                 pin(P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'),
                 pin(Path(__file__).resolve())])
    with (HERE/'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(protocol=pin(HERE/'protocol.json'), fit_ranks=fit,
                          variants=len(variants), model_jobs_run=False)))


if __name__ == '__main__':
    main()
