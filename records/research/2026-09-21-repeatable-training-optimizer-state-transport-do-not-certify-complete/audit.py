"""One-time, stdlib-only review of existing Phase 2 result metadata."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = OUT.parent
A = D.parent / 'astra_phase2_reexamination_v1'
Q = D / 'current_p_signed_branch_qualification_v2'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def record(path, expected):
    identity = pin(path)
    assert identity['sha256'] == expected, str(path)
    return json.loads(path.read_text()), identity


def exclusive_json(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not (OUT / 'audit.json').exists()
    assert not (OUT / 'RESULT.md').exists()
    bindings = [
        ('current_torch', Q / 'torch_qualification.json', 'fbc330548efca383fce61763f5c6e894b172ae1dc2c619e86e4e3f75460e70b8'),
        ('current_encoder', Q / 'unit_torch_result.json', 'e4dec606010bf701b62e693bd30a660b247ce8d05aa1b834704e13a907419dce'),
        ('cold_search', A / 'current_d50_native_cold_search_v1/judgment_v1/result.json', '026fb7b07f5f78aa14003760969d19887bdcf4db72a0f193773a29661e7f4bf8'),
        ('cold_watch', A / 'current_d50_native_cold_search_v1/watch_v1/result.json', 'c46a92b56ec2ef2683458bdb61bc5b9947873581bc7c723385cd16bdf4d1be52'),
        ('trajectory', A / 'current_native_trajectory_adjoint_v1/native_v1/result.json', '5443f413c7a740c49b62ec0b84d99af1ebd58db4b1f0a2439a00855747f4bb1b'),
        ('trajectory_watch', A / 'current_native_trajectory_adjoint_v1/native_watch_v1/result.json', '754aea4d14bfca3fbb61f519b973ab81a5cdb96fd7ca3c709de70d26ba6847ae'),
        ('rangefinder_vjp', A / 'current_shared384_causal_p_v1/rangefinder_vjp_v1/result.json', '55eec7f39a7c17af1e65c89b72d22b38b22bb40508db9e2acc6297ff2610e805'),
        ('native_y_full_forward', A / 'current_shared384_causal_p_v1/cpu_native_y_forward_v1/result.json', 'cb7734bea7845ded9a887d0f92a76483390174b1747bed1bafe588fd597aea83'),
        ('covariance_gap', A / 'current_covariance_gradient_gap_v1/stage_judge_v1/result.json', 'dd55930107acaa58b4dd37a4de409d7a73ea956f6ff9bdce16594f7cc4dd8533'),
    ]
    records, pins = {}, {}
    for name, path, expected in bindings:
        records[name], pins[name] = record(path, expected)
    t, u = records['current_torch'], records['current_encoder']
    assert t['passed'] and t['completed'] and t['repeat_gradient_bytes_exact']
    assert t['native_derivative_certified'] is False and not t['score_labels_accessed']
    assert u['passed'] and u['gradient_scope'] == 'small synthetic encoder only'
    assert u['native_derivative_certified'] is False and u['optimizer_updates'] == 0
    c, cw = records['cold_search'], records['cold_watch']
    assert c['completed'] and not c['passed'] and c['actual_native_candidate_verified']
    assert c['decision'] == cw['decision'] == 'FAIL_fixed_advancement_gate'
    assert c['native_score_measured'] and not c['native_gradient_qualified']
    assert cw['completed'] and not cw['passed'] and cw['scientific_stop']
    comparisons = {}
    for role, wins, ratio in [('fit', 2, .9913750625804131), ('check', 0, 1.020150121188754)]:
        row = c['comparisons'][role]
        assert row['wins'] == wins and row['mean_adjusted_ratio'] == ratio and not row['passed']
        comparisons[role] = dict(
            wins=wins, mean_adjusted_ratio=ratio,
            mean_raw_ratio=row['mean_raw_ratio'], mean_complete_flops_ratio=row['mean_complete_flops_ratio'],
            baseline_mean_adjusted=row['baseline']['mean_adjusted'],
            candidate_mean_adjusted=row['candidate']['mean_adjusted'],
            ranks=row['baseline']['ranks'], passed=False,
        )
    n, nw = records['trajectory'], records['trajectory_watch']
    assert n['completed'] and n['passed'] and n['outcome'] == 'native_primal_and_branch_qualified'
    assert n['partial_model_calls'] == 3 and n['full_model_calls'] == 0
    assert n['native_gradient_certified'] is False and n['secant_qualified'] is False
    assert nw['completed'] and nw['passed'] and nw['whole_owned_tree_retired']
    v = records['rangefinder_vjp']
    assert v['completed'] and v['passed'] and v['rangefinder_real_algebra_VJP_qualified']
    assert v['unit_cases'] == 3 and v['full_model_calls'] == 0 and not v['native_gradient_certified']
    y = records['native_y_full_forward']
    assert y['completed'] and not y['passed'] and not y['fidelity_passed']
    assert y['full_model_calls'] == 1 and not y['native_gradient_certified']
    assert y['fidelity']['max_abs'] == 3.6835670471191406e-5
    assert y['fidelity']['max_abs_limit'] == 1e-5
    g = records['covariance_gap']
    assert g['completed'] and g['passed'] and g['scientific_conclusion'] == 'numerically_inconclusive'
    assert not g['gradient_numerical_qualified'] and not g['information_gate_passed']
    assert g['no_fit'] and g['no_native_candidate'] and not g['complete_score_measured']
    documents = [Q / 'PROTOCOL.md', A / 'current_d50_native_cold_search_v1/RESULT.md',
        A / 'current_native_trajectory_adjoint_v1/NATIVE_PRIMAL_POSITIVE_SECANT_INCONCLUSIVE_20260913_UTC.md',
        A / 'current_shared384_causal_p_v1/RANGEFINDER_VJP_POSITIVE_20260913_UTC.md',
        A / 'current_d50_native_objective_p1_v1/DECISION_AND_ENDPOINT.md']
    result = dict(
        completed=True, metadata_assertions_passed=True, utc=datetime.now(timezone.utc).isoformat(),
        scope='Bounded review of the nine named Phase 2 result receipts and five context documents; not an exhaustive history or novelty claim.',
        primary_receipts=pins, context_documents=[pin(p) for p in documents], auditor=pin(Path(__file__)),
        current_encoder_scope=u['gradient_scope'], current_full_torch_repeatable=True,
        current_native_derivative_certified=False, cold_search_comparisons=comparisons,
        cold_search_completed_scientific_failure=True,
        trajectory_native_primal_qualified=True, trajectory_secant_qualified=False,
        trajectory_secant_values_recomputed_by_this_audit=False,
        rangefinder_component_real_algebra_vjp_qualified=True,
        native_y_full_forward_fidelity_passed=False,
        native_y_full_forward_max_abs=y['fidelity']['max_abs'],
        native_y_full_forward_max_abs_limit=y['fidelity']['max_abs_limit'],
        covariance_gap_scientific_conclusion=g['scientific_conclusion'],
        numerical_model_calls=0, parameter_updates=0, numerical_libraries_imported=False,
        target_or_prediction_arrays_opened=False, existing_scalar_score_metadata_read=True,
        new_candidate_created=False, failed_study_restarted=False,
        active_fit_or_validation_sources_modified=False, new_gate_or_derivative_prerequisite_added=False,
        decision='Preserve each result at its actual scope. Continue the existing fixed fit and terminal native-objective validation pipeline unchanged; select no new experiment from this audit.',
    )
    exclusive_json('audit.json', result)
    report = '''# Native gradient history: bounded metadata review

The current signed-message Torch checks establish repeatability and a small
synthetic encoder check. They explicitly do not certify the derivative of the
complete native predictor. Exact optimizer/RNG transport across processes has
the same limitation: reproducing an approximate gradient does not certify it.

The earlier d50 native cold search did execute the native objective, following
the original resource-failed attempt. It completed and failed its scientific
gate: FIT4 adjusted ratio 0.9913750625804131 with 2/4 wins; CHECK4 ratio
1.020150121188754 with 0/4 wins. These are old TRAIN8 research results, not
current-model, VALID, independent or official scores. Do not describe this
search as unexecuted or promote the small FIT-only gain.

The two-layer native trajectory acquisition passed primal and branch checks,
but its secant remained unqualified. Its report records an insufficient signal
and excessive curvature; this audit did not recompute that scalar reduction.
No backward qualification follows from the successful forward acquisition.

The shared384 native-backed rangefinder passed three component real-algebra
VJP cases. The subsequent full-model forward check with native Y still failed:
maximum absolute discrepancy 3.6835670471191406e-5 against a 1e-5 limit.
Neither receipt certifies the complete current model's native gradient.

The covariance-gradient gap study completed its bookkeeping successfully but
reported numerically inconclusive gradients and a failed information gate.
Its top-level passed flag must not be read as a positive scientific result.

All nine result hashes were checked and their relevant scalar fields asserted.
The five context documents are separately pinned. The review opened JSON and
text metadata only, used no numerical library, decoded no prediction or target
arrays, changed no model, and started no experiment. This is a bounded review
of named studies, not an exhaustive assertion that no other method exists.

The existing fixed 160-graph fit and terminal native-objective acceptance gates
remain the next test. Approximate training proposals must be judged by actual
native predictions and their complete costs. This audit adds no derivative
qualification detour and reopens no failed study. Official submission 331760
remains 3.507738856975084e-9 adjusted; the larger goal remains active.
'''
    with (OUT / 'RESULT.md').open('x') as stream:
        stream.write(report)
    print(json.dumps(dict(completed=True, audit=pin(OUT / 'audit.json'), report=pin(OUT / 'RESULT.md'),
        primary_receipts_verified=len(pins), numerical_model_calls=0)))


if __name__ == '__main__':
    main()
