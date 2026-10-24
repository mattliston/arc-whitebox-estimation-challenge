from pathlib import Path
import ast
import hashlib
import json
import subprocess

R = Path('/Users/matt/arc-white-box-estimation')
O = R / 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
D = O / 'lead/runs/astra_oct20_bill_v1'
V = O / 'lead/runs/astra_oct20_vm_frontier_v1'
H = D / 'runtime_first128_frozen_census_v1'
P = D / 'runtime_smoff_npxoff_first128_v1'
A = O / 'lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1'
LA = A / 'lead_audit_2323_v1'
refs = {}

def read(p):
    return json.loads(p.read_text())

def pin(key, p, expected=None):
    p = p.resolve()
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, (p, digest, expected)
    refs[key] = {'path': str(p), 'sha256': digest}
    return key

paths = {
    'failure_review': LA / 'ROOT_337473_FAILURE_REVIEW_v1.json',
    'failure_case': LA / 'PUBLIC337473_FAILURE_INDEX27_v1.json',
    'hosted_runtime_pairs': LA / 'PUBLIC337473_FINAL_RUNTIME_PAIRS_VS466_v1.json',
    'sdk_version_parity': LA / 'HOSTED_SDK_VERSION_PARITY_SEAL_v1.json',
    'exact_failed_sync_audit': V / 'sync_exact4685_v1/EXACT4685_SYNC_AUDIT_v1.json',
    'exact_failed_sync_recipe': V / 'sync_exact4685_v1/RECIPE_v1.json',
    'exact_failed_sync_root_review': V / 'sync_exact4685_v1/ROOT_REVIEW_v1.json',
    'paid_build': P / 'BUILD_v1.json',
    'paid_source_ready': P / 'READY_SOURCE_v1.json',
    'paid_root_source_review': P / 'ROOT_SOURCE_REVIEW_v1.json',
    'paid_imports': P / 'cloud_checks_v1/IMPORT_CHECK_v1.json',
    'paid_vm32_invoice_audit': P / 'native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json',
    'census_build': H / 'BUILD_v1.json',
    'census_static_review': H / 'SOURCE_REVIEW_v1.json',
    'census_root_source_import_review': H / 'ROOT_SOURCE_IMPORT_REVIEW_v1.json',
    'census_ready_source': H / 'READY_SOURCE_v1.json',
    'census_full_imports': H / 'IMPORT_CHECK_v1.json',
    'census_descriptive_typo_correction': H / 'EVIDENCE_CORRECTION_v1.json',
    'profile_strict_equality_audit': V / 'profile_first128_d399_v2/INDEPENDENT_AUDIT_v1.json',
    'final_mac32_audit': H / 'mac32_packet_v1/INDEPENDENT_FINAL_MAC32_AUDIT_v1.json',
    'final_vm32_audit': H / 'native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json',
    'root_vm32_knowncold4_audit': H / 'ROOT_VM32_AND_KNOWNCOLD4_CUSTODY_v1.json',
    'knowncold4_audit': H / 'fresh4_92_95_prepared_v1/INDEPENDENT_NATIVE_CUSTODY_AUDIT_v1.json',
    'knowncold4_exact_parent_audit': H / 'fresh4_92_95_prepared_v1/OWN_PARENT_EQUALITY_AUDIT_v1.json',
    'whole12_confirmation_score': H / 'fresh12_confirmation_v1/SCORE_WHOLE12_ACTUAL337466_v1.json',
    'whole12_confirmation_root_review': H / 'fresh12_confirmation_v1/ROOT_RESULT_REVIEW_v1.json',
    'cold13_plan': H / 'root_cold13_v1/COVERAGE_PLAN_v1.json',
    'cold13_result': H / 'root_cold13_v1/COVERAGE_RESULT_v1.json',
    'cold13_root_result_review': H / 'root_cold13_v1/ROOT_RESULT_REVIEW_v1.json',
    'cold13_isolation': H / 'root_cold13_v1/ISOLATION_v1.json',
    'warm8_ready': H / 'root_warm8_v1/READY_ROOT_REVIEW_v1.json',
    'warm8_plan': H / 'root_warm8_v1/COVERAGE_PLAN_v1.json',
    'warm8_root_review': H / 'root_warm8_v1/ROOT_REVIEW_v1.json',
    'warm8_live_preflight': H / 'root_warm8_v1/ROOT_LIVE_PREFLIGHT_v1.json',
    'warm8_analyzer': H / 'root_warm8_v1/analyze_coverage_v1.py',
    'warm8_runner': H / 'root_warm8_v1/run_isolated_root_v1.py',
    'preserved_original_group_control_memory_failure': H / 'root_coverage13_v1/CONTROL_MEMORY_FAILURE_REVIEW_v1.json',
    'preserved_paid_parent_rank316_outlier': H / 'PARENT_RANK316_REVIEW_v1.json',
    'descriptive_all32_runtime': H / 'MAC32_DESCRIPTIVE_RUNTIME_VS_FE8_466_v1.json',
    'warm_census_mechanism_and_limits': H / 'WARM_CENSUS_MECHANISM_AND_LIMITATION_v1.json',
    'remaining_profile_closed_frontier': H / 'REMAINING_PROFILE_FRONTIER_v1.json',
    'closed_generic_hotloop_hold': O / 'lead/runs/astra_crunch_runtime_v1/postsubmit_margin_v1/CURRENT_HOST_CAP_CLOSURE_v1.json',
    'closed_generic_callcost': O / 'lead/runs/opus_callcost_v1/CALLCOST_RESULT_v1.json',
    'closed_generic_callform': O / 'lead/runs/opus_callform_v1/mb_v1.json',
    'closed_dag_actual_quiet': V / 'runtime_smdag_schedule_v1/root_quiet_v1/QUIET_RESULT_smdag_v1.json',
    'closed_current896_incidence': D / 'runtime_first128_inner896_v1/CURRENT32_INCIDENCE_BUDGET_v1.json',
    'closed_current896_budget': D / 'runtime_first128_inner896_v1/CLOSURE_v1.json',
    'unchanged_release_script': O / 'lead/runs/fable_oct19_l13rank_v1/release_min_v1.sh',
    'unchanged_fire_once': O / 'fire/fire_once.py',
}
for k, p in paths.items():
    pin(k, p)

build = read(H / 'BUILD_v1.json')
paid = read(P / 'BUILD_v1.json')
source_checks = []
old_text = "_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=_setup_gc.get_freeze_count())"
new_text = "_GCZ_STATS.update(predicts=_GCZ_STATS['predicts'] + 1, frozen_objects=None)"
for role in ('science', 'release'):
    b = build['outputs']['paid'][role]
    final = Path(b['path'])
    parent = Path(b['parent_path'])
    pin('final_' + role, final, b['sha256'])
    pin('paid_parent_' + role, parent, b['parent_sha256'])
    physics = paid['outputs']['first128'][role]
    pin('first128_physics_' + role, Path(physics['parent_path']), physics['parent_sha256'])
    f, p = final.read_text(), parent.read_text()
    assert p.count(old_text) == 1 and f.count(new_text) == 1
    assert p.replace(old_text, new_text, 1) == f
    assert ast.dump(ast.parse(f.replace(new_text, old_text, 1)), include_attributes=False) == ast.dump(ast.parse(p), include_attributes=False)
    source_checks.append({'role': role, 'exact_one_text_expression_replacement': True, 'reconstituted_parent_AST_exact': True})

sr = read(V / 'sync_exact4685_v1/RECIPE_v1.json')
failed = R / sr['candidate']['path']
pin('exact_failed_release', failed, sr['candidate']['sha256'])
final = Path(build['outputs']['paid']['release']['path'])
old_tree, new_tree = ast.parse(failed.read_text()), ast.parse(final.read_text())
old_defs = {n.name: n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_defs = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
changed = [k for k, n in new_defs.items() if k not in old_defs or ast.dump(n, include_attributes=False) != ast.dump(old_defs[k], include_attributes=False)]
assert set(changed) == {'_gop2_gnn_step', '_gcz_predict_v1', '_fh90_first_head'}, changed
unchanged_critical = []
for name in ('_dcr_right', '_slr_virtual_rotation_envelope', '_E2Hold'):
    a, b = old_defs[name], new_defs[name]
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    unchanged_critical.append({'function': name, 'failed_line': a.lineno, 'final_line': b.lineno, 'AST_exact_to_failed_source': True,
                               'AST_sha256': hashlib.sha256(ast.dump(b, include_attributes=False).encode()).hexdigest()})
changed_flow = []
for name in changed:
    n = new_defs[name]
    tests = [{'line': x.lineno, 'expression': ast.unparse(x.test)} for x in ast.walk(n) if isinstance(x, (ast.If, ast.IfExp, ast.While, ast.Assert))]
    casts = [{'line': x.lineno, 'expression': ast.unparse(x)} for x in ast.walk(n) if isinstance(x, ast.Call) and ast.unparse(x.func) in ('bool', 'float', 'int', 'len', 'list', 'tuple', 'str')]
    changed_flow.append({'function': name, 'line': n.lineno, 'control_tests': tests, 'explicit_scalar_or_container_conversions': casts})

for k in ('tiny_reuse', 'np_tiny_reuse'):
    p = paid[k]
    pin(k, Path(p['path']), p['sha256'])
    if 'guard_path' in p:
        pin(k + '_guard', Path(p['guard_path']), p['guard_sha256'])
for k in ('tail', 'np_tail', 'nogov'):
    p = paid[k]
    pin('paid_' + k, Path(p['path']), p['sha256'])
ready = read(H / 'READY_SOURCE_v1.json')
for p in H.glob('*GUARD*.json'):
    if hashlib.sha256(p.read_bytes()).hexdigest() == ready['import_guard_sha256']:
        pin('census_import_original_guard', p, ready['import_guard_sha256'])
assert 'census_import_original_guard' in refs

mac = read(H / 'mac32_packet_v1/INDEPENDENT_FINAL_MAC32_AUDIT_v1.json')
vm = read(H / 'native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json')
paidvm = read(P / 'native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json')
cold = read(H / 'root_cold13_v1/ROOT_RESULT_REVIEW_v1.json')
conf = read(H / 'fresh12_confirmation_v1/ROOT_RESULT_REVIEW_v1.json')
desc = read(H / 'MAC32_DESCRIPTIVE_RUNTIME_VS_FE8_466_v1.json')
for name, row in mac['paired_same_host_comparisons'].items():
    pin('mac32_paired_' + name, Path(row['receipt']), row['sha256'])
pin('vm32_paired_best337466', Path(vm['current_best_report']), vm['current_best_report_sha256'])
assert sum(v['actual_paid_delta'] for v in mac['paid_rows'].values()) == paidvm['actual_added_flops_sum'] == 13629046784
assert all(v['actual_diagnostic_omission_incremental_flops'] == 0 for v in mac['paid_rows'].values())
assert vm['all_invoices_and_operations_exact_to_first128fe8'] and mac['all32_exact_own_first128_prediction_bytes']

result = {
    'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
    'status': 'EVIDENCE_INDEX_ONLY_FINAL_WARM_RUNTIME_REVIEW_PENDING_NO_RELEASE_OR_FIRE',
    'candidate': {'science_sha256': build['outputs']['paid']['science']['sha256'], 'release_sha256': build['outputs']['paid']['release']['sha256'],
                  'lineage': 'First128 4da/e491 physics, fixed paid SMoff+NPoff fe8/d90, one unused diagnostic census expression omitted 49bd/f938.'},
    'public_context': {'best_verified_submission': 337466, 'best_verified_score': 1.4167660830412778e-9, 'last_verified_J2W_score': 1.416013781559781e-9,
                       'failed_submission': 337473, 'failed_residual_s': 0.401729281, 'cap_s': 0.400,
                       'unchanged_failed_retry_forbidden': True, 'public_refresh_and_quota_owner': 'root; this read-only evidence index does not refresh or spend quota'},
    'materialization_audit': {
        'exact_failed_trace': 'exact_failed_sync_audit',
        'local_rank35_observed_bool_fetches': 9,
        'observed_DCR_bool_calls': 8,
        'observed_SLR_bool_calls': 1,
        'observed_other_explicit_or_implicit_array_coercions': [],
        'observed_metadata_len_calls': 224,
        'local_bool_outside_counted_dispatch_ms': 0.109944,
        'required_final_fetch_was_separate_harness_phase': True,
        'static_final_delta': 'Final f938 retains both required array-check sites and their exact definitions. It is not a zero-materialization source. No extra array scalar materialization was introduced by the three changed/added effective definitions.',
        'critical_definitions': unchanged_critical,
        'effective_definitions_changed_vs_failed': changed_flow,
        'changed_function_control_provenance': 'New/changed GNN branches use layer integers, cached dtype/shape and Python None; its only explicit scalar conversion is int(sigma.shape[0]) via Python list.append(None) or metadata. The head cache uses rows is None. No bool/float/int(array), array iteration, implicit if(array), or array formatting was added.',
        'implicit_or_provenance': '_E2Hold inherits list.append unchanged. append(value) or value tests Python None, not the backend array. That invariant remains exact to failed4685.',
        'guard_preservation': 'DCR ownership identity, tie-safe scatter fallback, SLR overflow predicate and exception remain unchanged. Zero-sync DCR/SLR descendants were separately measured and are not part of f938.',
        'limitations': ['Single exact4685 rank35 diagnostic is not an all-case count or a finalf938 dynamic audit.', 'Local observer timing is not a public failure attribution or a promised runtime saving.', 'Hosted and local flopscope versions are 0.12.3; no historical 0.12.0 mismatch rationale.', 'Final actual timing qualification comes from fresh paired exactf938 runs, not source request counts.'],
        'forum_url': sr['forum_url'],
    },
    'minimal_census_change': {
        'checks': source_checks, 'old_expression': old_text, 'new_expression': new_text,
        'meaning_of_None': 'Diagnostic count was not measured; no prediction-control consumer reads it.',
        'preserved': ['GC freeze/collect/thresholds/order', 'prediction counter and statistics update', 'all FNP arithmetic and request/free order', 'all physical ownership and overflow guards', 'all official/resource guards', 'rescue disabled with no governors'],
        'full_imports': 8, 'metadata_return_exception_finally_fixtures': 8, 'numeric_models_in_import_check': 0,
        'parent_import_cache_key': '(parent_sha256, role)',
        'profile_mechanism_evidence_only': 'Exact qualified d399 profile found 8.581425ms C self-time in get_freeze_count; instrumentation inflates timing, and the source differs from f938. No additive elapsed-time credit is taken.',
    },
    'paid_score_evidence': {
        'all32_Mac_same_host_vs_latest_verified_best': mac['paired_same_host_comparisons']['best337466']['all32'],
        'held25_Mac_same_host_vs_latest_verified_best': mac['paired_same_host_comparisons']['best337466']['heldout25'],
        'all32_VM_same_host_vs_latest_verified_best': vm['current_best_all'],
        'held25_VM_same_host_vs_latest_verified_best': vm['current_best_held25'],
        'all32_Mac_own_first128_paid_debit': mac['paired_same_host_comparisons']['own_first128']['all32'],
        'all32_paid_added_FLOPs_min': paidvm['actual_added_flops_min'],
        'all32_paid_added_FLOPs_max': paidvm['actual_added_flops_max'],
        'all32_paid_added_FLOPs_sum': paidvm['actual_added_flops_sum'],
        'census_incremental_FLOPs_all32': 0,
        'byte_and_invoice_proof': 'Both final Mac32 and VM32 equal their own fe8 predictions and complete invoices; fe8 equals own first128 predictions with every paid stack/copy debit included.',
        'NP_decomposition_limit': 'Original native rows lack an independent NP counter. Total actual invoice debit is authoritative; no separate unproved NP counter tariff is claimed.',
        'fresh12_confirmation': conf['all'],
        'fresh12_confirmation_only_not_new_unseen_holdout': True,
        'not_a_hosted_score_or_banked_gain': True,
    },
    'runtime_evidence': {
        'cold13': {'root_review': 'cold13_root_result_review', 'fixed_pairs': 13, 'original_guards': 26,
                   'mean_changes_pct': cold['mean_changes_pct'], 'actual_residual_pct_range': cold['residual_pct_range'],
                   'max_rss_bytes': cold['max_rss_bytes'], 'max_swap_growth_bytes': cold['max_swap_growth_bytes'], 'min_pressure_free_pct': cold['min_pressure_free_pct'],
                   'all_actual_invoices_retained': True, 'old_bill_equality_context': 'Only original five group-first cold ranks; no old warm invoice transfer.', 'limitation': 'All singleton cold; does not by itself qualify warm retained-cache runtime.'},
        'warm8': {'status': 'PENDING_FULL_ROOT_RESULT_REVIEW', 'root_exec_session': 65595, 'launched_utc': '2026-10-21T09:17:34Z',
                  'fixed_pairs': [[35,150],[35,200],[250,259],[277,287],[16,293],[16,300],[310,312],[310,316]],
                  'fresh466_then_f938_each_pair': True, 'unchanged_guards': '125s, 7.5GB RSS backstop, free50; all original isolation checks',
                  'all32_prediction_observations_anchored': True, 'actual_new_invoices_required_for_every_observation': True,
                  'exact_old_invoice_contexts': 'Eight prefixes and five original-second targets 150,259,287,293,312.',
                  'reduced_retained_depth_ranks': [200,300,316], 'limitation': 'Those three former third positions are now second. This probes one retained prediction, not original three-prediction memory depth.',
                  'no_partial_runtime_selection_or_verdict': True},
        'original_group_memory_limit': {'receipt': 'preserved_original_group_control_memory_failure', 'candidate_predictions': 0,
                                        'failed_reference_third_rank': 200, 'rss_bytes': 7510163456, 'original_limit_bytes': 7500000000,
                                        'failed_partials_reused': False, 'thresholds_raised': False,
                                        'separate_latent_metadata_defect': 'Old isolation metadata omitted analyzer command key; not reached and not causal to RSS failure. New cold13/warm8 packets explicitly verify every command key.'},
        'canonical_Mac32': {'accuracy_and_invoice_qualified': True, 'all_original_groups_retained': True, 'maximum_system_swap_growth_bytes': mac['maximum_system_swap_growth_bytes'],
                            'original_conditional_swap_guards_passed': True, 'quiet_or_hosted_margin_proof': False},
        'descriptive_noncontemporaneous_all32': {'receipt': 'descriptive_all32_runtime', 'ownfe8_all32': desc['comparisons']['ownfe8']['all32'],
                                               'ownfe8_warm20': desc['comparisons']['ownfe8']['warm20'],
                                               'qualifier': 'Parallel5 science runs at different times with product-census instrumentation. Process counters, not thread counters. Every outlier retained; no causal/hosted timing claim.'},
        'hosted_dispersion_receipt': 'hosted_runtime_pairs',
        'final_runtime_verdict': 'WITHHELD_PENDING_FULL_WARM8_AND_ROOT_REVIEW',
    },
    'closed_frontier_audit': [
        {'family': 'Generic hotloop/unroll/hold inline and shape fit-check rewrites', 'state': 'CLOSED_NO_REOPEN', 'evidence': ['closed_generic_hotloop_hold', 'remaining_profile_closed_frontier'], 'reason': 'Existing warm instruction savings about .019%; ideal hold ceiling about .47%. Profile fit checks only 1.374572ms instrumented; no distinct material mechanism.'},
        {'family': 'Generic SDK call forms/cache/wrapper/prefix/compact', 'state': 'CLOSED_NO_REOPEN', 'evidence': ['closed_generic_callcost', 'closed_generic_callform', 'remaining_profile_closed_frontier'], 'reason': 'No supported material new source mechanism; no repeat based on plausibility.'},
        {'family': 'Qualified exact DAG schedule', 'state': 'CLOSED_NO_REOPEN', 'evidence': ['closed_dag_actual_quiet'], 'reason': 'Measured about .092% mean instruction reduction insufficient; request savings are not time headroom.'},
        {'family': 'Fixed inner896 paid fallback on current stack', 'state': 'CLOSED_WITHOUT_NATIVE', 'evidence': ['closed_current896_incidence', 'closed_current896_budget'], 'reason': 'Exact current32 incidence 88 calls and old tariff forecast leave about .0092 percentage point all32 J2W cushion and inadequate held25 cushion; no cheaper shape/layer grid.'},
        {'family': 'Covariance output-copy batching and fixed polynomial cohort batching', 'state': 'NOT_QUALIFIED_NOT_OPEN_NATIVE_WORK', 'reason': 'Read-only rank35 profile ceilings about135 and189 calls before overhead, together under1%; source ownership/rounding and paid debit unproved. Root rejected native work absent material mechanism.'},
        {'family': 'Removing required GC or original array guards', 'state': 'FORBIDDEN_UNSUPPORTED', 'reason': 'Required GC policy, DCR ownership/tie handling and SLR overflow failure behavior are retained.'},
    ],
    'release_state': {'release_script_run': False, 'fire_run': False, 'BEST_changed': False, 'source_changed_by_index': False,
                      'runtime_qualified_by_index': False, 'release_qualified_by_index': False,
                      'remaining': ['Full warm8 completion and independent root review including every fixed target and limitations.', 'Root judgment of materially improved actual residual margin and adjusted J2W cushion, then all unchanged release gates and once-only submission workflow if authorized.']},
    'references': refs,
}
out = H / 'RELEASE_EVIDENCE_INDEX_v1.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'path': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'references': len(refs), 'status': result['status']}))
