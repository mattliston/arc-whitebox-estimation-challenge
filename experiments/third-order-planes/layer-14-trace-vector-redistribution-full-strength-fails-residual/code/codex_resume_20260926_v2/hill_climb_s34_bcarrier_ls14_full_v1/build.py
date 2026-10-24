"""One post-birth L14 b/star redistribution; stdlib source checks only."""
from pathlib import Path
import ast, copy, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE = H / 'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
PRIOR = H / 'hill_climb_s34_bcarrier_ls14_v1'
BEFORE = """            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag
            _ps_star_diag = (3 * _ps_rest_diag - 2 * _ps_rest_diag).astype(fnp.float32)
            rest = (_ps_target32(target) - cp_post) - trace_post
            star = 3 * rest.T - 2 * fnp.diag(fnp.diag(rest))
"""
AFTER = """            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag
            rest = (_ps_target32(target) - cp_post) - trace_post
            if l == 14 and _FULL_LOWER_MOMENTS:
                b, rest, _ps_rest_diag = _bcls14(Bpost, b, rest, _ps_rest_diag)
            _ps_star_diag = (3 * _ps_rest_diag - 2 * _ps_rest_diag).astype(fnp.float32)
            star = 3 * rest.T - 2 * fnp.diag(fnp.diag(rest))
"""

def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

def rewrite(value):
    if isinstance(value, dict): return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list): return [rewrite(item) for item in value]
    if isinstance(value, str): return value.replace(TEMPLATE.parent.name, D.name)
    return value

assert pin(SOURCE)['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(TEMPLATE)['sha256'] == '442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source = SOURCE.read_text(); fragment = (D / 'fragment.py').read_text()
assert source.count(BEFORE) == 1 and '_bcls14' not in source
candidate = source.replace(BEFORE, AFTER, 1) + '\n' + fragment
inverse = candidate[:-len('\n' + fragment)]
assert inverse.count(AFTER) == 1 and inverse.replace(AFTER, BEFORE, 1) == source
old_tree, new_tree = ast.parse(source), ast.parse(candidate)
assert len(new_tree.body) == len(old_tree.body) + 2
changed = [(a, b) for a, b in zip(old_tree.body, new_tree.body) if ast.dump(a) != ast.dump(b)]
assert len(changed) == 1
assert isinstance(changed[0][0], ast.FunctionDef) and changed[0][0].name == '_predict_binary'
assert new_tree.body[-1].name == '_bcls14'
assert [ast.dump(n) for n in new_tree.body[-2:]] == [ast.dump(n) for n in ast.parse(fragment).body]
compile((D / 'probe.py').read_text(), str(D / 'probe.py'), 'exec')
with (D / 'candidate.py').open('x') as stream: stream.write(candidate)
prior_result_path = PRIOR / 'native2_assessment.json'
assert pin(prior_result_path)['sha256'] == '89911a40ca5a158b4a59efb681607330940ca366477c3827ed9559c13be3f77e'
prior_result = json.loads(prior_result_path.read_text())
assert prior_result['admitted'] is False and prior_result['ranks'] == [35, 100]
selection = dict(assessment=pin(prior_result_path),
    run_guard=pin(PRIOR / 'native2_guard.json'),
    assessment_guard=pin(PRIOR / 'native2_assess_guard.json'),
    config=pin(PRIOR / 'native_screen.json'),
    adjusted_pct=prior_result['adjusted_pct'], raw_pct=prior_result['raw_pct'],
    bill_pct=prior_result['bill_pct'], adjusted_wins=prior_result['adjusted_wins'],
    old_gate_failure_preserved=True, TRAIN_guided_selection=True,
    root_selected_one_followon=True, VALID_targets_read=False,
    interpretation='One stronger full-step follow-on to the below-gate1/8 result; no other strengths or layers authorized.')
protocol = dict(schema='s34-bcarrier-ls14-full-v1', root_selected=True, baseline=pin(SOURCE),
    training_selection=selection, layer=14, blend=1.0, denominator_floor=1e-30, full_lower_moments_only=True,
    insertion='After births, CP state, carry commit and rest assembly; before star assembly.',
    plane_map='L_C(d)[i,j] = diag(C)[i]*d[j] + 2*C[i,j]*d[i], C=Bpost.',
    residual_diagonal='Replace FP32 rest diagonal with the separate _ps_rest_diag before LS.',
    method='One diagonally preconditioned steepest-descent LS line minimizer, then fixed1.0 full step; not a dense solve or converged LS.',
    rhs='R.T@v + 2*rowsum(C*R)',
    normal_action='sum(v*v)*p + 4*rowsum(C*C)*p + 2*v*(C@p) + 2*C.T@(v*p)',
    preservation='Same-layer full K21 repeated-index plane including k3 diagonal in real arithmetic; finite-precision rounding changes.',
    not_preserved='Distinct-index third tensor and subsequent approximated star/b transport; no endpoint gain claim.',
    learned_literals_and_dials_unchanged=True, births_CP_mean_covariance_unchanged=True,
    wrappers_captures_reset_unchanged=True, scratch_arena_and_setup_keys_unchanged=True,
    related_failure='Earlier b changes before births and truth-plane repartition failures remain closed. This fixed post-birth L14 projection has a different hook.',
    arithmetic_cost='O(n^2) FP64, three matrix-vector products and pointwise/reductions; no GEMM/dense solve.',
    fit_or_sweep=False, fixture_required_before_native2=True,
    primitive_limits=dict(seconds=60, rss_bytes=1_000_000_000, added_swap_bytes=0, threads=1),
    native2_roster=[35, 100], parity_layers=list(range(15)),
    native2_gates=dict(adjusted_pct_at_most=-0.1, wins_min=1, max_per_network_regression_pct=1.0),
    scope='native2_only', boundary_authorized=False, broad_authorized=False, serving_qualified=False)
once(D / 'protocol.json', protocol)
build = dict(candidate=pin(D / 'candidate.py'), parent=pin(SOURCE), builder=pin(__file__),
    fragment=pin(D / 'fragment.py'), protocol=pin(D / 'protocol.json'), primitive=pin(D / 'probe.py'),
    edits=[dict(label='post_birth_l14_b_carrier', before=BEFORE, after=AFTER)],
    source_inverse_exact=True, changed_function_asts=['_predict_binary'], appended_helper='_bcls14',
    unchanged_top_level_statements_exact=True, numpy_torch_or_model_executed=False,
    serving_qualified=False, submitted=False)
once(D / 'build.json', build)
base_config = json.loads(TEMPLATE.read_text()); config = rewrite(copy.deepcopy(base_config))
config['arms']['candidate']['source'] = build['candidate']
assert config['arms']['base'] == base_config['arms']['base']
assert config['arms']['parity_parent'] == base_config['arms']['parity_parent']
assert config['arms']['base']['source'] == pin(SOURCE)
config['parity_layers'] = list(range(15))
config['provenance'] = [pin(path) for path in (
    TEMPLATE, SOURCE, D / 'build.py', D / 'fragment.py', D / 'probe.py', D / 'protocol.json',
    D / 'build.json', H / 'official_bank_S34_v1.json', prior_result_path,
    PRIOR / 'native2_guard.json', PRIOR / 'native2_assess_guard.json', PRIOR / 'native_screen.json')]
config['interpretation'] = (
    'One explicit TRAIN-guided full-step follow-on to the preserved below-gate1/8 pilot; same L14 LS direction and genuine S34 controls. '
    'Same-layer K21 preservation is real-arithmetic only; distinct tensor and final transport change. '
    'Root must pass source-bound paid primitive before N2. Exact L0-14 prediction parity; '
    'unchanged gates/resources. No fit/sweep, boundary/additional-six/broad/serving authority.')
assert config['prospective_scope'] == 'native2_only'
for stage in ('2', '8'):
    assert config['stages'][stage]['gates'] == base_config['stages'][stage]['gates']
once(D / 'native_screen.json', config)
print(json.dumps(dict(candidate=build['candidate'], config=pin(D / 'native_screen.json'),
                      build=pin(D / 'build.json'), protocol=pin(D / 'protocol.json'))))
