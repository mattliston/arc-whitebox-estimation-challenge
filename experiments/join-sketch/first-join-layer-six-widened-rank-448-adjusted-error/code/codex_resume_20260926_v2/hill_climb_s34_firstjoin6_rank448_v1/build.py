"""One TRAIN-guided first-join rank tradeoff; stdlib source/scalar verification only."""
from pathlib import Path
import ast
import copy
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE = H / 'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
PRIOR = H / 'hill_climb_s34_firstjoin6_v1'

def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def rewrite(value):
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, str):
        return value.replace(TEMPLATE.parent.name, D.name)
    return value

assert pin(SOURCE)['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(TEMPLATE)['sha256'] == '442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source = SOURCE.read_text()
fragment = (D / 'fragment.py').read_text()
expected = ('# Fixed S34 allocation: first join L6 at rank448; later joins and ranks unchanged.\n'
    '_JOIN_SCHEDULE = {6: (0, 2), 10: (3, 5), 13: (6, 8)}\n'
    '_R320_RULE = {6: 448, 10: 320, 13: 384}\n')
assert fragment == expected and source.endswith('\n')
candidate = source + '\n' + fragment
assert candidate[:-len('\n' + fragment)] == source
old_tree, new_tree = ast.parse(source), ast.parse(candidate)
assert len(new_tree.body) == len(old_tree.body) + 2
assert all(ast.dump(a) == ast.dump(b) for a, b in zip(old_tree.body, new_tree.body))
assert [ast.dump(node) for node in new_tree.body[-2:]] == [ast.dump(node) for node in ast.parse(fragment).body]
assignments = {}
functions = {}
for node in old_tree.body:
    if isinstance(node, ast.FunctionDef):
        functions[node.name] = node
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        assignments[node.targets[0].id] = node
assert ast.literal_eval(assignments['_JOIN_SCHEDULE'].value) == {7: (0, 2), 10: (3, 5), 13: (6, 8)}
assert ast.literal_eval(assignments['_R320_RULE'].value) == {7: 384, 10: 320, 13: 384}
assert ast.literal_eval(assignments['_SLR_LAZY_LAYERS'].value) == (13,)
assert ast.literal_eval(assignments['_SLR_MATERIALIZE_LAYERS'].value) == (10,)
assert ast.literal_eval(assignments['_SLR_CONSOLIDATE_LAYERS'].value) == ()
scope = {}
exec(compile(fragment, '<two literal dictionaries>', 'exec'), scope)
joins, ranks = scope['_JOIN_SCHEDULE'], scope['_R320_RULE']
assert list(joins) == list(ranks) == [6, 10, 13]
assert list(joins.values()) == [(0, 2), (3, 5), (6, 8)]
assert list(ranks.values()) == [448, 320, 384]
# Scalar cohort ownership for any positive even cohort widths. The current newborn
# is layer-1; these unchanged join intervals never include it and remain contiguous.
next_cohort = 0
for layer, (first, last) in joins.items():
    assert first == next_cohort and first <= last <= layer-2
    next_cohort = last+1
assert next_cohort == 9
active = ast.get_source_segment(source, functions['_shared_old384_push'])
# The final definition is OTT context ownership; its captured predecessor is the
# last implementation with the actual join. Inspect that exact predecessor body.
implementations = [node for node in old_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_shared_old384_push' and
                   "joins_here = _JOIN_SCHEDULE.get(layer)" in ast.get_source_segment(source, node)]
join_body = ast.get_source_segment(source, implementations[-1])
assert "if old_columns:\n            overlap =" in join_body
assert "else:\n            factors = Fj" in join_body
assert "if newborn is None or len(state['cohorts']) != layer:" in join_body
assert "if start != old_columns or width <= 0 or stop - _cp_old0 > int(p.shape[1]):" in join_body
assert "if old_columns > int(inherited_columns):" in join_body
assert 'layer == 7' not in join_body and 'layer < 7' not in join_body
assert '_R320_RULE[layer]' in ast.get_source_segment(source, functions['_r320_rank'])
# Fixed-rank names are historical; active allocation and carried state are dynamic.
assert "int(previous.shape[0]) < rank" in join_body
assert "state['rank'] = state.pop('rank_new')" in join_body
assert "Qn.shape) != (int(previous.shape[0]), rank)" in join_body
assert "factors = Fj" in join_body
assert "overlap =" in join_body and "Qn.T" in join_body and "Qp" in join_body
rank_flow = {6: 448, 7: 448, 8: 448, 9: 448, 10: 320, 13: 384}
assert rank_flow[10] == ranks[10] and rank_flow[13] == ranks[13]
# The only constant-384 virtual-rotation envelope executes at lazy L13, after
# L10 has materialized the 448 -> 320 map; it does not restrict L6-9 to 384.
assert "_slr_virtual_rotation_envelope" in source
prior_assessment = PRIOR / 'native2_assessment.json'
assert pin(prior_assessment)['sha256'] == 'b8afddc65e58b2738501878f0a19053ffc073556b675defca1f4db4129c594ee'
prior_result = json.loads(prior_assessment.read_text())
assert prior_result['admitted'] is False and prior_result['ranks'] == [35, 100]
assert prior_result['candidate']['sha256'] == 'c1bbceff696f83c34464a495e2639a9786812b12659ef2dd76c07a7bb1b0b642'
selection_evidence = dict(
    assessment=pin(prior_assessment), run_guard=pin(PRIOR / 'native2_guard.json'),
    assessment_guard=pin(PRIOR / 'native2_assess_guard.json'), config=pin(PRIOR / 'native_screen.json'),
    raw_pct=prior_result['raw_pct'], adjusted_pct=prior_result['adjusted_pct'],
    bill_pct=prior_result['bill_pct'], adjusted_wins=prior_result['adjusted_wins'],
    max_per_network_regression_pct=prior_result['max_per_network_adjusted_regression_pct'],
    prior_failed_result_preserved=True, selection_uses_TRAIN_outcomes=True,
    selection_type='Explicit root-selected post-outcome rank/fidelity tradeoff, one rank448 endpoint; not a blind confirmatory test.')
with (D / 'candidate.py').open('x') as stream:
    stream.write(candidate)
protocol = dict(schema='s34-firstjoin6-rank448-v1', root_selected=True, baseline=pin(SOURCE),
    join_schedule={str(key): list(value) for key, value in joins.items()},
    ranks={str(key): value for key, value in ranks.items()},
    sole_allocation_change='Versus S34: first join L7/rank384 to L6/rank448; same cohorts0-2. Later L10/rank320 and L13/rank384 unchanged.',
    hypothesis='One extra 64-column rank block may recover fidelity lost by the failed L6/rank384 pilot while retaining some measured earlier-join cost savings.',
    accuracy_risk='Earlier rank448 projection still changes all later states; greater rank need not improve adjusted score and no gain is promised.',
    cost_evidence_scope='Failed S34 L6/rank384 N2 saved 1.949019% bill but worsened raw 2.653327%; rank448 increases its cost by an unmeasured amount.',
    training_selection=selection_evidence, rank_flow=rank_flow,
    source_checks=dict(first_join_uses_Fj=True, strict_rank_key_present=True,
        unchanged_cohort_intervals_contiguous=True, current_newborn_not_joined=True,
        materialize_layer10_unchanged=True, lazy_layer13_unchanged=True,
        consolidate_empty_previous_join_unused=True,
        layer6_OTT_YNT_fallback_unchanged=True, original_runtime_guards_preserved=True,
        dynamic_rank448_QR_core_layout_carry=True, layer10_overlap_shape=[320, 448],
        layer10_materializes_rank320_before_lazy13=True,
        constant384_virtual_envelope_only_lazy13=True),
    all_other_source_bytes_unchanged=True, arithmetic_coefficients_unchanged=True,
    scratch_setup_and_cache_rules_unchanged=True, no_new_arithmetic_fixture=True,
    no_rank_or_schedule_sweep=True, native2_roster=[35, 100], parity_layers=list(range(6)),
    native2_gates=dict(adjusted_pct_at_most=-0.1, wins_min=1, max_per_network_regression_pct=1.0),
    scope='native2_only', boundary_authorized=False, broad_authorized=False,
    serving_qualified=False, numpy_torch_or_model_executed=False)
once(D / 'protocol.json', protocol)
build = dict(candidate=pin(D / 'candidate.py'), parent=pin(SOURCE), builder=pin(__file__),
    fragment=pin(D / 'fragment.py'), protocol=pin(D / 'protocol.json'),
    source_inverse_exact=True, original_statements_ast_exact=True,
    appended_literal_dictionary_assignments=['_JOIN_SCHEDULE', '_R320_RULE'],
    scalar_cohort_and_rank_checks_passed=True, no_new_runtime_guards_or_wrappers=True,
    serving_qualified=False, submitted=False, numpy_torch_or_model_executed=False)
once(D / 'build.json', build)
base_config = json.loads(TEMPLATE.read_text())
config = rewrite(copy.deepcopy(base_config))
config['arms']['candidate']['source'] = build['candidate']
assert config['arms']['base'] == base_config['arms']['base']
assert config['arms']['parity_parent'] == base_config['arms']['parity_parent']
assert config['arms']['base']['source'] == pin(SOURCE)
config['parity_layers'] = list(range(6))
config['provenance'] = [pin(path) for path in (
    TEMPLATE, SOURCE, D / 'build.py', D / 'fragment.py', D / 'protocol.json',
    D / 'build.json', H / 'official_bank_S34_v1.json', prior_assessment,
    PRIOR / 'native2_guard.json', PRIOR / 'native2_assess_guard.json', PRIOR / 'native_screen.json')]
config['interpretation'] = (
    'One explicit TRAIN-guided follow-on to failed L6/rank384: current-S34 first join L6/rank448, same cohorts0-2. '
    'Prior failed result pinned and preserved; one selected rank, no sweep. Later joins/ranks/lazy-materialize schedules unchanged. Two literal dictionary assignments '
    'with exact source inverse and scalar cohort proof. Genuine S34 controls; exact L0-5 parity; '
    'unchanged score/resource gates. No new arithmetic fixture, sweep, boundary, additional-six, '
    'broad or serving authority.')
assert config['prospective_scope'] == 'native2_only'
for stage in ('2', '8'):
    assert config['stages'][stage]['gates'] == base_config['stages'][stage]['gates']
once(D / 'native_screen.json', config)
print(json.dumps(dict(candidate=build['candidate'], config=pin(D / 'native_screen.json'),
                      build=pin(D / 'build.json'), protocol=pin(D / 'protocol.json'))))
