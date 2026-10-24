"""One fixed late-c0 change on official S34; source/scalar proof only."""
from pathlib import Path
import ast
import copy
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE = H / 'hill_climb_s34_ynt_zonal3_v1/native_screen.json'

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
expected = ('# Fixed S34 experiment: reduce the pair-map c0 dial only at layers 10 through 14.\n'
    '_K31R_TABLE = {layer: (0.480758989687834, values[1]) if layer in (10, 11, 12, 13, 14) '
    'else values for layer, values in _K31R_TABLE.items()}\n')
assert fragment == expected
assert source.endswith('\n')
candidate = source + '\n' + fragment
assert candidate[:-len('\n' + fragment)] == source
old_tree, new_tree = ast.parse(source), ast.parse(candidate)
assert len(new_tree.body) == len(old_tree.body) + 1
assert all(ast.dump(a) == ast.dump(b) for a, b in zip(old_tree.body, new_tree.body))
assert ast.dump(new_tree.body[-1]) == ast.dump(ast.parse(fragment).body[0])
# Execute only the three scalar table assignments, never the candidate module.
selected = []
for node in old_tree.body:
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('_K31R_TABLE', '_K31R_TABLE_GN')):
        selected.append(node)
assert [node.targets[0].id for node in selected] == ['_K31R_TABLE', '_K31R_TABLE_GN', '_K31R_TABLE']
scope = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<scalar-initial-tables>', 'exec'), scope)
before = dict(scope['_K31R_TABLE'])
gn_before = scope['_K31R_TABLE_GN']
gn_values = dict(gn_before)
assert before == {layer: (0.5341766552087044, 0.5026934507950427) for layer in range(1, 15)}
assert gn_values == {layer: (0.85, 0.5) for layer in range(1, 15)}
exec(compile(fragment, '<scalar-late-c0>', 'exec'), scope)
after = scope['_K31R_TABLE']
assert list(after) == list(before)
assert scope['_K31R_TABLE_GN'] is gn_before and scope['_K31R_TABLE_GN'] == gn_values
assert 0.90 * 0.5341766552087044 == 0.480758989687834
for layer in before:
    assert after[layer] == ((0.480758989687834, before[layer][1]) if 10 <= layer <= 14 else before[layer])
with (D / 'candidate.py').open('x') as stream:
    stream.write(candidate)
protocol = dict(schema='s34-late-c0-90-v1', root_selected=True, baseline=pin(SOURCE),
    exact_layers=[10, 11, 12, 13, 14], original_c0=0.5341766552087044,
    fixed_c0=0.480758989687834, coefficient=0.90, c1_unchanged=True,
    earlier_layer_values_unchanged=True, gn_table_unchanged=True,
    hypothesis='A fixed 10% reduction of late pair-map c0 may reduce current-S34 mean error.',
    prior_related_failures_remain_closed=True, exact_duplicate_found=False,
    roster=[35, 100], parity_layers=list(range(10)), bounded_training_follow_on=True, additional_strengths_authorized=False,
    native2_gates=dict(adjusted_pct_at_most=-0.1, wins_min=1, max_per_network_regression_pct=1.0),
    native8_gates=dict(adjusted_pct_at_most=-0.1, wins_min=6, ci_upper_below=0, additional6_negative=True),
    scope='native2_only', boundary_authorized=False, broad_authorized=False, serving_qualified=False,
    native_gradient_certified=False, numpy_torch_or_model_executed=False)
protocol['selection'] = 'Root selected one TRAIN-guided stronger step after the .95 pilot improved .0671% but missed .1%. Prior failure preserved. No VALID used; no further strength authorized.'
protocol['prior_closed_result'] = pin(H/'hill_climb_s34_late_c0_95_v1/closeout.json')
once(D / 'protocol.json', protocol)
build = dict(candidate=pin(D / 'candidate.py'), parent=pin(SOURCE), builder=pin(__file__),
    fragment=pin(D / 'fragment.py'), protocol=pin(D / 'protocol.json'),
    source_inverse_exact=True, original_statements_ast_exact=True, appended_assignments=1,
    scalar_table_check_passed=True, changed_layers=[10, 11, 12, 13, 14],
    c1_values_exact=True, earlier_layer_values_exact=True, gn_values_and_identity_exact=True,
    numpy_torch_or_model_executed=False, serving_qualified=False, submitted=False)
once(D / 'build.json', build)
base_config = json.loads(TEMPLATE.read_text())
config = rewrite(copy.deepcopy(base_config))
config['arms']['candidate']['source'] = build['candidate']
assert config['arms']['base'] == base_config['arms']['base']
assert config['arms']['parity_parent'] == base_config['arms']['parity_parent']
assert config['arms']['base']['source'] == pin(SOURCE)
config['parity_layers'] = list(range(10))
config['provenance'] = [pin(path) for path in (
    TEMPLATE, SOURCE, D / 'build.py', D / 'fragment.py', D / 'protocol.json',
    D / 'build.json', H / 'official_bank_S34_v1.json')]
config['interpretation'] = (
    'One fixed current-S34 literal-table experiment: c0=0.480758989687834 only at L10-14. '
    'c1, earlier layers and configure-time GN table unchanged. Scalar AST/source proof only; '
    'N2 uses original engine and genuine S34 controls, exact L0-9 parity, all-finite checks, '
    'unchanged score/resource gates. One TRAIN-guided follow-on only; no boundary, additional-six, broad or serving authority.')
assert config['prospective_scope'] == 'native2_only'
for stage in ('2', '8'):
    assert config['stages'][stage]['gates'] == base_config['stages'][stage]['gates']
once(D / 'native_screen.json', config)
print(json.dumps(dict(candidate=build['candidate'], config=pin(D / 'native_screen.json'),
                      build=pin(D / 'build.json'), protocol=pin(D / 'protocol.json'))))
