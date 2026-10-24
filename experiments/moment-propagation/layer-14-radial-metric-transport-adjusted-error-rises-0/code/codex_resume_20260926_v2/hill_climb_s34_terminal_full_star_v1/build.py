"""One terminal full-star operator on exact S34; stdlib source preparation only."""
from pathlib import Path
import ast, copy, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE = H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'

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
source = SOURCE.read_text(); fragment = (D/'fragment.py').read_text()
old_tree = ast.parse(source); extra = ast.parse(fragment)
assert len(extra.body) == 1 and isinstance(extra.body[0], ast.FunctionDef)
assert extra.body[0].name == '_flms_star_diagonal'
old_function = [n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_flms_star_diagonal'][-1]
new_function = extra.body[0]
# Shape/dtype checks, weight orientation and paid reduction remain literally identical.
assert [ast.dump(x) for x in new_function.body[:3]] == [ast.dump(x) for x in old_function.body[:3]]
assert ast.dump(new_function.body[-1]) == ast.dump(old_function.body[-1])
expected_product = ast.parse("pushed = _mm(wt, A if str(A.dtype) == 'float32' else A.astype(fnp.float32))").body[0]
assert ast.dump(new_function.body[3]) == ast.dump(expected_product)
assert len(new_function.body) == 5
callers = []
for node in old_tree.body:
    if isinstance(node, ast.FunctionDef):
        callers.extend(node.name for call in ast.walk(node) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name) and call.func.id == '_flms_star_diagonal')
assert callers == ['_l15_rowcut_k3']
candidate = source + '\n' + fragment
assert candidate[:-len('\n' + fragment)] == source
tree = ast.parse(candidate)
assert len(tree.body) == len(old_tree.body)+1
assert all(ast.dump(a) == ast.dump(b) for a,b in zip(old_tree.body, tree.body))
assert ast.dump(tree.body[-1]) == ast.dump(new_function)
with (D/'candidate.py').open('x') as stream: stream.write(candidate)
protocol = dict(schema='s34-terminal-full-star-v1', root_selected=True, baseline=pin(SOURCE),
    hypothesis='Restore complete discarded star content in the terminal third-moment read; no later approximation propagates the change.',
    change='Only override _flms_star_diagonal: replace rank32 factor contraction by the existing paid _mm(w.T,A32).',
    formula='For each retained terminal weight column w, sum_j ((w.T@A32)[j]*w[j])*w[j].',
    sole_caller='_l15_rowcut_k3', layers=[15], parity_layers=list(range(15)),
    terminal_row_selection_unchanged=True, terminal_cp_and_b_carriers_unchanged=True,
    hidden_layer_star_factors_unchanged=True, learned_literals_and_dials_unchanged=True,
    wrappers_captures_reset_unchanged=True, scratch_arena_and_setup_keys_unchanged=True,
    reduction='Same FP32 products and float64 row sum; _mm may change rounding relative to the factor route.',
    exactness='Full stored star contraction in real arithmetic; stored star itself remains an approximation.',
    cost=dict(dense_product_upper_flops=2147483648, removed_factor_cost_not_subtracted=True,
        actual_paid_bill_required=True, no_extra_global_arena=True),
    prior_related_failures_remain_closed=True,
    prior_context='All-layer rank-one consumer residual repair failed CI; spherical scalar trace completion failed N2. This is a full terminal operator replacement, not either correction or a retuned coefficient.',
    duplicate_review='Bounded existing star study filenames/protocols and analyst RESULT sections32/36: no exact terminal full-star study found.',
    roster=[35,100], fit_or_sweep=False,
    native2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1.),
    scope='native2_only', boundary_authorized=False, broad_authorized=False, serving_qualified=False,
    numpy_torch_or_model_executed=False)
once(D/'protocol.json', protocol)
build = dict(candidate=pin(D/'candidate.py'), parent=pin(SOURCE), builder=pin(__file__),
    fragment=pin(D/'fragment.py'), protocol=pin(D/'protocol.json'), source_inverse_exact=True,
    exact_parent_prefix=True, original_statements_ast_exact=True,
    appended_override='_flms_star_diagonal', sole_original_caller=callers,
    shape_dtype_weight_orientation_reduction_ast_exact=True, numpy_torch_or_model_executed=False,
    serving_qualified=False, submitted=False)
once(D/'build.json', build)
base_config=json.loads(TEMPLATE.read_text()); config=rewrite(copy.deepcopy(base_config))
config['arms']['candidate']['source']=build['candidate']
assert config['arms']['base']==base_config['arms']['base']
assert config['arms']['parity_parent']==base_config['arms']['parity_parent']
assert config['arms']['base']['source']==pin(SOURCE)
config['parity_layers']=list(range(15))
config['provenance']=[pin(path) for path in (TEMPLATE,SOURCE,D/'build.py',D/'fragment.py',D/'protocol.json',
    D/'build.json',H/'official_bank_S34_v1.json',
    H/'hill_climb_star_consumer_residual_v2/RESULT.md')]
config['interpretation']=('One root-selected terminal full-star contraction on S34. Hidden factors and terminal row roster unchanged; '
    'same weighted row reduction, all computation paid. Exact L0-14 parity; original TRAIN2 gates/resources. '
    'No fit, sweep, boundary/additional-six/broad/serving authority.')
assert config['prospective_scope']=='native2_only'
for stage in ('2','8'):
    assert config['stages'][stage]['gates']==base_config['stages'][stage]['gates']
once(D/'native_screen.json',config)
print(json.dumps(dict(candidate=build['candidate'],config=pin(D/'native_screen.json'),
    build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
