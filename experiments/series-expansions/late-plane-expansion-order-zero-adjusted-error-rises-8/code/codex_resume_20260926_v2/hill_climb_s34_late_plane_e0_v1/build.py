"""Fixed S34 L10-14 plane E0 and matched GN; source-only construction."""
from pathlib import Path
import ast
import copy
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE = H / 'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
PAIR_BEFORE = """    X = _e2_A['X2']
    _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]], st['Gr'][[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
    _e2_hold.append(fnp.multiply(X, st['RR'], out=X))
"""
PAIR_AFTER = """    _late_plane_e0 = 10 <= _PHYS_LAYER[0] <= 14
    X = _e2_A['X2'][:, :1] if _late_plane_e0 else _e2_A['X2']
    if _late_plane_e0:
        _e2_hold.append(fnp.multiply(st['Fr'][[[2], [1], [2], [3], [1]]], st['Gr'][[[1], [2], [2], [1], [3]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['RR'][:1], out=X))
    else:
        _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]], st['Gr'][[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['RR'], out=X))
"""
GN_BEFORE = """    hold(fnp.multiply(g['left1'], g['right1'], out=g['H']))
    hold(fnp.multiply(rho, g['H'], out=g['H']))
    hold(fnp.add(g['B'], g['H'], out=g['B']))
"""
GN_AFTER = """    if not 10 <= _PHYS_LAYER[0] <= 14:
        hold(fnp.multiply(g['left1'], g['right1'], out=g['H']))
        hold(fnp.multiply(rho, g['H'], out=g['H']))
        hold(fnp.add(g['B'], g['H'], out=g['B']))
"""

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
candidate = source
edits = [dict(label='late_pair_r0', before=PAIR_BEFORE, after=PAIR_AFTER),
         dict(label='matched_late_gn_r0', before=GN_BEFORE, after=GN_AFTER)]
for edit in edits:
    assert candidate.count(edit['before']) == 1
    candidate = candidate.replace(edit['before'], edit['after'], 1)
inverse = candidate
for edit in reversed(edits):
    assert inverse.count(edit['after']) == 1
    inverse = inverse.replace(edit['after'], edit['before'], 1)
assert inverse == source
old_tree, new_tree = ast.parse(source), ast.parse(candidate)
assert len(old_tree.body) == len(new_tree.body)
changed = []
for old, new in zip(old_tree.body, new_tree.body):
    if ast.dump(old) != ast.dump(new):
        assert isinstance(old, ast.FunctionDef) and isinstance(new, ast.FunctionDef)
        assert old.name == new.name
        changed.append(old.name)
assert changed == ['_nonlinear_pair_batch', '_e1gn_contract']
assert candidate.count('_PHYS_LAYER[0] = l') == source.count('_PHYS_LAYER[0] = l')
with (D / 'candidate.py').open('x') as stream:
    stream.write(candidate)
protocol = dict(schema='s34-late-plane-e0-v1', root_selected=True, baseline=pin(SOURCE),
    layers=[10, 11, 12, 13, 14], gaussian_order=3, late_plane_order=0,
    other_plane_order=1, local_order=2, existing_layer_capture='_PHYS_LAYER[0]',
    learned_literals_and_dials_unchanged=True, wrappers_captures_reset_unchanged=True,
    scratch_arena_and_setup_keys_unchanged=True, coefficient_or_scope_sweep=False,
    hypothesis='Fixed late plane-order reduction may trade raw error against lower bill.',
    direct_pair_saving_flops=5 * 45 * 1024**2,
    direct_gn_saving_flops=5 * 6 * 1024**2,
    direct_total_saving_flops=5 * 51 * 1024**2,
    cost_caveat='Excludes view/gather tariffs and downstream route effects; no bank-sized gain promised.',
    related_e1_standalone_failed8=dict(adjusted_pct=-0.30813599054250185, wins=7,
        ci95_pct=[-0.6688088603749809, 0.09477072562433797]),
    successful_bundle='E1 plus YNT plus D5 in official S34; not isolated E1 validation.',
    related_failures_preserved=True, exact_e0_duplicate_found_in_bounded_check=False,
    fixture_required_before_native2=True,
    primitive_limits=dict(seconds=60, rss_bytes=1_000_000_000, added_swap_bytes=0, threads=1),
    native2_roster=[35, 100], parity_layers=list(range(10)),
    native2_gates=dict(adjusted_pct_at_most=-0.1, wins_min=1, max_per_network_regression_pct=1.0),
    scope='native2_only', boundary_authorized=False, broad_authorized=False, serving_qualified=False)
once(D / 'protocol.json', protocol)
build = dict(candidate=pin(D / 'candidate.py'), parent=pin(SOURCE), builder=pin(__file__),
    protocol=pin(D / 'protocol.json'), primitive=pin(D / 'probe.py'), edits=edits,
    source_inverse_exact=True, changed_function_asts=changed,
    unchanged_top_level_statements_exact=True, existing_layer_capture_only=True,
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
    TEMPLATE, SOURCE, D / 'build.py', D / 'probe.py', D / 'protocol.json', D / 'build.json',
    H / 'hill_climb_precision_g3_plane_e1_gnbatch_v2/RESULT.md',
    H / 'official_bank_S34_v1.json')]
config['interpretation'] = (
    'One fixed current-S34 L10-14 plane E0 pilot with matching GN. GaussianR3/localEL2, '
    'coefficients, captures and setup unchanged. Root must pass the source-bound tiny paid fixture '
    'before N2. Genuine S34 controls; exact L0-9 parity; frozen score/resource gates. '
    'No boundary, additional-six, broad or serving authority.')
assert config['prospective_scope'] == 'native2_only'
for stage in ('2', '8'):
    assert config['stages'][stage]['gates'] == base_config['stages'][stage]['gates']
once(D / 'native_screen.json', config)
print(json.dumps(dict(candidate=build['candidate'], config=pin(D / 'native_screen.json'),
                      build=pin(D / 'build.json'), protocol=pin(D / 'protocol.json'))))
