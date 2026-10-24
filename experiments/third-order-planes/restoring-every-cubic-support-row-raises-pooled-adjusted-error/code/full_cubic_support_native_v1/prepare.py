import ast
from common import *
assert sha(PARENT) == PARENT_SHA
source = PARENT.read_text()
tree = ast.parse(source)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]
original = ast.get_source_segment(source, node)
changed = original
edits = [
    ('live_indices, alive = select(response_gate, gate_floor)',
     'live_indices, alive = _cubic_support_select(response_gate, gate_floor)'),
    ('response_indices, response_mask = select(response_score, gate_floor)',
     'response_indices, response_mask = _cubic_support_select(response_score, gate_floor)')]
for old, new in edits:
    assert changed.count(old) == 1
    changed = changed.replace(old, new, 1)
restored = changed
for old, new in edits:
    restored = restored.replace(new, old, 1)
assert restored == original
assert 'selected, births = select(fnp.minimum(g, 1 - g), gate_floor)' in changed
fragment = (HERE / 'helper.py').read_bytes() + b'\n\n' + changed.encode() + b'\n'
with (HERE / 'fragment.py').open('xb') as f:
    f.write(fragment)
candidate = PARENT.read_bytes() + b'\n\n' + fragment
ast.parse(candidate.decode())
with (HERE / 'candidate.py').open('xb') as f:
    f.write(candidate)
write(HERE / 'build.json', dict(parent=pin(PARENT), helper=pin(HERE / 'helper.py'),
    fragment=pin(HERE / 'fragment.py'), candidate=pin(HERE / 'candidate.py'),
    original_predictor_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),
    predictor_inverse_exact=True, whole_source_inverse_byte_exact=candidate[:-len(fragment)-2] == PARENT.read_bytes(),
    changed_calls=edits, unchanged_birth_call=True))
rows = []
for rank, control in [(35, 'parent_triple'), (100, 'parent_cold100')]:
    path = D / f'full_lower_multilayer_mean_observer_v1/{control}/result.json'
    r = read(path)['records'][0]
    wp = P2 / f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
    tp = P2 / f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    assert sha(wp) == r['weights_sha256'] and sha(r['output']) == r['output_sha256']
    rows.append(dict(rank=rank, weights=pin(wp), truth=pin(tp), parent_result=pin(path),
                     parent=r, prediction=pin(r['output'])))
assert [r['parent']['flops'] for r in rows] == [413448696909, 416264792412]
write(HERE / 'inputs.json', dict(rows=rows, arrays_not_decoded=True))
external = [PARENT, D / 'run_guarded.py', D / 'official_best_331684.json', D / 'continuation_checkpoint_v75.json',
    D / 'full_lower_moment_support_v1/PROTOCOL.md', D / 'birth05_objective_v1/NEGATIVE_RESULT.md',
    D / 'canonical_selector_response_v1/RESULT.md', D / 'terminal_full_plane_information_v1/closeout.json',
    D / 'lazy_rotation_v1/SOURCE_REVIEW.md', D / 'shared_fourth_odd_lineage_v1/NEGATIVE_RESULT.md',
    A / 'terminal_full_fourth_tree_v1/RESULT.md', A / 'EXACT_HEAVY_MIXED_BIRTH_REPRESENTATION_SOURCE_AUDIT_V1.md']
for p in HERE.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
pins = [pin(p) for p in sorted(HERE.iterdir()) if p.is_file()] + [pin(p) for p in external]
write(HERE / 'sources_sealed.json', dict(pins=pins, targets_not_decoded=True,
    public_search_sources=['https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214',
    'https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200']))
print(dict(sealed=True, candidate=pin(HERE / 'candidate.py'), source_files=len(pins)), flush=True)

