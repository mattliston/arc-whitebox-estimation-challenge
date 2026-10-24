"""Freeze an append-only candidate and its source-only audit receipts."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + '\n')


bank = H2 / 'candidate_old_both_tail_trace_bounded_v6.py'
component = H2 / 'candidate_thin_star_route_v1.py'
assert pin(bank)['sha256'] == '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert pin(component)['sha256'] == '25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
prefix = component.read_bytes()
footer = (HERE / 'fragment.py').read_bytes()
candidate = HERE / 'candidate.py'
candidate.write_bytes(prefix + footer)
tree = ast.parse(candidate.read_bytes())
base_tree = ast.parse(prefix)
assert ast.dump(ast.Module(body=tree.body[:len(base_tree.body)], type_ignores=[])) == ast.dump(base_tree)
assert candidate.read_bytes()[:len(prefix)] == prefix
assert candidate.read_bytes()[len(prefix):] == footer
write('source_audit.json', dict(
    candidate=pin(candidate), component=pin(component), fragment=pin(HERE/'fragment.py'),
    exact_prefix=True, ast_prefix_identical=True, new_top_level_nodes=len(tree.body)-len(base_tree.body),
    source_weights_changed=False, estimator_imported=False, numerical_estimator_runs=0))
write('candidates.json', dict(bank=pin(bank), component=pin(component),
    protocol=pin(HERE/'PROTOCOL.md'),
    arms=[dict(name='deployed_edge_offdiag_project', candidate=pin(candidate))],
    scores_read=False))
paths = [
    H2/'RESULTS.md',
    H2/'current_covariance_pair_supervision_v1/PROTOCOL.md',
    H2/'current_covariance_pair_supervision_v1/projection.py',
    H2/'native_covariance_innovation_fit_v1/candidate.py',
    H2/'covariance_complement_feedback_v1/RESULT.md',
    H2/'covariance_supplier_feedback_v1/RESULT.md',
    H2/'terminal_contraction_actions_v1/PROTOCOL.md',
    P2/'codex_resume_20260919_v1/current_edge_nonlinear_transfer_v1/PROTOCOL.md',
    P2/'codex_resume_20260919_v1/REEXAMINATION_INDEX_20260919.md',
    P2/'fable_refit_v1/TEAM_LOG.md',
    P2/'fable_analyst_v1/RESULT.md',
]
write('prior_sources.json', dict(sources=[pin(p) for p in paths],
    scope='Primary records and deployed callsites; not every archived file.',
    same_scope_executed_found=False))
print(json.dumps(pin(candidate)))
