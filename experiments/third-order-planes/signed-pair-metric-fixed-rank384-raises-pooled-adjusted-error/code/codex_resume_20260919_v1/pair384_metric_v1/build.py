"""Port only the archived metric edits into the exact submitted Lazy parent."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
A = D.parent / 'astra_phase2_reexamination_v1'
PARENT = D / 'lazy_rotation_objective_v1/candidate.py'
ORIGINAL = A / 'current_comp_shared_pair256_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw = PARENT.read_bytes()
assert sha(PARENT) == 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
fragment = (ORIGINAL / 'metric_fnp.py').read_bytes()
assert sha(ORIGINAL / 'metric_fnp.py') == '52731186b0ee2ecd932fb9b3a7a44bd23a305dd17d3afbf58f8949019df5c44e'
composer = ast.parse((ORIGINAL / 'compose.py').read_text())
derive = next(n for n in composer.body if isinstance(n, ast.FunctionDef) and n.name == 'derive_source')
substitutions = next(n for n in derive.body if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'substitutions' for t in n.targets))
edits = ast.literal_eval(substitutions.value)
assert len(edits) == 8
assert all(b'384' in before and b'256' in after for before, after in edits[:3])
tree = ast.parse(raw)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_shared_old384_push'][-1]
lines = raw.splitlines(keepends=True)
start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
original = raw[start:stop]
changed = original
positions = []
for before, after in edits[3:]:
    assert changed.count(before) == 1
    at = changed.index(before)
    positions.append((at, before, after))
    changed = changed[:at] + after + changed[at+len(before):]
assert changed.count(b'    rank = 384\n') == 1
assert b'rank = 256' not in changed
for statement in [b'        Fj = _shared_old384_mm(Qn.T, J)\n',
                  b'            factors = _slr_join_factors(layer, overlap, state, Fj)\n',
                  b'        _slr_finish_factors(state, factors)\n']:
    assert changed.count(statement) == original.count(statement) == 1
inverse = changed
for at, before, after in reversed(positions):
    assert inverse[at:at+len(after)] == after
    inverse = inverse[:at] + before + inverse[at+len(after):]
assert inverse == original
delimiter = b'\n\n# BEGIN ARCHIVED ENTRY METRIC AT UNCHANGED RANK384\n'
candidate = raw[:start] + changed + raw[stop:] + delimiter + fragment
recovered = candidate[:-len(delimiter)-len(fragment)]
recovered = recovered[:start] + inverse + recovered[start+len(changed):]
assert recovered == raw
ast.parse(candidate)
with (HERE / 'candidate.py').open('xb') as stream:
    stream.write(candidate)
with (HERE / 'metric_fnp.py').open('xb') as stream:
    stream.write(fragment)
proof = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
             parent_sha256=sha(PARENT), candidate_sha256=sha(HERE/'candidate.py'),
             fragment_sha256=sha(HERE/'metric_fnp.py'), original_composer_sha256=sha(ORIGINAL/'compose.py'),
             build_source_sha256=sha(__file__), protocol_sha256=sha(HERE/'PROTOCOL.md'),
             exact_metric_edits=5, omitted_rank_edits=3, rank=384, age=4,
             whole_source_inverse_byte_exact=True, physical_factor_projection_unchanged=True,
             lazy_schedule_unchanged=True, heads_and_all_other_source_bytes_unchanged=True,
             original_helper_bytes_unchanged=True, full_model_executed=False, truth_accessed=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(proof, stream, indent=2)
    stream.write('\n')
print(json.dumps(proof))
