"""Build two fixed cohort-boundary candidates without importing the estimator."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent


def record(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


base = H2 / 'candidate_thin_star_route_v1.py'
assert record(base)['sha256'] == '25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
source = base.read_text()
arms = []
for name, schedule in [
    ('boundary_all', {7: (0, 3), 10: (4, 6), 13: (7, 9)}),
    ('boundary_late', {7: (0, 2), 10: (3, 6), 13: (7, 9)}),
]:
    cursor = 0
    for layer, (start, stop) in schedule.items():
        assert start == cursor and stop < layer - 1
        cursor = stop + 1
    footer = f'_JOIN_SCHEDULE = {schedule!r}\n'
    candidate = HERE / f'{name}.py'
    assert not candidate.exists(), candidate
    text = source + '\n\n# Fixed additional-cohort admission at unchanged joins.\n' + footer
    ast.parse(text)
    compile(text, str(candidate), 'exec')
    candidate.write_text(text)
    arms.append(dict(name=name, candidate=record(candidate), footer=footer))
manifest = dict(
    bank=record(H2 / 'candidate_old_both_tail_trace_bounded_v6.py'),
    component=record(base), arms=arms, protocol=record(HERE / 'PROTOCOL.md'),
    scores_read=False,
)
assert manifest['bank']['sha256'] == '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
(HERE / 'candidates.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
