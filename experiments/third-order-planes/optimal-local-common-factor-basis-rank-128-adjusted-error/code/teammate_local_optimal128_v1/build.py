"""Build one fixed source; do not import or execute the estimator."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent


def record(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


base = H2 / 'teammate_recovery_20260927_v1/late_common192.py'
assert record(base)['sha256'] == 'e41e5d741fbfc1d99ab9b3ec61d33376ab30aec8b853793131001f083d1a927e'
candidate = HERE / 'local_optimal128.py'
assert not candidate.exists()
source = base.read_text() + '\n\n' + (HERE / 'fragment.py').read_text()
ast.parse(source)
compile(source, str(candidate), 'exec')
candidate.write_text(source)
manifest = dict(
    bank=record(H2 / 'candidate_old_both_tail_trace_bounded_v6.py'),
    component=record(H2 / 'candidate_thin_star_route_v1.py'),
    parent=record(base),
    arms=[dict(name='local_optimal128', candidate=record(candidate))],
    protocol=record(HERE / 'PROTOCOL.md'),
    scores_read=False,
)
(HERE / 'candidates.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
