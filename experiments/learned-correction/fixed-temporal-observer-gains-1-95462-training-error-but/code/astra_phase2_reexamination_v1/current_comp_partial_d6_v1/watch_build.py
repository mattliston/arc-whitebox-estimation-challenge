"""Inert fixed-COMP/D6 source-build watch; complete original resource inverse."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE.parent/'current_comp_joint_p_covariance_v1/watch_build.py'
_OWNER_SHA = '76eb2ae785c44fcd2d9a98aab97d0107f84766ebc62b5dcb34180f3d8af96095'
_original = _OWNER.read_text()
if hashlib.sha256(_original.encode()).hexdigest() != _OWNER_SHA:
    raise RuntimeError('Frozen source-build supervisor owner changed')


def _assignment(name):
    nodes = [n for n in ast.parse(_original).body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    if len(nodes) != 1:
        raise RuntimeError('Unique original supervisor assignment '+name)
    node = nodes[0]
    return ''.join(_original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


_changes = (
    (_assignment('COMMON'), 'COMMON = {}\n'),
    (_assignment('TRAIN'), 'TRAIN = {}\n'),
    ("    parser.add_argument('--run', action='store_true'); parser.add_argument('--role', choices=('zero', 'terminal'))\n"
     "    parser.add_argument('--source-sha256'); args = parser.parse_args()\n",
     "    parser.add_argument('--run', action='store_true')\n"
     "    for key in ('source','builder','protocol','spec'):\n"
     "        parser.add_argument('--'+key+'-sha256')\n"
     "    args = parser.parse_args(); args.role = 'terminal'\n"),
    ("    pins = {**FIXED, Path(__file__).resolve(): args.source_sha256,\n"
     "            HERE/'build.py': COMMON['source'], HERE/'BUILD_PROTOCOL.md': COMMON['protocol']}\n",
     "    pins = {**FIXED, Path(__file__).resolve(): args.source_sha256,\n"
     "            HERE/'build.py': args.builder_sha256, HERE/'BUILD_PROTOCOL.md': args.protocol_sha256,\n"
     "            HERE/'BUILD_WATCH_SPEC.md': args.spec_sha256, _OWNER: _OWNER_SHA}\n"),
    ("out = HERE/f'build_{args.role}_watch_v1'", "out = HERE/'build_watch_v1'"),
    ('current_comp_joint_p_covariance_build_watch_v1', 'current_comp_partial_d6_build_watch_v1'),
    ("source_bindings={str(p): h for p, h in pins.items()}, maximum_combined_rss_bytes=0,",
     "source_bindings={str(p): h for p, h in pins.items()},\n"
     "        supervisor_source_proof=dict(_ADAPTER_PROOF), maximum_combined_rss_bytes=0,"),
    ("reviewed = {**COMMON, **(TRAIN if args.role == 'terminal' else {})}",
     "reviewed = dict(source=args.builder_sha256, protocol=args.protocol_sha256)"),
    ("argv = ['/Users/matt/anaconda3/bin/python', str(HERE/'build.py'), '--build', '--role', args.role]",
     "argv = ['/Users/matt/anaconda3/bin/python', '-B', str(HERE/'build.py'), '--build']"),
    ("folder = HERE/f'build_{args.role}_v1'", "folder = HERE/'build_v1'"),
    ("and r['peak_rss_bytes'] < 2_000_000_000 and r['role'] == args.role and r['parameter_updates'] == 0",
     "and r['peak_rss_bytes'] < 2_000_000_000 and r['role'] == args.role and r['parameter_updates'] == 0\n"
     "                and r['schema'] == 'current_comp_partial_d6_build_v1'\n"
     "                and r['hashes_unchanged'] and r['sources_unchanged']\n"
     "                and r['source_bindings'].get(str(HERE/'build.py')) == args.builder_sha256\n"
     "                and r['source_bindings'].get(str(HERE/'BUILD_PROTOCOL.md')) == args.protocol_sha256"),
)
_derived = _original
for _before, _after in _changes:
    if _derived.count(_before) != 1:
        raise RuntimeError('Unique fixed-composition build-watch span '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after in reversed(_changes):
    if _inverse.count(_after) != 1:
        raise RuntimeError('Unique fixed-composition build-watch inverse '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Complete source-build supervisor inverse failed')
_ADAPTER_PROOF = dict(owner=str(_OWNER), owner_sha256=_OWNER_SHA,
    full_source_inverse_verified=True, resource_supervision_unchanged=True,
    no_model_or_training_launch=True,
    derived_source_sha256=hashlib.sha256(_derived.encode()).hexdigest())
exec(compile(_derived, str(_OWNER)+':fixed-composition-build-watch', 'exec'), globals())
