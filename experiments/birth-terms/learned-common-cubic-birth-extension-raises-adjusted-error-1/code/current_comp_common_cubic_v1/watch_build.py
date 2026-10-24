"""Fixed common-Q build authority on the unchanged external60s/2GB watch."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE.parent/'current_comp_joint_p_covariance_v1/watch_build.py'
_OWNER_SHA = '76eb2ae785c44fcd2d9a98aab97d0107f84766ebc62b5dcb34180f3d8af96095'
if hashlib.sha256(_OWNER.read_bytes()).hexdigest() != _OWNER_SHA:
    raise RuntimeError('Frozen external source-build supervisor changed')
_original = _OWNER.read_text()


def _assignment(name):
    nodes = [n for n in ast.parse(_original).body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    if len(nodes) != 1:
        raise RuntimeError('Unique frozen supervisor assignment '+name)
    node = nodes[0]
    return ''.join(_original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


_common = """COMMON = dict(
    source='a9a9ac7e38a4ef221f441aef4456e371d1db91b277389a3d205de7bcb99fc485',
    protocol='c73760023b5117da457f060ca22beb5ea41059d75f3d4a8ee2ba68638aca455a',
    model='f068071bca67d052da101b47952eff15216d6e408a485148d2f250829ee8d37d',
    model_protocol='6eff6e23216dbc2f5df30c368cb0a3d39ead2f2dc736c13ef2a1567b61d0ccba',
    qualification_source='fd92241b5342d0e7b603b87418351fa08a8c56991da4f0ffa50aa5d1a54c36a0',
    qualification_protocol='bce829f79260d9af7f7ea2e7ee4756283dc80df8a4fbd080d5e32454781e83b9',
    qualification_watcher='935eb9ffd0d040309b406121e800cabae62f6bf7fd5b2d4e0a85063c2f9a17bb',
    qualification_watch_spec='1800774bd42608104fc9d99f9fc36f94f77a12993b3d5839ee6925ae54a8ecb2',
    qualification_result='e15610a1c80b145a2e00804bc7afaf1732cf1c50430ab4f5408a1e62d1ae526d',
    qualification_watch='0b52cecdbc259b5bab2758686fab27839cfc1f19f447029ef5330d743e4762cc')
"""
_training = """TRAIN_KEYS = ('training_source', 'training_protocol', 'training_watcher', 'training_result', 'training_watch')
"""
_pin_before = """    pins = {**FIXED, Path(__file__).resolve(): args.source_sha256,
            HERE/'build.py': COMMON['source'], HERE/'BUILD_PROTOCOL.md': COMMON['protocol']}
"""
_pin_after = """    reviewed_inputs = dict(COMMON)
    if args.role == 'terminal':
        for key in TRAIN_KEYS:
            value = getattr(args, key+'_sha256')
            require(re.fullmatch('[0-9a-f]{64}', value or ''), 'Explicit future training SHA: '+key)
            reviewed_inputs[key] = value
    paths = dict(source='build.py', protocol='BUILD_PROTOCOL.md', model='model.py',
        model_protocol='MODEL_PROTOCOL.md', qualification_source='qualification.py',
        qualification_protocol='QUALIFICATION_PROTOCOL.md', qualification_watcher='watch_qualification.py',
        qualification_watch_spec='WATCH_SPEC.md', qualification_result='qualification_v1/result.json',
        qualification_watch='qualification_watch_v1/result.json', training_source='train.py',
        training_protocol='TRAIN_PROTOCOL.md', training_watcher='watch_train.py',
        training_result='train_v1/result.json', training_watch='train_watch_v1/result.json')
    pins = {**FIXED, Path(__file__).resolve(): args.source_sha256,
        _OWNER: _OWNER_SHA,
        HERE/'BUILD_WATCH_SPEC.md': 'a33e4c8f7f8136afded9aeb99e56289f5291e29da51607e107c08bad0c8ad7cd',
        **{HERE/paths[key]: value for key, value in reviewed_inputs.items()}}
"""
_changes = (
    (_assignment('COMMON'), _common),
    (_assignment('TRAIN'), _training),
    ("    parser.add_argument('--source-sha256'); args = parser.parse_args()\n",
     "    parser.add_argument('--source-sha256')\n"
     "    for key in TRAIN_KEYS:\n"
     "        parser.add_argument('--'+key.replace('_', '-')+'-sha256')\n"
     "    args = parser.parse_args()\n"),
    (_pin_before, _pin_after),
    ('current_comp_joint_p_covariance_build_watch_v1', 'current_comp_common_cubic_build_watch_v1'),
    ("source_bindings={str(p): h for p, h in pins.items()}, maximum_combined_rss_bytes=0,",
     "source_bindings={str(p): h for p, h in pins.items() if p.suffix in ('.py', '.md')},\n"
     "        input_bindings={str(p): h for p, h in pins.items() if p.suffix not in ('.py', '.md')},\n"
     "        supervisor_source_proof=dict(_ADAPTER_PROOF), maximum_combined_rss_bytes=0,"),
    ("reviewed = {**COMMON, **(TRAIN if args.role == 'terminal' else {})}",
     "reviewed = dict(reviewed_inputs)"),
    ("and r['peak_rss_bytes'] < 2_000_000_000 and r['role'] == args.role and r['parameter_updates'] == 0",
     "and r['peak_rss_bytes'] < 2_000_000_000 and r['role'] == args.role and r['parameter_updates'] == 0\n"
     "                and r['schema'] == 'current_comp_common_cubic_build_v1'\n"
     "                and r['hashes_unchanged'] and r['sources_unchanged']\n"
     "                and r['source_bindings'].get(str(HERE/'build.py')) == COMMON['source']\n"
     "                and r['source_bindings'].get(str(HERE/'BUILD_PROTOCOL.md')) == COMMON['protocol']"),
)
_derived = _original
for _before, _after in _changes:
    if _derived.count(_before) != 1:
        raise RuntimeError('Unique common-cubic build-watch source span: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after in reversed(_changes):
    if _inverse.count(_after) != 1:
        raise RuntimeError('Unique common-cubic build-watch inverse span: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Complete original build-watch source inverse failed')
_ADAPTER_PROOF = dict(owner=str(_OWNER), owner_sha256=_OWNER_SHA,
    full_source_inverse_verified=True, numerical_supervision_unchanged=True,
    explicit_future_training_authority=True,
    derived_source_sha256=hashlib.sha256(_derived.encode()).hexdigest())
exec(compile(_derived, str(_OWNER)+':common-cubic-build-watch', 'exec'), globals())
