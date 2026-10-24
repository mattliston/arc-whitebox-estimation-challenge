"""Source-only common-Q65 adapter of the preserved qualified RD builder."""
import ast
import hashlib
import io
from pathlib import Path
import tokenize

_here = Path(__file__).resolve().parent
_rd = _here.parent/'current_comp_receiver_directions_v1'
_owner = _rd/'build_v2.py'
_owner_sha = '7d4c6ee2b5cf7ab9990569e579862a4c3648d00b36cec4d0e5d5e18ad0ebb89f'
_parent_sha = 'c844c0526e1d98d7138a8958c9a9501e9a794274222294cbc6e30dbf4073aa39'
if hashlib.sha256(_owner.read_bytes()).hexdigest() != _owner_sha:
    raise RuntimeError('Preserved tuple-compatible builder owner changed')
if hashlib.sha256((_rd/'build.py').read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Preserved first builder source changed')
# The reviewed owner defines stdlib functions only unless its --build main
# is invoked. A private non-main namespace obtains its exact derived source.
_owner_namespace = {'__file__': str(_owner), '__name__': '_cc_preserved_build_source'}
exec(compile(_owner.read_text(), str(_owner)+':cc-source-owner', 'exec'), _owner_namespace)
_original = _owner_namespace['_derived']
del _owner_namespace


def _rename_identifier(source, old, new):
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    if not any(t.type == tokenize.NAME and t.string == old for t in tokens):
        raise RuntimeError('Missing sole parameter identifier '+old)
    if any(t.type == tokenize.NAME and t.string == new for t in tokens):
        raise RuntimeError('Parameter identifier collision '+new)
    return tokenize.untokenize([t._replace(string=new)
        if t.type == tokenize.NAME and t.string == old else t for t in tokens])


def _assignment(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    if len(nodes) != 1:
        raise RuntimeError('Unique source assignment owner '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


_renamed = _rename_identifier(_original, 'R', 'Q')
if _rename_identifier(_renamed, 'Q', 'R') != _original:
    raise RuntimeError('Parameter identifier byte inverse failed')
_pins = """PINS = {
    PARENT: PARENT_SHA,
    PORT: '6b1e873939f496244d756b4129535112f2145cbbb10fd58baf3b75afff4ada58',
    HERE/'fnp.py': '3731227fbec3e551a3fdb66ae627b003cc3eb7e0c785d154aaa9c0521ceded96',
    HERE/'FNP_PROTOCOL.md': '065fe2ef59f5d981fba1b9ae6189d0c09a06e218d009ad980cce41aea63ca675',
    HERE/'DECISION.md': 'd90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80',
    JOINT/'build.py': '11e7f7588e692587c11480f6a09bd7cfb9492d02851381755821b7dbf01a9ed4',
    NEW/'current_comp_receiver_directions_v1/build_v2.py': '7d4c6ee2b5cf7ab9990569e579862a4c3648d00b36cec4d0e5d5e18ad0ebb89f',
    NEW/'current_comp_receiver_directions_v1/BUILD_PROTOCOL_V2.md': 'db2304a1dde8d07407d513474718038eef90c4dfd6d3897eb0641b5763b01aaf',
    NEW/'current_comp_receiver_directions_v1/build.py': 'c844c0526e1d98d7138a8958c9a9501e9a794274222294cbc6e30dbf4073aa39',
    NEW/'current_comp_receiver_directions_v1/BUILD_PROTOCOL.md': '767e8f39000facef13a17da5636c8750229e9bdda1f4550ed8581471ad605ca5',
    NEW/'current_comp_receiver_directions_v1/SOURCE_PREFLIGHT_V1.md': '1f1b49229c083e359196f54a417ce2b6b50dd222cf5d7670a9f6003515bcb91e',
}
"""
_atomic_write = '''def publish(path, data):
    """Atomic no-clobber publication within this exclusive build directory."""
    path = Path(path).resolve()
    require(path.parent.parent == HERE and path.parent.name in ('build_zero_v1', 'build_terminal_v1'),
            'Only this role-owned build output may be published')
    pending = path.with_name(path.name+'.pending')
    with pending.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def write(path, value):
    publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\\n').encode('utf-8'))
'''
_write_before = '''def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\\n')
'''
_changes = (
    (_assignment(_renamed, 'PINS'), _pins, 1),
    ("port_receiver_directions_working_v1.py", "port_common_cubic_working_v1.py", 1),
    ("current_comp_receiver_directions_build_v1", "current_comp_common_cubic_build_v1", 1),
    ("current_comp_receiver_directions_training_v1", "current_comp_common_cubic_training_v1", 1),
    ("current_comp_receiver_directions_qualification_v1", "current_comp_common_cubic_qualification_v1", 1),
    ("HERE/'BUILD_PROTOCOL_V2.md'", "HERE/'BUILD_PROTOCOL.md'", 3),
    ("_BDR_LITERAL", "_CC_LITERAL", None),
    ("receiver_literal", "common_literal", None),
    ("terminal_R", "terminal_Q", None),
    ("initial_R", "initial_Q", None),
    ("R_before_sha256", "Q_before_sha256", 1),
    ("q['receiver_backwards']", "q['common_backwards']", 1),
    ("q['tiny']['null_R_gradient_connected']", "q['tiny']['zero_Q_gradient_connected']", 1),
    ("q['tiny']['endogenous_H_gradient_connected'] and q['tiny']['repeated_plane_restored']",
     "q['tiny']['endogenous_H_and_physical_gradients_connected'] and q['tiny']['full_stored_pair_plane_checked']\n"
     "        and q['tiny']['null_values_exact_both_devices']\n"
     "        and all(q['tiny']['records'][device]['repeated_plane_restored'] for device in ('cpu', 'mps'))", 1),
    ("train['arm'] == 'receiver'", "train['arm'] == 'common_cubic'", 1),
    ("roster['arm'] == 'receiver'", "roster['arm'] == 'common_cubic'", 1),
    ("terminal['arm'] == validation['arm'] == 'receiver'", "terminal['arm'] == validation['arm'] == 'common_cubic'", 1),
    ("train['trainable_keys'] == ['R'] and train['trainable_parameter_count'] == 256",
     "train['trainable_keys'] == ['Q'] and train['trainable_parameter_count'] == 65", 1),
    ("meta['shape'] == [64, 4]", "meta['shape'] == [65]", 1),
    ("bytes(1024)", "bytes(260)", 2),
    ("port.compose([[0.0]*4 for _ in range(64)])", "port.compose([0.0]*65)", 1),
    ("shape=[64, 4], raw_sha256=hashlib.sha256(bytes(260))", "shape=[65], raw_sha256=hashlib.sha256(bytes(260))", 1),
    ("Q = np.zeros((64, 4), dtype=np.float32)", "Q = np.zeros((65,), dtype=np.float32)", 1),
    ("Q.shape == (64, 4)", "Q.shape == (65,)", 1),
    ("roster['seed'] == train['seed'] == 9309093", "roster['seed'] == train['seed'] == 9309101", 1),
    ("[.001+(.01-.001)*cosine]", "[.0002+(.002-.0002)*cosine]", 1),
    ("and proof['original_P_A_cstar_literals_unchanged'], 'Whole parent inverse and sole original repair')",
     "and proof['original_P_A_cstar_literals_unchanged'] and proof['original_born_static_defect_preserved']\n"
     "        and proof['original_factor_columns_and_shared_newborn_unchanged']\n"
     "        and proof['new_stored_pair_increment_applied_once'] and proof['actual_stored_d0_d1_d2_d3_preserved']\n"
     "        and proof['nonzero_diagonal_increment_preserved'] and proof['native_repair_FP64']\n"
     "        and proof['fixed_write_layers'] == [10, 11, 12, 13, 14]\n"
     "        and proof['new_parameter_values'] == 65 and proof['new_parameter_initialization_flops'] == 130,\n"
     "        'Whole parent inverse, actual stored coefficients and sole new repair')", 1),
    (_write_before, _atomic_write, 1),
    ("        with target.open('xb') as stream:\n            stream.write(candidate)\n",
     "        publish(target, candidate)\n", 1),
    ("record = dict(schema=SCHEMA, role=args.role, passed=False", 
     "record = dict(schema=SCHEMA, role=args.role, builder_adapter_source_proof=dict(_CC_BUILD_SOURCE_PROOF), passed=False", 1),
    ("and result['source_bindings'].get(str(Path(__file__).resolve())) == source_sha",
     "and result['builder_adapter_source_proof'] == _CC_BUILD_SOURCE_PROOF\n"
     "        and result['source_bindings'].get(str(Path(__file__).resolve())) == source_sha", 1),
    ("and sys.executable == '/Users/matt/anaconda3/bin/python'",
     "and sys.executable == '/Users/matt/anaconda3/bin/python'\n"
     "            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'", 1),
    ("Sole256-coordinate receiver literal", "Sole65-coordinate common-cubic literal", 1),
    ("the256-value receiver literal", "the65-value common-cubic literal", 1),
    ("qualified source-only build scope", "qualified common-cubic source-only build scope", 1),
)
_derived = _renamed
_applied = []
for _before, _after, _expected in _changes:
    _count = _derived.count(_before)
    if _count == 0 or (_expected is not None and _count != _expected):
        raise RuntimeError('Common-cubic builder source span mismatch: '+_before)
    _derived = _derived.replace(_before, _after)
    _applied.append((_before, _after, _count))
_inverse = _derived
for _before, _after, _count in reversed(_applied):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Common-cubic builder inverse span mismatch: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _renamed or _rename_identifier(_inverse, 'Q', 'R') != _original:
    raise RuntimeError('Preserved tuple-compatible builder not restored byte-exactly')
_CC_BUILD_SOURCE_PROOF = dict(owner=str(_owner), owner_sha256=_owner_sha,
    original_builder_sha256=_parent_sha, source_inverse_byte_exact=True,
    parameter_identifier_roundtrip=True, derived_source_sha256=hashlib.sha256(_derived.encode()).hexdigest(),
    source_only_Q65_adaptation=True, original_admission_resource_and_three_seal_contract=True,
    atomic_exclusive_output_publication=True)
exec(compile(_derived, str(_owner)+':common-cubic-build', 'exec'), globals())
