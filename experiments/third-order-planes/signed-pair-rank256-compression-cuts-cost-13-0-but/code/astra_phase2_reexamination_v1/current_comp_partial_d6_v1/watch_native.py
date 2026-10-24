"""Fixed COMP/partial-D6 CHECK4 watch; no numerical work on import."""
import hashlib
from pathlib import Path
import runpy

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent/'current_comp_joint_p_node_v1/watch_native.py'
_PARENT_SHA = 'd35ba7cbe406ae5e6200f8778b7a582bc6ea17336be8086b5e813e5a25dbe65e'
if hashlib.sha256(_PARENT.read_bytes()).hexdigest() != _PARENT_SHA:
    raise RuntimeError('Frozen five-call native supervisor source changed')
_parent_namespace = runpy.run_path(str(_PARENT), run_name='inert_comp_d6_native_watch_parent')
_original = _parent_namespace['_derived']
_changes = (
    ("('p_only','joint')", "('terminal',)"),
    ('args.arm', 'args.role'),
    ("'--arm'", "'--role'"),
    ("d['arm']", "d['role']"),
    ("('source','worker','model','protocol')", "('source','worker','builder','protocol','spec')"),
    ("HERE/'model.py':args.model_sha256", "HERE/'build.py':args.builder_sha256"),
    ("HERE/'NATIVE_WATCH_SPEC.md':sha(HERE/'NATIVE_WATCH_SPEC.md')",
     "HERE/'NATIVE_WATCH_SPEC.md':args.spec_sha256"),
    ("'--model-sha256',args.model_sha256", "'--builder-sha256',args.builder_sha256"),
    ("MON:MON_SHA,P1W:P1W_SHA,\n",
     "MON:MON_SHA,P1W:P1W_SHA,\n"
     "          _PARENT:_PARENT_SHA,\n"),
    ("and d['repeat_output_domain_byte_exact'],'Complete fixed native5 with owned bills')",
     "and d['repeat_output_domain_byte_exact'] and d['additional_bindings_paid_once']\n"
     "                  and d['same_existing_arena_qualified'] and d['dispatch_tariffs_qualified'],\n"
     "                  'Complete fixed native5 with own bills and serial-D6 binding/dispatch owners')"),
    ("mps_slot_clear_attested=True,numerical_imports=False,whole_cap_s=600,",
     "mps_slot_clear_attested=True,numerical_imports=False,whole_cap_s=600,\n"
     "                supervisor_source_proof=dict(_PROOF),schema='current_comp_partial_d6_native_watch_v1',"),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing fixed-composition native-watch source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unique fixed-composition native-watch inverse '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Complete resolved native supervisor inverse failed')
_PROOF = dict(parent=str(_PARENT), parent_sha256=_PARENT_SHA,
    resolved_parent_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),
    full_source_inverse_verified=True, resource_loop_and_finalization_unchanged=True,
    only_fixed_terminal5=True)
exec(compile(_derived, str(_PARENT)+':fixed-comp-d6-watch', 'exec'), globals())
