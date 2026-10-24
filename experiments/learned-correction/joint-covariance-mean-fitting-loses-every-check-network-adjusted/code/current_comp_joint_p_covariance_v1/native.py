"""Independent covariance endpoint: unchanged paid acquisition, one Q getter."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_owner = _here.parent/'current_comp_receiver_directions_v1/native.py'
_owner_sha = '0f107dcd4a6d27da4208a563f0b79c4dcfb966ab8423df3dad492c799edc4a9e'
if hashlib.sha256(_owner.read_bytes()).hexdigest() != _owner_sha:
    raise RuntimeError('Frozen completed native acquisition owner changed')
_original = _owner.read_text()
_changes = (
    ("'current_comp_receiver_directions_native_v1'", "'current_comp_joint_p_covariance_native_v1'"),
    ('OWNERS = {\n', 'OWNERS = {\n'
     "    NEW/'current_comp_receiver_directions_v1/native.py': '"+_owner_sha+"',\n"),
    ("HERE/'build_v2.py'", "HERE/'build.py'"),
    ("HERE/'BUILD_PROTOCOL_V2.md'", "HERE/'BUILD_PROTOCOL.md'"),
    ('_bdr_parameters', '_pcc_parameters'),
    ('_BDR_PARAMETERS', '_PCC_PARAMETERS'),
    ('_BDR_LITERAL', '_PCC_LITERAL'),
    ('R_empty_at_entry', 'Q_empty_at_entry'),
    ('R_ready_at_binary_entry', 'Q_ready_at_binary_entry'),
    ('receiver_R_materialization_flops', 'covariance_Q_materialization_flops'),
    ('receiver_R_initialization_owned', 'covariance_Q_initialization_owned'),
    ("receiver[0]['literal_elements'] == 256", "receiver[0]['literal_elements'] == 910"),
    ('Sole actual256-value receiver getter', 'Sole actual910-value covariance getter'),
    ('Eight original plus one R getter code identity', 'Eight original plus one Q getter code identity'),
    ('NODE A, receiver R, original zero-state', 'NODE A, covariance Q, original zero-state'),
    ("(512 if cold else 0), 'Actual once-only paid R materialization'",
     "(1820 if cold else 0), 'Actual once-only paid Q materialization'"),
    ('receiver_observer_inverse_verified', 'covariance_observer_inverse_verified'),
)
_derived = _original
_counts = []
for _before, _after in _changes:
    _count = _derived.count(_before)
    if not _count:
        raise RuntimeError('Missing covariance native source span '+_before)
    _counts.append(_count)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for (_before, _after), _count in reversed(list(zip(_changes, _counts))):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unexpected covariance native inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole native acquisition source inverse failed')
exec(compile(_derived, str(_owner)+':P-covariance-endpoint', 'exec'),
     {'__name__': __name__, '__file__': __file__})
