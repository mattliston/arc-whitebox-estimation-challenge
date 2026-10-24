"""One new covariance score plus authenticated archived control, no model work."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_joint_p_node_v1/watch_score.py'
_parent_sha = 'cca85002951bedef0891610c086a3984aaa6f887e4a772b8dc85c38701bfd071'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


if _sha(_parent) != _parent_sha:
    raise RuntimeError('Frozen bounded offline-score supervisor changed')
_original = _parent.read_text()
_changes = (
    ("('p-only', 'joint')", "('terminal',)", 1),
    ("('p_only', 'joint')", "('terminal', )", 1),
    ("    return {path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}",
     "    return {here.parent/'current_comp_joint_p_node_v1/watch_score.py': '"+_parent_sha+"',\n"
     "        **{path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}}", 1),
    ("'current_comp_joint_p_node_score_v1'", "'current_comp_joint_p_covariance_score_v1'", 1),
    ("'current_comp_joint_p_node_score_watch_v1'", "'current_comp_joint_p_covariance_score_watch_v1'", 1),
    ('all15 supplied bindings', 'all11 supplied bindings', 1),
    ("'exact_COMP_score_replay', 'repeat_excluded', 'no_native_gradient_claim'",
     "'exact_COMP_score_replay', 'repeat_excluded', 'no_native_gradient_claim',\n"
     "        'zero_prerequisite_qualified', 'exact_P_only_score_replay',\n"
     "        'matched_P_only_training_frame_verified', 'matched_P_only_reused_without_rerun'", 1),
    ('all_ten_new_outputs_verified', 'all_five_new_and_five_matched_outputs_verified', 1),
    ('All10 outputs and own bills before any truth', 'All5 new and5 matched outputs/own bills before truth', 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Exact covariance score-watch source span '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Exact covariance score-watch inverse span '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Full original offline supervisor inverse failed')
exec(compile(_derived, str(_parent)+':covariance-actual-objective', 'exec'),
     {'__name__': __name__, '__file__': __file__})
