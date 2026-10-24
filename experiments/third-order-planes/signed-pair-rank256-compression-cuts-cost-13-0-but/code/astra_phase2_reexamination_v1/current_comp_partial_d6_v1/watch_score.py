"""Fixed COMP/partial-D6 saved-output score watch; no numerical imports."""
import hashlib
from pathlib import Path

_here = Path(__file__).resolve().parent
_parent = _here.parent/'current_comp_joint_p_node_v1/watch_score.py'
_parent_sha = 'cca85002951bedef0891610c086a3984aaa6f887e4a772b8dc85c38701bfd071'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Frozen complete offline-score supervisor changed')
_original = _parent.read_text()
_changes = (
    ("('p-only', 'joint')", "('terminal',)", 1),
    ("('p_only', 'joint')", "('terminal', )", 1),
    ("'builder', 'build-protocol') + tuple", "'builder', 'build-protocol', 'native-watch-spec') + tuple", 1),
    ("'builder': here/'build.py', 'build-protocol': here/'BUILD_PROTOCOL.md'}",
     "'builder': here/'build.py', 'build-protocol': here/'BUILD_PROTOCOL.md',\n"
     "        'native-watch-spec': here/'NATIVE_WATCH_SPEC.md'}", 1),
    ("here/f'build_{arm}_v1/result.json'", "here/'build_v1/result.json'", 1),
    ("here/f'build_{arm}_v1/estimator.py'", "here/'build_v1/estimator.py'", 1),
    ("    return {path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}",
     "    return {here.parent/'current_comp_joint_p_node_v1/watch_score.py': '"+_parent_sha+"',\n"
     "        **{path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}}", 1),
    ("'current_comp_joint_p_node_score_v1'", "'current_comp_partial_d6_score_v1'", 1),
    ("'current_comp_joint_p_node_score_watch_v1'", "'current_comp_partial_d6_score_watch_v1'", 1),
    ('all15 supplied bindings', 'all12 supplied bindings', 1),
    ("'exact_COMP_score_replay', 'repeat_excluded', 'no_native_gradient_claim'",
     "'exact_COMP_score_replay', 'repeat_excluded', 'no_native_gradient_claim',\n"
     "        'complete_D6_acquisition_reverified', 'source_build_watch_qualified',\n"
     "        'all_five_domains_and_selectors_reverified', 'no_model_imported',\n"
     "        'old_max_read_timing_gate_still_failed', 'frozen_geometry_estimate_not_used_for_scoring'", 1),
    ('all_ten_new_outputs_verified', 'all_five_new_outputs_verified', 1),
    ('All10 outputs and own bills before any truth', 'All5 outputs and own bills before any truth', 1),
    ("for k in ('source','worker')+SCORE_FORWARDED:ap.add_argument('--'+k+'-sha256')",
     "for k in ('source','worker','spec')+SCORE_FORWARDED:ap.add_argument('--'+k+'-sha256')", 1),
    ("HERE/'SCORE_WATCH_SPEC.md':sha(HERE/'SCORE_WATCH_SPEC.md')",
     "HERE/'SCORE_WATCH_SPEC.md':args.spec_sha256", 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Exact fixed-composition score-watch span missing: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Exact fixed-composition score-watch inverse missing: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Full original score-watch byte inverse failed')
exec(compile(_derived, str(_parent)+':fixed-comp-d6-offline-score', 'exec'),
     {'__name__': __name__, '__file__': __file__})
