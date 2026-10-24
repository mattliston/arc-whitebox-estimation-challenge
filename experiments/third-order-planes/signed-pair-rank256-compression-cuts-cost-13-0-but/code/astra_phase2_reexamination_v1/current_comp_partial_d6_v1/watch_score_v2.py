"""Versioned source-list correction; original score and watch remain frozen."""
import hashlib
from pathlib import Path
import runpy

_here = Path(__file__).resolve().parent
_parent = _here/'watch_score.py'
_parent_sha = '6e2aa9b1b700e850c9099bfc885fd994f40e96e91bdd122db43402f02a5ec49f'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Preserved failed watch source changed')
_namespace = runpy.run_path(str(_parent), run_name='inert_preserved_score_watch_v1')
_original = _namespace['_derived']
_ancestor_sha = 'cca85002951bedef0891610c086a3984aaa6f887e4a772b8dc85c38701bfd071'
_changes = (
    ("    return {here.parent/'current_comp_joint_p_node_v1/watch_score.py': '"+_ancestor_sha+"',\n"
     "        **{path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}}",
     "    return {path: getattr(args, name.replace('-', '_')+'_sha256') for name, path in paths.items()}", 1),
    ('     "          HERE.parent/\'current_comp_receiver_birth_v1/watch_qualification.py\':SCORE_OWNER_SHA,\\n"\n',
     '     "          HERE.parent/\'current_comp_receiver_birth_v1/watch_qualification.py\':SCORE_OWNER_SHA,\\n"\n'
     '     "          HERE/\'watch_score.py\':\''+_parent_sha+'\',\\n"\n'
     '     "          HERE.parent/\'current_comp_joint_p_node_v1/watch_score.py\':\''+_ancestor_sha+'\',\\n"\n', 1),
    ("here/'score.py'", "here/'score_v2.py'", 1),
    ("HERE/'score.py'", "HERE/'score_v2.py'", 2),
    ("here/'SCORE_PROTOCOL.md'", "here/'SCORE_PROTOCOL_V2.md'", 1),
    ('SCORE_WATCH_SPEC.md', 'SCORE_WATCH_SPEC_V2.md', 1),
    ("here/'score_v1/predictions_and_bills_sealed.json'", "here/'score_v2/predictions_and_bills_sealed.json'", 1),
    ("HERE/'score_v1/result.json'", "HERE/'score_v2/result.json'", 1),
    ("HERE/'score_watch_v1'", "HERE/'score_watch_v2'", 1),
    ('current_comp_partial_d6_score_v1', 'current_comp_partial_d6_score_v2', 1),
    ('current_comp_partial_d6_score_watch_v1', 'current_comp_partial_d6_score_watch_v2', 1),
    ("'all_five_domains_and_selectors_reverified', 'no_model_imported',",
     "'all_five_domains_and_selectors_reverified', 'no_model_imported',\n"
     "        'exact_v1_records_summary_comparisons_replay', 'no_tuning_or_new_scientific_choice',\n"
     "        'prior_scoring_worker_passed',", 1),
    ("'independent_panel_qualified'", "'independent_panel_qualified', 'prior_external_watch_passed'", 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Unique v2 watch source span: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unique v2 watch inverse: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Entire resolved v1 supervisor inverse failed')
exec(compile(_derived, str(_parent)+':v2-supervisor-owner-correction', 'exec'),
     {'__name__': __name__, '__file__': __file__})
