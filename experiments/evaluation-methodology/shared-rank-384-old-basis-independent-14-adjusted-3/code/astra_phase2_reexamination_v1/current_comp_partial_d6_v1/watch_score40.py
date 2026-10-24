"""Original complete offline-score supervisor; unchanged candidate, fixed40."""
import hashlib
from pathlib import Path
import runpy

_here = Path(__file__).resolve().parent
_parent = _here/'watch_score_v2.py'
_parent_sha = 'c2e464345e0a77da272fef90f661286fe1205da24153a122126067ad4d90ca90'
if hashlib.sha256(_parent.read_bytes()).hexdigest() != _parent_sha:
    raise RuntimeError('Passed CHECK4 score supervisor source changed')
_namespace = runpy.run_path(str(_parent), run_name='inert_passed_CHECK4_score_watch')
_original = _namespace['_derived']
_ranks = [37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234]
_changes = (
    ("'builder', 'build-protocol', 'native-watch-spec') + tuple",
     "'builder', 'build-protocol', 'native-watch-spec', 'check4-score', 'check4-watch') + tuple", 1),
    ("here/'native.py'", "here/'native40.py'", 1),
    ("here/'watch_native.py'", "here/'watch_native40.py'", 1),
    ("here/'NATIVE_PROTOCOL.md'", "here/'NATIVE40_PROTOCOL.md'", 1),
    ("'native-watch-spec': here/'NATIVE_WATCH_SPEC.md'}",
     "'native-watch-spec': here/'NATIVE40_WATCH_SPEC.md',\n"
     "        'check4-score': here/'score_v2/result.json', 'check4-watch': here/'score_watch_v2/result.json'}", 1),
    ("here/f'native_{arm}_v1/result.json'", "here/f'native40_{arm}_v1/result.json'", 1),
    ("here/f'native_{arm}_watch_v1/result.json'", "here/f'native40_{arm}_watch_v1/result.json'", 1),
    ("here/'score_v2.py'", "here/'score40.py'", 1),
    ("HERE/'score_v2.py'", "HERE/'score40.py'", 2),
    ("here/'SCORE_PROTOCOL_V2.md'", "here/'SCORE40_PROTOCOL.md'", 1),
    ('SCORE_WATCH_SPEC_V2.md', 'SCORE40_WATCH_SPEC.md', 1),
    ("here/'score_v2/predictions_and_bills_sealed.json'",
     "here/'score40_v1/predictions_and_bills_sealed.json'", 1),
    ("HERE/'score_v2/result.json'", "HERE/'score40_v1/result.json'", 1),
    ("HERE/'score_watch_v2'", "HERE/'score40_watch_v1'", 1),
    ("'current_comp_partial_d6_score_v2'", "'current_comp_partial_d6_score40_v1'", 1),
    ("'current_comp_partial_d6_score_watch_v2'", "'current_comp_partial_d6_score40_watch_v1'", 1),
    ('all12 supplied bindings', 'all14 supplied bindings', 1),
    ('[37, 39, 42, 48]', repr(_ranks), 3),
    ('All four fixed results within own offline bounds', 'All40 fixed results within own offline bounds', 1),
    ('all_five_domains_and_selectors_reverified', 'all41_domains_and_selectors_reverified', 1),
    ('exact_v1_records_summary_comparisons_replay', 'exact_CHECK4_first4_score_replay', 1),
    ("'prior_scoring_worker_passed',",
     "'prior_scoring_worker_passed', 'check4_expansion_prerequisite_passed',\n"
     "        'check4_first4_native_packets_reverified',", 1),
    ('all_five_new_outputs_verified', 'all41_new_outputs_verified', 1),
    ('All5 outputs and own bills before any truth', 'All41 outputs and own bills before any truth', 1),
    ('     "          HERE/\'watch_score.py\':\'6e2aa9b1b700e850c9099bfc885fd994f40e96e91bdd122db43402f02a5ec49f\',\\n"\n',
     '     "          HERE/\'watch_score.py\':\'6e2aa9b1b700e850c9099bfc885fd994f40e96e91bdd122db43402f02a5ec49f\',\\n"\n'
     '     "          HERE/\'watch_score_v2.py\':\''+_parent_sha+'\',\\n"\n', 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Exact fixed40 supervisor span: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Exact fixed40 supervisor inverse span: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Full passed CHECK4 supervisor inverse failed')
_proof = dict(parent=str(_parent), parent_sha256=_parent_sha,
    full_source_inverse_verified=True, complete_resource_and_cleanup_loop_unchanged=True,
    fixed_original40_roster=True, no_candidate_or_arithmetic_change=True,
    all14_actual_scorer_bindings_separate_from_supervisor_ancestors=True,
    preserved_failed_v1_watch_remains_failed=True)
exec(compile(_derived, str(_parent)+':fixed40-own-bill-score', 'exec'),
    {'__name__': __name__, '__file__': __file__})
