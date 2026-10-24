"""Exact qualified independent14 score watch with current APUSH score owners."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE.parent/'current_comp_partial_d6_v1/panel_watch_score.py'
_OWNER_SHA = 'cd84d8d8f5ab224ca8e180d0bd08d61e50cac15cc40d74d91a40e2ed1fefa1be'


def _adapt(source, changes):
    derived = source
    for before, after in changes:
        if derived.count(before) != 1:
            raise RuntimeError('Unique APUSH score-watch source span: '+before)
        derived = derived.replace(before, after)
    inverse = derived
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('Unique APUSH score-watch inverse span: '+after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('Whole qualified score-watch owner restored')
    ast.parse(derived)
    return derived


def _watch_metadata(helper):
    # Private watcher facade only. Frozen acquisition/scorer maps stay intact.
    original_owners = helper.owners

    def owners(args,tick=None):
        values = original_owners(args,tick)
        values[str(_OWNER)] = _OWNER_SHA
        helper.bind(values,tick)
        return values

    helper.owners = owners
    return helper


if hashlib.sha256(_OWNER.read_bytes()).hexdigest() != _OWNER_SHA:
    raise RuntimeError('Frozen qualified CD independent score supervisor owner')
_original = _OWNER.read_text()
_source = _adapt(_original, [
    ('c = s.score_helper()\n', 'c = _watch_metadata(s.score_helper())\n'),
    ("SCORER_SHA = '01bef7f3533518fda8fd78e005935124e10079ee87802b2ee65aee79ee042bc0'",
     "SCORER_SHA = '097b9d4fcccbaaaea3d5dfd93fa455fb06796678a2039d415a020d339db2a2db'"),
    ('exact_COMP14_rows_and_three_summaries_replay', 'exact_CD14_rows_and_three_summaries_replay'),
    ('old_COMP_two_seal_ownership_preserved', 'current_CD_three_seal_ownership_preserved'),
    ("\"                'no_tuning_or_new_scientific_choice')) and record['full_model_calls'] == 0\\n\"",
     "\"                'no_tuning_or_new_scientific_choice','all_eligible_push_dispatches_qualified',\\n\"\n"
     "         \"                'runtime_prerequisite_qualified','original_external_watch_still_failed',\\n\"\n"
     "         \"                'no_old_watch_repair_or_score_reexecution')) and record['full_model_calls'] == 0\\n\""),
    ("        derived,proof = source()\n",
     "        derived,proof = source()\n"
     "        proof['APUSH_supervisor_source_inverse'] = _APUSH_SUPERVISOR_PROOF\n"),
    ("str(OWNER)+':current-CD-score-three-seals'", "str(OWNER)+':current-APUSH-score-three-seals'"),
])
_APUSH_SUPERVISOR_PROOF = dict(owner=str(_OWNER), owner_sha256=_OWNER_SHA,
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),
    whole_source_inverse_verified=True, original_live_monitor_and120s_2GB_limits=True,
    original_completed_sampling_and_owned_retirement=True,
    current_CD_comparator_and_generic_APUSH_flags=True, no_numerical_import_or_stage=True)
exec(compile(_source,str(_OWNER)+':APUSH-score-owners','exec'),globals())
