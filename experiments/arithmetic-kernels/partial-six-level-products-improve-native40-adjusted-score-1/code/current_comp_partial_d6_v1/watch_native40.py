"""Fixed unchanged COMP/partial-D6 native40; no numerical work on import."""
import hashlib
from pathlib import Path
import runpy

_here = Path(__file__).resolve().parent
_base_watch = _here/'watch_native.py'
_base_sha = '9eb726e3387c3d4bf5f4890ddea6aeb83d6b81bb9277954558dbac39f759e08a'
if hashlib.sha256(_base_watch.read_bytes()).hexdigest() != _base_sha:
    raise RuntimeError('Completed CHECK4 supervisor source changed')
_prior = runpy.run_path(str(_base_watch), run_name='inert_completed_native5_watch')
_original = _prior['_derived']
_ranks = [37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234]
_changes = (
    ("HERE/'native.py'", "HERE/'native40.py'", 2),
    ("HERE/'NATIVE_PROTOCOL.md'", "HERE/'NATIVE40_PROTOCOL.md'", 1),
    ("HERE/'NATIVE_WATCH_SPEC.md'", "HERE/'NATIVE40_WATCH_SPEC.md'", 1),
    ("('native_'+args.role+", "('native40_'+args.role+", 3),
    ("          _PARENT:_PARENT_SHA,\n", "          _PARENT:_PARENT_SHA,_BASE_WATCH:_BASE_SHA,\n", 1),
    ('600', '1900', 8),
    ("schema='current_comp_partial_d6_native_watch_v1'",
     "schema='current_comp_partial_d6_native40_watch_v1'", 1),
    ("result['retired_calls']==d['retired_call_count']==5 and d['full_model_calls']==5",
     "result['retired_calls']==d['retired_call_count']==41 and d['full_model_calls']==41", 1),
    ("d['ranks']==[37,39,42,48]", "d['ranks']=="+repr(_ranks), 1),
    ("and d['same_existing_arena_qualified'] and d['dispatch_tariffs_qualified'],",
     "and d['same_existing_arena_qualified'] and d['dispatch_tariffs_qualified']\n"
     "                  and d['check4_expansion_prerequisite_passed']\n"
     "                  and d['check4_first4_output_byte_exact'] and d['check4_first4_deterministic_bills_exact']\n"
     "                  and d['check4_first4_domains_exact'] and d['check4_first4_dispatches_exact'],", 1),
    ('Complete fixed native5 with own bills and serial-D6 binding/dispatch owners',
     'Complete unchanged native41 with own bills, CHECK4 replay and serial-D6 owners', 1),
)
_derived = _original
for _before, _after, _count in _changes:
    if _derived.count(_before) != _count:
        raise RuntimeError('Unique native40 supervisor source span: '+_before)
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after, _count in reversed(_changes):
    if _inverse.count(_after) != _count:
        raise RuntimeError('Unique native40 supervisor inverse: '+_after)
    _inverse = _inverse.replace(_after, _before)
if _inverse != _original:
    raise RuntimeError('Whole resolved CHECK4 native supervisor inverse failed')
_proof = dict(parent=str(_base_watch), parent_sha256=_base_sha,
    resolved_parent_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),
    full_source_inverse_verified=True, process_and_resource_loop_unchanged=True,
    whole1900s_original_native40_envelope=True, per_call120s_and8GB_unchanged=True,
    separately_retired_source_admission_not_retrospectively_observed=True,
    same_candidate_only_roster_expansion=True)
exec(compile(_derived, str(_base_watch)+':fixed-unchanged-native40', 'exec'),
    {'__name__': __name__, '__file__': __file__, '_PROOF': _proof,
     '_PARENT': _prior['_PARENT'], '_PARENT_SHA': _prior['_PARENT_SHA'],
     '_BASE_WATCH': _base_watch, '_BASE_SHA': _base_sha})
