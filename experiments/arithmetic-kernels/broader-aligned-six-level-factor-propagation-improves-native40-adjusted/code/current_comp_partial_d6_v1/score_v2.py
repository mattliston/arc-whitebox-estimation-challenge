"""Versioned saved-output replay; only source ownership/output and replay evidence change."""
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
PARENT = HERE/'score.py'
PARENT_SHA = '900d99f3dd4b195922cec27cefcfa23c002bd09e086d71c6b3eeeb7685337b03'
PRIOR_PINS = {
    'score.py': PARENT_SHA,
    'SCORE_PROTOCOL.md': 'b59c3b0c125859cbfc64d1c614f155069d08bc8160c05ac407326af4200c0dc3',
    'watch_score.py': '6e2aa9b1b700e850c9099bfc885fd994f40e96e91bdd122db43402f02a5ec49f',
    'SCORE_WATCH_SPEC.md': '68fe4181d94f3727be4f788bfb8f0b424b517b3a97896d33cd6c5ea1a2a4c969',
    'score_v1/result.json': '4c8cd4978c1fd62e5cdbc06cf2f0513ff4e1ba4be4a7f8d7d56ec6f472853ce1',
    'score_v1/completion.json': 'a4443fdeaf8f11b487488184bd91d25f5176c5fe76da6e96584e014d2a42dac2',
    'score_v1/finalization.json': 'dc1af67c380ebedfa193f78edf922b510ca86f876b4d0794f81777e77bcdec7c',
    'score_watch_v1/result.json': 'b8b24348ef07f2d83fd3284417fc23c736235a3222a55fbcf8dc8b244106293d',
    'score_watch_v1/completion.json': '3eedc922c9b602a01183a534075a61b8a04067569e1d1a877670e808f00e5a1c',
    'score_watch_v1/finalization.json': '92b8a90a21eb014c60f03dc77c5808d884c26c9ce48b8fc46d7586175963da7e',
}


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _prior_admission(g, args, admit):
    require, read, here = g['require'], g['read'], g['HERE']
    for relative, digest in PRIOR_PINS.items():
        admit.pin(here/relative, digest)
    old, end, final = (read(here/'score_v1'/name) for name in ('result.json', 'completion.json', 'finalization.json'))
    failed, failed_end, failed_final = (read(here/'score_watch_v1'/name)
        for name in ('result.json', 'completion.json', 'finalization.json'))
    require(old['schema'] == 'current_comp_partial_d6_score_v1'
        and all(old[key] is True for key in ('passed', 'completed', 'integrity_passed', 'hashes_unchanged',
            'sources_unchanged', 'all_predictions_verified_before_truth', 'exact_COMP_score_replay'))
        and all(end[key] is True and final[key] is True for key in ('passed', 'completed'))
        and end['result_sha256'] == final['result_sha256'] == PRIOR_PINS['score_v1/result.json']
        and final['completion_sha256'] == PRIOR_PINS['score_v1/completion.json']
        and 0 <= old['elapsed_s'] <= final['elapsed_s'] < 120 and 0 < old['peak_rss_bytes'] < 1_000_000_000,
        'Preserved complete v1 scoring worker and three seals, not retrospective watch qualification')
    require(failed['schema'] == 'current_comp_partial_d6_score_watch_v1'
        and failed['passed'] is False and failed['completed'] is False
        and failed_end['passed'] is False and failed_final['passed'] is False
        and failed['error'] == 'RuntimeError: Scorer owns every reviewed source/result/candidate'
        and failed_end['result_sha256'] == PRIOR_PINS['score_watch_v1/result.json']
        and failed_final['completion_sha256'] == PRIOR_PINS['score_watch_v1/completion.json']
        and failed['cleanup_passed'], 'Exact frozen supervisor postcondition failure remains failed')
    expected = dict(vars(args), source_sha256=PARENT_SHA, protocol_sha256=PRIOR_PINS['SCORE_PROTOCOL.md'])
    require(old['requested_seals'] == expected, 'Same candidate/build/native/watch/owners; only scorer source and protocol version change')
    admit.result.update(prior_scoring_worker_passed=True, prior_external_watch_passed=False,
        prior_worker_result_sha256=PRIOR_PINS['score_v1/result.json'],
        prior_failed_watch_result_sha256=PRIOR_PINS['score_watch_v1/result.json'],
        versioned_source_inverse_proof=g['V2_SOURCE_PROOF'])


def _replay_prior(g, result, records, summary, comparisons):
    old = g['read'](g['HERE']/'score_v1/result.json')
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    actual = canonical(dict(records=records, summary=summary, comparisons=comparisons))
    prior = canonical({key: old[key] for key in ('records', 'summary', 'comparisons')})
    g['require'](actual == prior and result['truth_bindings'] == old['truth_bindings'],
        'Exact canonical-JSON bytes for every v1 record/summary/comparison and identical four truth bindings')
    result.update(exact_v1_records_summary_comparisons_replay=True,
        prior_scientific_payload_sha256=hashlib.sha256(prior).hexdigest(),
        replay_scientific_payload_sha256=hashlib.sha256(actual).hexdigest(),
        no_tuning_or_new_scientific_choice=True)


_require(_sha(PARENT) == PARENT_SHA, 'Frozen v1 scorer source owner')
_parent_namespace = runpy.run_path(str(PARENT), run_name='inert_comp_partial_d6_score_v1')
_original = _parent_namespace['_source']
_pin_lines = ''.join('    HERE/'+repr(path)+': '+repr(digest)+',\n' for path, digest in PRIOR_PINS.items())
_changes = (
    ("SCHEMA = 'current_comp_partial_d6_score_v1'", "SCHEMA = 'current_comp_partial_d6_score_v2'", 1),
    ("HERE/'SCORE_PROTOCOL.md': args.protocol_sha256", "HERE/'SCORE_PROTOCOL_V2.md': args.protocol_sha256", 1),
    ("out = HERE/'score_v1'", "out = HERE/'score_v2'", 1),
    ('PINS = {\n', 'PINS = {\n'+_pin_lines, 1),
    ("    spec = importlib.util.spec_from_file_location('matched_score_stdlib_build_verifier', HERE/'build.py')",
     "    v2_prior_admission(globals(), args, admit)\n"
     "    spec = importlib.util.spec_from_file_location('matched_score_stdlib_build_verifier', HERE/'build.py')", 1),
    ("    for path, digest in {**admit.pins, **result['truth_bindings'], **result['artifacts']}.items():",
     "    v2_replay_prior(globals(), result, records, summary, comparisons)\n"
     "    for path, digest in {**admit.pins, **result['truth_bindings'], **result['artifacts']}.items():", 1),
)
_source = _original
for _before, _after, _count in _changes:
    _require(_source.count(_before) == _count, 'Exact versioned scorer source span: '+_before)
    _source = _source.replace(_before, _after)
_inverse = _source
for _before, _after, _count in reversed(_changes):
    _require(_inverse.count(_after) == _count, 'Exact versioned scorer inverse span: '+_after)
    _inverse = _inverse.replace(_after, _before)
_require(_inverse == _original, 'Complete resolved v1 scorer source inverse')
_proof = dict(parent=str(PARENT), parent_sha256=PARENT_SHA,
    resolved_parent_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),
    full_source_inverse_verified=True, original_arithmetic_and_native_admission_unchanged=True,
    original_resources_and_finalization_unchanged=True, exact_v1_scientific_replay_required=True)
SCORER_NAMESPACE = {'__name__': __name__, '__file__': __file__,
    'composition_admission': _parent_namespace['_composition_admission'],
    'ENDPOINT_SOURCE_PROOF': _parent_namespace['_proof'], 'V2_SOURCE_PROOF': _proof,
    'v2_prior_admission': _prior_admission, 'v2_replay_prior': _replay_prior}
exec(compile(_source, str(PARENT)+':ownership-replay-v2', 'exec'), SCORER_NAMESPACE)
