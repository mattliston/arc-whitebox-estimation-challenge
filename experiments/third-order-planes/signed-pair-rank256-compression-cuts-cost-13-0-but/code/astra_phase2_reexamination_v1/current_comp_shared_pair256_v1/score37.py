"""Fixed saved-output rank37 scorer; no model, fit, retry or implicit stage.

The parked draft is preserved byte-exact. This outer source adapter adds only
its immutable source ownership and current independent-supervisor ownership;
the original JOINT FP64 arithmetic, .95 gate and all acquisition checks remain.
Only explicit --run can create score37_v1 or import NumPy after admission.
"""
import hashlib
from pathlib import Path

_S37_HERE = Path(__file__).resolve().parent
_S37_OWNER = _S37_HERE/'source_pre_review_v1/score37.py'
_S37_OWNER_SHA = 'dc31c824a2a78200d01a99038e2e2acb76f3e77f65039e7601692cb272e10af9'
_S37_WATCH_OWNER = _S37_HERE.parent/'current_comp_shared_old384_v1/watch_score_probe.py'
_S37_WATCH_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'


def adapter_source():
    original = _S37_OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _S37_OWNER_SHA:
        raise RuntimeError('The parked, unexecuted score37 draft is immutable')
    edits = (
        ('_PAIR_PINS = {\n',
         '_PAIR_PINS = {\n    _S37_OWNER:_S37_OWNER_SHA,\n'
         '    _S37_WATCH_OWNER:_S37_WATCH_OWNER_SHA,\n'),
        ('source_adapter_proof=PAIR_SCORE_SOURCE_PROOF,comparison_is_local_current_c2=True,',
         'source_adapter_proof=PAIR_SCORE_SOURCE_PROOF,score37_wrapper_proof=SCORE37_WRAPPER_PROOF,\n'
         '        comparison_is_local_current_c2=True,'),
        ("and result['control_candidate_sha256'] == _PAIR_CURRENT and result['source_adapter_proof'] == PAIR_SCORE_SOURCE_PROOF",
         "and result['control_candidate_sha256'] == _PAIR_CURRENT and result['source_adapter_proof'] == PAIR_SCORE_SOURCE_PROOF\n"
         "        and result['score37_wrapper_proof'] == SCORE37_WRAPPER_PROOF"),
    )
    changed,spans = original,[]
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique score37 ownership-only source splice')
        pos = changed.index(before); spans.append((pos,before,after))
        changed = changed[:pos]+after+changed[pos+len(before):]
    inverse = changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Positional score37 ownership inverse')
        inverse = inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Entire parked score37 draft must be restored')
    return changed,dict(owner_sha256=_S37_OWNER_SHA,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        no_arithmetic_gate_roster_or_acquisition_change=True,immutable_previous_draft=True)


_S37_SOURCE,SCORE37_WRAPPER_PROOF = adapter_source()
exec(compile(_S37_SOURCE,str(_S37_OWNER)+':ownership-only','exec'),globals())
