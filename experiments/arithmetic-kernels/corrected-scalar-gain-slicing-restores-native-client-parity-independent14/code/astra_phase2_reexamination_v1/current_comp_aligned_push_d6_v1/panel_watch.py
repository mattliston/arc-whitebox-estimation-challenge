"""Fixed APUSH independent14 watch; same qualified CD live resource loop."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CD = _HERE.parent/'current_comp_partial_d6_v1'
_PARENT = _CD/'panel_watch_v2.py'
_PARENT_SHA = 'b9a37cc507ccfdf36ca8375d2cbfb7fb13b1b3452507501538fcfe0f9d7f7b92'
_V1_SHA = '56793bf236fb43ef363abbff7cf8d323a828715947a625ec01b4ecd1c469335f'


def _adapt(source, changes):
    changed = source
    for before, after in changes:
        if changed.count(before) != 1:
            raise RuntimeError('Unique APUSH panel-watch source span: '+before)
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('Unique APUSH panel-watch inverse span: '+after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('Whole qualified CD v2 watch restored')
    ast.parse(changed)
    return changed


if hashlib.sha256(_PARENT.read_bytes()).hexdigest() != _PARENT_SHA:
    raise RuntimeError('Qualified CD v2 supervisor source owner')
if hashlib.sha256((_CD/'panel_watch.py').read_bytes()).hexdigest() != _V1_SHA:
    raise RuntimeError('Preserved CD v1 supervisor source owner')
_original = _PARENT.read_text()
_source = _adapt(_original, [
    ('import panel_common_v2 as c\n', 'import panel_common as c\n'),
    ("OWNER = c.HERE/'panel_watch.py'", "OWNER = c.CD/'panel_watch.py'"),
    ('    return c.splice(OWNER.read_text(), [\n',
     '    return c.splice(OWNER.read_text(), [\n'
     '        ("    \'full_admission_in_retired_process\')\\n",\n'
     '         "    \'full_admission_in_retired_process\',\\n"\n'
     '         "    \'all_eligible_push_dispatches_qualified\', \'runtime_prerequisite_qualified\')\\n"),\n'),
    ("        __name__='cd_panel_v2_original_supervisor', REVIEWED_PANEL_V2_METADATA=c)",
     "        __name__='apush_panel_original_supervisor', REVIEWED_PANEL_V2_METADATA=c)"),
])
_APUSH_WATCH_SOURCE_PROOF = dict(
    parent=str(_PARENT), parent_sha256=_PARENT_SHA,
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),
    full_source_inverse_verified=True, same_fixed15_paid_contexts=True,
    same1200s_8GB_and120s_progress=True, same60s_2GB_admission=True,
    same_completed_live_sample_and_retirement_checks=True,
    same_unconditional_final_union=True, extra_generic_dispatch_flags=True,
    no_array_import_or_arithmetic=True, no_automatic_score_or_submission=True)
exec(compile(_source,str(_PARENT)+':APUSH-panel-metadata','exec'),globals())
