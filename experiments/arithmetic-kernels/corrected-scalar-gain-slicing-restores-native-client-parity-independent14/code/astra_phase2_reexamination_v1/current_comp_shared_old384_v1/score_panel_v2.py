"""Namespace-isolation repair only; original failed independent14 score is immutable.

Import exposes stdlib metadata APIs. The original main remains inert without
--run. No array/model/target work is performed while deriving this source.
"""
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V1_OWNER = _HERE/'score_panel.py'
_V1_SHA = 'b368c905bc05e01551555fc4a59fc5639b1d7cc2d66b07756f54a69b29bba42b'
_V1_SOURCE_PINS = {
    _V1_OWNER:_V1_SHA,
    _HERE/'watch_score_panel.py':'2c3cdbcd317aad378a2eaa5c36b2420247d74abb70eeff6d401b181a037de931',
    _HERE/'SCORE_PANEL_PROTOCOL.md':'0a0f541212da7b72951caec5e31c49a157e4ba4101be8e906a331c971ead8dfb',
    _HERE/'SCORE_PANEL_WATCH_SPEC.md':'d84cfe225ed7ba2af2facbb59680f419f751b588c8f196ed7191a91ed602bd53',
}


def derive_source():
    """Whole frozen-source inverse; one shallow-copy repair and version/owner metadata."""
    for path,digest in _V1_SOURCE_PINS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Immutable original panel score owner: '+str(path))
    original = _V1_OWNER.read_text()
    edits = (
        ('    return env,dict(selected_helper_source_sha256=',
         '    return dict(env),dict(selected_helper_source_sha256=',1),
        ('PINS = {\n','PINS = {\n    **_V1_SOURCE_PINS,\n',1),
        ("'current_comp_shared_old384_score_panel_v1'","'current_comp_shared_old384_score_panel_v2'",1),
        ("HERE/'score_panel.py'","HERE/'score_panel_v2.py'",1),
        ("HERE/'watch_score_panel.py'","HERE/'watch_score_panel_v2.py'",1),
        ("HERE/'SCORE_PANEL_PROTOCOL.md'","HERE/'SCORE_PANEL_V2_PROTOCOL.md'",1),
        ("HERE/'SCORE_PANEL_WATCH_SPEC.md'","HERE/'SCORE_PANEL_V2_WATCH_SPEC.md'",1),
        ("HERE/'score_panel_v1'","HERE/'score_panel_v2'",2),
        ("HERE/'score_panel_watch_v1'","HERE/'score_panel_watch_v2'",1),
    )
    changed = original
    for old,new,count in edits:
        if changed.count(old) != count or new in changed:
            raise RuntimeError('Exact v2 scorer source boundary: '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        if inverse.count(new) != count:
            raise RuntimeError('Exact v2 scorer inverse boundary: '+new)
        inverse = inverse.replace(new,old)
    if inverse != original:
        raise RuntimeError('Entire original failed scorer must be recovered byte-exactly')
    return changed,dict(owner_sha256=_V1_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        helper_execution_globals_not_returned_for_mutation=True,
        sole_semantic_repair='support returns dict(env), preserving definitions original globals',
        numeric_body_gates_source_authority_and_resource_limits_unchanged=True,
        original_v1_sources_and_negative_receipts_unchanged=True)


_DERIVED_SOURCE, _V2_SOURCE_PROOF = derive_source()
exec(compile(_DERIVED_SOURCE,str(_V1_OWNER)+':namespace-copy-panel-v2','exec'),globals())
