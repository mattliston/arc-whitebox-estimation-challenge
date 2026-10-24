"""Conditional fixed15 OpenBLAS panel; unchanged1200s sampled resource loop."""
import hashlib
from pathlib import Path
import runpy

_OBW_HERE = Path(__file__).resolve().parent
_OBW_OWNER = _OBW_HERE/'watch_native_panel.py'
_OBW_OWNER_SHA = '30c733b47a685387050d7277072ebb996204e85f5e4fd8a5f31486f68ded1b24'
if hashlib.sha256(_OBW_OWNER.read_bytes()).hexdigest() != _OBW_OWNER_SHA:
    raise RuntimeError('Immutable completed panel supervisor owner')
_OBW_PARENT = runpy.run_path(str(_OBW_OWNER),run_name='inert_openblas_panel_watch_parent')
for _obw_key,_obw_value in _OBW_PARENT.items():
    if not _obw_key.startswith('__') and not _obw_key.startswith('_OBW'):
        globals()[_obw_key] = _obw_value
BACKEND_GATE_KEYS = ('backend-probe','backend-probe-watch')
_OBW_ORIGINAL = _OBW_PARENT['_NPW_DERIVED']
_OBW_LINEAGE = """('native40_prerequisite_seals','control_seals','APUSH_panel_seals','panel_rows',
                'fixture_bindings','production_runtime_seals','production_overlap')"""
_OBW_NEW_LINEAGE = """('native40_prerequisite_seals','control_seals','APUSH_panel_seals','panel_rows',
                'fixture_bindings','production_runtime_seals','production_overlap',
                'backend_prerequisite_seals','backend_probe_prerequisite_qualified',
                'historical_native40_backend','historical_native40_selection_only',
                'APUSH_comparator_backend','APUSH_comparator_backend_matched','old_fidelity_still_failed')"""
_OBW_EDITS = [
    ("HERE/'watch_native_panel.py'","HERE/'watch_native_panel_openblas.py'",1),
    ("HERE/'native_panel.py'","HERE/'native_panel_openblas.py'",3),
    ("HERE/'NATIVE_PANEL_PROTOCOL.md'","HERE/'NATIVE_PANEL_OPENBLAS_PROTOCOL.md'",1),
    ("HERE/'NATIVE_PANEL_WATCH_SPEC.md'","HERE/'NATIVE_PANEL_OPENBLAS_WATCH_SPEC.md'",1),
    ("HERE/'native_panel_v1'","HERE/'native_panel_openblas_v1'",1),
    ("HERE/'native_panel_watch_v1'","HERE/'native_panel_openblas_watch_v1'",1),
    ("'current_comp_shared_old384_native_panel_watch_v1'",
     "'current_comp_shared_old384_native_panel_openblas_watch_v1'",1),
    ("'current_comp_shared_old384_native_panel_v1'",
     "'current_comp_shared_old384_native_panel_openblas_v1'",1),
    ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')+GATE_KEYS",
     "KEYS = ('source','worker','protocol','spec','compose','fnp','design')+GATE_KEYS+BACKEND_GATE_KEYS",1),
    ("_NPW_OWNER:_NPW_OWNER_SHA}",
     "_NPW_OWNER:_NPW_OWNER_SHA,_OBW_OWNER:_OBW_OWNER_SHA}",1),
    (_OBW_LINEAGE,_OBW_NEW_LINEAGE,2),
    ("        for key in GATE_KEYS:\n            argv.extend(",
     "        for key in GATE_KEYS+BACKEND_GATE_KEYS:\n            argv.extend(",1),
    ("record['unaffected_prefix_checks'] == [dict(call=i,layers=5,byte_exact=True) for i in range(15)]",
     "record['unaffected_prefix_checks'] == []\n"
     "            and record['cross_backend_APUSH_prefix_equality_required'] is False\n"
     "            and record['backend_probe35_cold_exact'] and record['backend_probe35_repeat_physics_exact']",1),
    ("        result.update(checked)\n",
     "        result.update(checked,backend_attestation=record['backend_attestation'],\n"
     "            cross_backend_APUSH_prefix_equality_required=False,unaffected_prefix_checks=[])\n",1)]
_OBW_DERIVED,_obw_spans = _OBW_ORIGINAL,[]
for _obw_before,_obw_after,_obw_count in _OBW_EDITS:
    if _OBW_DERIVED.count(_obw_before) != _obw_count:
        raise RuntimeError('Exact OpenBLAS panel-watch span count: '+_obw_before[:100])
    _obw_cursor = 0
    for _ in range(_obw_count):
        _obw_pos = _OBW_DERIVED.index(_obw_before,_obw_cursor)
        _obw_spans.append((_obw_pos,_obw_before,_obw_after))
        _OBW_DERIVED = _OBW_DERIVED[:_obw_pos]+_obw_after+_OBW_DERIVED[_obw_pos+len(_obw_before):]
        _obw_cursor = _obw_pos+len(_obw_after)
_OBW_INVERSE = _OBW_DERIVED
for _obw_pos,_obw_before,_obw_after in reversed(_obw_spans):
    if _OBW_INVERSE[_obw_pos:_obw_pos+len(_obw_after)] != _obw_after:
        raise RuntimeError('Positional OpenBLAS panel-watch inverse')
    _OBW_INVERSE = _OBW_INVERSE[:_obw_pos]+_obw_before+_OBW_INVERSE[_obw_pos+len(_obw_after):]
if _OBW_INVERSE != _OBW_ORIGINAL:
    raise RuntimeError('Entire frozen effective panel supervisor restored byte-exact')
PANEL_WATCH_SOURCE_INVERSE_PROOF = dict(owner=str(_OBW_OWNER),owner_sha256=_OBW_OWNER_SHA,
    whole_source_inverse_verified=True,original_resolved_sha256=hashlib.sha256(_OBW_ORIGINAL.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_OBW_DERIVED.encode()).hexdigest(),splices=len(_obw_spans),
    original_sampling_resource_progress_cleanup_and_finalization_unchanged=True,
    whole_cap_s=1200.,fixed_roster=list(SEQUENCE),explicit_positive_backend_pair_required=True,
    historical_Accelerate40_selection_gate_retained=True,supervisor_backend_not_changed=True,
    worker_only_explicit_backend_selection=True,no_production_rerun_or_fidelity_claim=True)
exec(compile(_OBW_DERIVED,str(_OBW_OWNER)+':same-production-backend-panel-watch','exec'),globals())
