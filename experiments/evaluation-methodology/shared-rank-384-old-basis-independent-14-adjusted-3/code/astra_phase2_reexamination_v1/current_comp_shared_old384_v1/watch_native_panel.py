"""Conditional fixed15 panel; unchanged owned-tree monitor, deadlines and cleanup."""
import ast
import hashlib
from pathlib import Path
import runpy

_NPW_HERE = Path(__file__).resolve().parent
_NPW_OWNER = _NPW_HERE/'watch_native40.py'
_NPW_OWNER_SHA = 'c4536d3c237f393efe5cef180057b4e72650c31e2d8589411ea3bb022fabd98f'
if hashlib.sha256(_NPW_OWNER.read_bytes()).hexdigest() != _NPW_OWNER_SHA:
    raise RuntimeError('Frozen native40 supervision owner')
_NPW_PARENT = runpy.run_path(str(_NPW_OWNER),run_name='inert_native_panel_watch_parent')
for _npw_key,_npw_value in _NPW_PARENT.items():
    if not _npw_key.startswith('__') and not _npw_key.startswith('_NPW'):
        globals()[_npw_key] = _npw_value
SEQUENCE = (35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505,35)
GATE_KEYS = ('native40','native40-watch','score40','score40-watch')
_NPW_ORIGINAL = _NPW_PARENT['_N40_WATCH_DERIVED']
_npw_gate_start = _NPW_ORIGINAL.index("        require(record['schema'] == 'current_comp_shared_old384_native40_v1'")
_npw_gate_end = _NPW_ORIGINAL.index("        for group in ('source_bindings','input_bindings','artifacts'):",_npw_gate_start)
_NPW_OLD_GATE = _NPW_ORIGINAL[_npw_gate_start:_npw_gate_end]
_NPW_NEW_GATE = """        require(record['schema'] == 'current_comp_shared_old384_native_panel_v1'
            and len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 15
            and len(record['artifacts']) == 105 and record['initialization_flops'] == 1944970298
            and record['panel_source_inverse_proof'] == probe.PANEL_SOURCE_INVERSE_PROOF
            and all(record.get(k) is True for k in ('native40_expansion_prerequisite_passed',
                'APUSH_panel_comparator_qualified','panel_repeat35_physics_exact','operation_ledgers_verified',
                'operation_counts_reported_not_rebased','repeat_operation_counts_history_aware',
                'production_same_input_controls_sealed'))
            and record['production_fidelity_checked'] is False and record['production_five_qualified'] is False,
            'Actual fixed15 own-bill acquisition; production comparison remains separately pending')
        for key in ('native40_prerequisite_seals','control_seals','APUSH_panel_seals','panel_rows',
                'fixture_bindings','production_runtime_seals','production_overlap'):
            require(record[key] == result[key], 'Same admitted positive boundary/physical inputs: '+key)
        checked = probe.panel_final_checks(record,tick=tick)
        require(all(record[k] == v for k,v in checked.items()),
            'Independent actual ledgers, repeat physics and exact production/native fixture mapping')
        result.update(checked)
"""
_NPW_EDITS = [
    ('CAP, RSS, PROGRESS = 3000.,8_000_000_000,120.','CAP, RSS, PROGRESS = 1200.,8_000_000_000,120.',1),
    ("HERE/'watch_native40.py'","HERE/'watch_native_panel.py'",1),
    ("HERE/'native40.py'","HERE/'native_panel.py'",3),
    ("HERE/'NATIVE40_PROTOCOL.md'","HERE/'NATIVE_PANEL_PROTOCOL.md'",1),
    ("HERE/'NATIVE40_WATCH_SPEC.md'","HERE/'NATIVE_PANEL_WATCH_SPEC.md'",1),
    ('_N40_WATCH_OWNER:_N40_WATCH_OWNER_SHA}',
     '_N40_WATCH_OWNER:_N40_WATCH_OWNER_SHA,_NPW_OWNER:_NPW_OWNER_SHA}',1),
    ("HERE/'native40_v1',HERE/'native40_watch_v1'","HERE/'native_panel_v1',HERE/'native_panel_watch_v1'",1),
    ('current_comp_shared_old384_native40_watch_v1','current_comp_shared_old384_native_panel_watch_v1',1),
    ('native40_watch_source_inverse_proof=NATIVE40_WATCH_SOURCE_INVERSE_PROOF',
     'panel_watch_source_inverse_proof=PANEL_WATCH_SOURCE_INVERSE_PROOF',1),
    ("'Whole3000s/next retired call120s'","'Whole1200s/next retired call120s'",1),
    ("index < 41 and item['call'] == index","index < 15 and item['call'] == index",1),
    ("        prior,check4_seals = probe._n40_prerequisites(args,result['input_bindings'],tick)\n"
     "        result.update(check4_prerequisite_qualified=True,check4_prerequisite_seals=check4_seals)\n        del prior",
     "        panel = probe.panel_admission(args,result['input_bindings'],tick)\n"
     "        for key in ('native40_prerequisite_seals','control_seals','APUSH_panel_seals','panel_rows',\n"
     "                'fixture_bindings','production_runtime_seals','production_overlap'):\n"
     "            result[key] = panel[key]\n"
     "        result.update(native40_expansion_prerequisite_passed=True,APUSH_panel_comparator_qualified=True)\n"
     "        del panel",1),
    ("child.wait(timeout=1) == 0 and result['retired_calls'] == 41,'All41 complete calls and exit0'",
     "child.wait(timeout=1) == 0 and result['retired_calls'] == 15,'All15 complete panel calls and exit0'",1),
    ("record['full_model_calls'] == record['retired_call_count'] == 41",
     "record['full_model_calls'] == record['retired_call_count'] == 15",1),
    ('for i in range(41)]','for i in range(15)]',1),
    (_NPW_OLD_GATE,_NPW_NEW_GATE,1),
]
_NPW_DERIVED = _NPW_ORIGINAL
for _npw_before,_npw_after,_npw_count in _NPW_EDITS:
    if _NPW_DERIVED.count(_npw_before) != _npw_count:
        raise RuntimeError('Unique panel-watch metadata span '+_npw_before[:100])
    _NPW_DERIVED = _NPW_DERIVED.replace(_npw_before,_npw_after)
_NPW_INVERSE = _NPW_DERIVED
for _npw_before,_npw_after,_npw_count in reversed(_NPW_EDITS):
    if _NPW_INVERSE.count(_npw_after) != _npw_count:
        raise RuntimeError('Unique panel-watch inverse span '+_npw_after[:100])
    _NPW_INVERSE = _NPW_INVERSE.replace(_npw_after,_npw_before)
if _NPW_INVERSE != _NPW_ORIGINAL:
    raise RuntimeError('Entire frozen native40 resolved supervisor restored byte-exact')
ast.parse(_NPW_DERIVED)
PANEL_WATCH_SOURCE_INVERSE_PROOF = dict(owner=str(_NPW_OWNER),owner_sha256=_NPW_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_NPW_DERIVED.encode()).hexdigest(),
    original_sampling_resource_progress_cleanup_and_finalization_unchanged=True,
    whole_cap_s=1200.,fixed_roster=list(SEQUENCE),direct_positive_native40_objective_required=True,
    no_production_rerun_or_fidelity_claim=True)
exec(compile(_NPW_DERIVED,str(_NPW_OWNER)+':conditional-panel-watch-metadata','exec'),globals())
