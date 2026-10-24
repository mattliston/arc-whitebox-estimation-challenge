"""Gain-slice panel with the original1200s owned-tree supervisor."""
import ast
import hashlib
from pathlib import Path
import runpy

_GW_HERE = Path(__file__).resolve().parent
_GW_SHARED = _GW_HERE.parent / 'current_comp_shared_old384_v1'
_GW_OWNER = _GW_SHARED / 'watch_native_panel_openblas.py'
_GW_OWNER_SHA = 'a497a5f181515173d97be4a50071ae68964f67a02645d879854ed8b4ac57dfbf'
if hashlib.sha256(_GW_OWNER.read_bytes()).hexdigest() != _GW_OWNER_SHA:
    raise RuntimeError('Frozen parked panel supervisor source')
_GW_PARENT = runpy.run_path(str(_GW_OWNER), run_name='_inert_gain_panel_watch_parent')
for _gw_key, _gw_value in _GW_PARENT.items():
    if not _gw_key.startswith('__') and not _gw_key.startswith('_GW'):
        globals()[_gw_key] = _gw_value
HERE, NEW, P2 = _GW_HERE, _GW_HERE.parent, _GW_HERE.parent.parent
COMPOSED_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
GATE_KEYS = ('native40','native40-watch','score40','score40-watch')


def watch_source_owners(args):
    return {HERE / 'watch_native_panel.py': args.source_sha256,
        HERE / 'native_panel.py': args.worker_sha256,
        HERE / 'NATIVE_PANEL_PROTOCOL.md': args.protocol_sha256,
        HERE / 'NATIVE_PANEL_WATCH_SPEC.md': args.spec_sha256,
        HERE / 'build.py': args.builder_sha256, HERE / 'BUILD_PROTOCOL.md': args.build_protocol_sha256,
        TRANSPORT: TRANSPORT_SHA, _GW_OWNER: _GW_OWNER_SHA}


def child_command(args):
    command = [str(OFFPY), '-B', '-u', str(HERE / 'native_panel.py'), '--run',
        '--source-sha256', args.worker_sha256, '--watch-sha256', args.source_sha256,
        '--protocol-sha256', args.protocol_sha256, '--watch-spec-sha256', args.spec_sha256,
        '--builder-sha256', args.builder_sha256, '--build-protocol-sha256', args.build_protocol_sha256,
        '--build-sha256', args.build_sha256, '--candidate-sha256', args.candidate_sha256,
        '--pair-sha256', args.pair_sha256]
    for key in GATE_KEYS:
        command.extend(['--'+key+'-sha256', getattr(args,key.replace('-','_')+'_sha256')])
    return command


def derive_source():
    original = _GW_PARENT['_OBW_DERIVED']
    main = next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    def assignment(name):
        nodes = [n for n in ast.walk(main) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == name for t in n.targets)]
        if len(nodes) != 1:
            raise RuntimeError('One original supervisor assignment: '+name)
        n = nodes[0]
        return ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    start = original.index("        require(record['schema'] == 'current_comp_shared_old384_native_panel_openblas_v1'")
    stop = original.index("        for group in ('source_bindings','input_bindings','artifacts'):",start)
    old_gate = original[start:stop]
    new_gate = """        require(record['schema'] == 'current_comp_shared_gain_slice_native_panel_v1'
            and record['panel_source_inverse_proof'] == probe.PANEL_SOURCE_INVERSE_PROOF
            and record['gain_slice_pair_qualified'] and record['historical_native40_selection_only']
            and record['current_candidate_native40_qualified'] is False
            and record['current_candidate_production_five_qualified'] is False,
            'New pair-selected measurement, never transferred full40 or production-five authority')
        for key in probe.PANEL_LINEAGE_KEYS:
            require(record[key] == result[key], 'Same admitted actual pair/history/physical inputs: '+key)
        checked = probe.verify_native(record,tick=tick)
        result.update(checked,backend_attestation=record['backend_attestation'],
            cross_backend_APUSH_prefix_equality_required=False,unaffected_prefix_checks=[])
"""
    edits = [
        (assignment('own'), '    own = watch_source_owners(args)\n', 1),
        (assignment('argv'), '        argv = child_command(args)\n', 1),
        ("        for key in GATE_KEYS+BACKEND_GATE_KEYS:\n"
         "            argv.extend(['--'+key+'-sha256',getattr(args,key.replace('-','_')+'_sha256')])",
         '        # child_command already forwards every actual source/result authority.', 1),
        (old_gate, new_gate, 1),
        (_GW_PARENT['_OBW_NEW_LINEAGE'], 'probe.PANEL_LINEAGE_KEYS', 1),
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')+GATE_KEYS+BACKEND_GATE_KEYS",
         "KEYS = ('source','worker','protocol','spec','builder','build-protocol','build','candidate','pair')+GATE_KEYS", 1),
        ("HERE/'native_panel_openblas.py'", "HERE/'native_panel.py'", 1),
        ("HERE/'native_panel_openblas_v1'", "HERE/'native_panel_v1'", 1),
        ("HERE/'native_panel_openblas_watch_v1'", "HERE/'native_panel_watch_v1'", 1),
        ("'current_comp_shared_old384_native_panel_openblas_watch_v1'",
         "'current_comp_shared_gain_slice_native_panel_watch_v1'", 1),
        ("composition = load(HERE/'compose.py','shared384_passive_composition',args.compose_sha256)",
         "composition = load(HERE/'build.py','gain_panel_passive_builder',args.builder_sha256)", 1),
        ('pins = composition.authenticate(tick)', 'pins = composition.authenticate()', 1),
        ('text,proof = composition.derive_source(tick)', 'text,proof = composition.derive_source()', 1),
        ("result.update(native40_expansion_prerequisite_passed=True,APUSH_panel_comparator_qualified=True)",
         "result.update(historical_native40_selection_only=True,current_candidate_native40_qualified=False,APUSH_panel_comparator_qualified=True)", 1),
        ("record['backend_probe35_cold_exact'] and record['backend_probe35_repeat_physics_exact']",
         "record['gain_pair35_cold_exact'] and record['gain_pair35_repeat_physics_exact']", 1),
    ]
    changed, spans = original, []
    for before, after, count in edits:
        if changed.count(before) != count:
            raise RuntimeError('Exact gain-panel watch source span: '+before[:100])
        cursor = 0
        for unused in range(count):
            pos = changed.index(before,cursor); spans.append((pos,before,after))
            changed = changed[:pos]+after+changed[pos+len(before):]; cursor = pos+len(after)
    inverse = changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Positional gain-panel watch inverse')
        inverse = inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Full parked OpenBLAS supervisor effective source restored byte-exact')
    return changed, dict(owner=str(_GW_OWNER),owner_sha256=_GW_OWNER_SHA,
        whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        source_splices=len(spans),original_sampling_resource_progress_cleanup_and_finalization_unchanged=True,
        whole_cap_s=1200.,fixed_roster=list(SEQUENCE),actual_positive_gain_pair_required=True,
        historical_four_chain_selection_unchanged=True,current_candidate_native40_qualified=False,
        no_production_five_evidence_borrowed=True)


_GW_DERIVED, PANEL_WATCH_SOURCE_INVERSE_PROOF = derive_source()
exec(compile(_GW_DERIVED,str(_GW_OWNER)+':gain-slice-panel-watch-authority','exec'),globals())
