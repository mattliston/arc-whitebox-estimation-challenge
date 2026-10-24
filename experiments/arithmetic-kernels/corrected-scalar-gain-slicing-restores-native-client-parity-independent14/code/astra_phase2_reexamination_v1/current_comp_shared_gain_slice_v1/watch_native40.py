"""Same owned-tree monitor for conditional current-source OpenBLAS41."""
import hashlib
from pathlib import Path
import runpy

_G4W_HERE=Path(__file__).resolve().parent
_G4W_OWNER=_G4W_HERE/'watch_native_panel.py'
_G4W_OWNER_SHA='e9b9774f6e5325b0b94725558ff2c319411575df738a1b94822fd7761a3c211b'
if hashlib.sha256(_G4W_OWNER.read_bytes()).hexdigest()!=_G4W_OWNER_SHA:
    raise RuntimeError('Frozen current OpenBLAS panel monitor owner')
_G4W_PARENT=runpy.run_path(str(_G4W_OWNER),run_name='_inert_gain_full40_watch_parent')
for _g4w_key,_g4w_value in _G4W_PARENT.items():
    if not _g4w_key.startswith('__') and not _g4w_key.startswith('_G4W'):
        globals()[_g4w_key]=_g4w_value
SEQUENCE=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234,37)
GATE_KEYS=('panel-native','panel-native-watch','panel-score','panel-score-watch')


def watch_source_owners(args):
    return {HERE/'watch_native40.py':args.source_sha256,HERE/'native40.py':args.worker_sha256,
        HERE/'NATIVE40_PROTOCOL.md':args.protocol_sha256,HERE/'NATIVE40_WATCH_SPEC.md':args.spec_sha256,
        HERE/'build.py':args.builder_sha256,HERE/'BUILD_PROTOCOL.md':args.build_protocol_sha256,
        HERE/'score_panel.py':args.panel_score_source_sha256,
        HERE/'watch_score_panel.py':args.panel_score_watch_source_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.panel_score_protocol_sha256,
        HERE/'SCORE_PANEL_WATCH_SPEC.md':args.panel_score_watch_spec_sha256,
        TRANSPORT:TRANSPORT_SHA,_G4W_OWNER:_G4W_OWNER_SHA}


def child_command(args):
    command=[str(OFFPY),'-B','-u',str(HERE/'native40.py'),'--run']
    values=dict(source=args.worker_sha256,watch=args.source_sha256,protocol=args.protocol_sha256,
        watch_spec=args.spec_sha256)
    for key in ('builder','build_protocol','build','candidate','pair','panel_score_source',
            'panel_score_watch_source','panel_score_protocol','panel_score_watch_spec','panel_native',
            'panel_native_watch','panel_score','panel_score_watch'):
        values[key]=getattr(args,key+'_sha256')
    for key,value in values.items():
        command.extend(['--'+key.replace('_','-')+'-sha256',value])
    return command


def derive_source():
    original=_G4W_PARENT['_GW_DERIVED']
    edits=(
        ('CAP, RSS, PROGRESS = 1200.,8_000_000_000,120.','CAP, RSS, PROGRESS = 3000.,8_000_000_000,120.',1),
        ("KEYS = ('source','worker','protocol','spec','builder','build-protocol','build','candidate','pair')+GATE_KEYS",
         "KEYS = ('source','worker','protocol','spec','builder','build-protocol','build','candidate','pair','panel-score-source','panel-score-watch-source','panel-score-protocol','panel-score-watch-spec')+GATE_KEYS",1),
        ("HERE/'native_panel.py'","HERE/'native40.py'",1),
        ("HERE/'native_panel_v1'","HERE/'native40_v1'",1),
        ("HERE/'native_panel_watch_v1'","HERE/'native40_watch_v1'",1),
        ("'current_comp_shared_gain_slice_native_panel_watch_v1'","'current_comp_shared_gain_slice_native40_watch_v1'",1),
        ("'current_comp_shared_gain_slice_native_panel_v1'","'current_comp_shared_gain_slice_native40_v1'",1),
        ('panel_watch_source_inverse_proof=PANEL_WATCH_SOURCE_INVERSE_PROOF',
         'native40_watch_source_inverse_proof=NATIVE40_WATCH_SOURCE_INVERSE_PROOF',1),
        ("'Whole1200s/next retired call120s'","'Whole3000s/next retired call120s'",1),
        ("index < 15 and item['call'] == index","index < 41 and item['call'] == index",1),
        ("child.wait(timeout=1) == 0 and result['retired_calls'] == 15,'All15 complete panel calls and exit0'",
         "child.wait(timeout=1) == 0 and result['retired_calls'] == 41,'All41 complete calls and exit0'",1),
        ("record['full_model_calls'] == record['retired_call_count'] == 15",
         "record['full_model_calls'] == record['retired_call_count'] == 41",1),
        ("record['panel_source_inverse_proof'] == probe.PANEL_SOURCE_INVERSE_PROOF",
         "record['native40_source_inverse_proof'] == probe.NATIVE40_SOURCE_INVERSE_PROOF",1),
        ("record['gain_slice_pair_qualified'] and record['historical_native40_selection_only']",
         "record['current_panel_objective_qualified'] and record['historical_native40_selection_only']",1),
        ("record['gain_pair35_cold_exact'] and record['gain_pair35_repeat_physics_exact']",
         "record['own_repeat37_physics_exact'] and record['current_panel150_physics_exact']",1),
    )
    changed,spans=original,[]
    for before,after,count in edits:
        if changed.count(before)!=count:
            raise RuntimeError('Exact current full40-watch metadata boundary: '+before[:100])
        pos=changed.index(before);spans.append((pos,before,after))
        changed=changed[:pos]+after+changed[pos+len(before):]
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional current full40-watch inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire current panel monitor restored byte-exact')
    return changed,dict(owner=str(_G4W_OWNER),owner_sha256=_G4W_OWNER_SHA,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_sampling_resource_progress_cleanup_and_finalization_unchanged=True,
        whole_cap_s=3000.,fixed_roster=list(SEQUENCE),actual_current_panel_objective_required=True,
        current_source150_replay_required=True,no_old_CHECK4_or_cross_backend_prefix_claim=True)


_G4W_DERIVED,NATIVE40_WATCH_SOURCE_INVERSE_PROOF=derive_source()
exec(compile(_G4W_DERIVED,str(_G4W_OWNER)+':current-source-full40-watch','exec'),globals())
