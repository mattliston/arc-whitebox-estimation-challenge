"""Inert projected TRAIN20/fresh CHECK5 supervisor; no numerical imports."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
CAPTURE_WATCH = HERE/'capture_watch.py'
CAPTURE_WATCH_SHA = '31192d37422d81922cc06ddd17d22b8f7d3d2655cff29ea6fd93fc63d0c796dd'
CAPTURE_WORKER = HERE/'capture_worker.py'
CAPTURE_WORKER_SHA = '44d750e051c7af0590600ade0dc653fe9cc58da67a077ca2bbde16acdbf1a6ba'
_OWNER = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner():
    global _OWNER
    if _OWNER is None:
        for path,digest in ((CAPTURE_WATCH,CAPTURE_WATCH_SHA),(CAPTURE_WORKER,CAPTURE_WORKER_SHA)):
            require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    'Immutable preceding capture source '+str(path))
        spec = importlib.util.spec_from_file_location('projected_capture_watch_owner',CAPTURE_WATCH)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        _OWNER = value
    return _OWNER


def rewrite(original, edits):
    """Complete positional inverse; no basename-substring reverse ambiguity."""
    changed, history = original, []
    for old,new,count in edits:
        require(old and changed.count(old) == count, 'Exact projected watcher splice '+old[:90])
        positions, cursor = [], 0
        while True:
            at = changed.find(old,cursor)
            if at < 0:
                break
            positions.append(at); cursor = at+len(old)
        pieces, cursor, shift = [], 0, 0
        for at in positions:
            pieces.extend((changed[cursor:at],new)); history.append((at+shift,old,new))
            cursor = at+len(old); shift += len(new)-len(old)
        pieces.append(changed[cursor:]); changed = ''.join(pieces)
    inverse = changed
    for at,old,new in reversed(history):
        require(inverse[at:at+len(new)] == new, 'Exact watcher inverse span')
        inverse = inverse[:at]+old+inverse[at+len(new):]
    require(inverse == original, 'Whole original capture-watch source inverse')
    return changed


def joint_source():
    """The original no-array joint-inventory helper with explicit projection metadata."""
    import ast
    value = owner(); original = CAPTURE_WATCH.read_text()
    nodes = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'seal_joint']
    require(len(nodes) == 1, 'One source-owned original inventory writer')
    n = nodes[0]
    packet = ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    changed = rewrite(packet,[
        ("HERE/f'{stage}_capture_v1'","HERE/f'{stage}_projection_capture_v1'",1),
        ("len(calls) == 200","len(calls) == 225",1),
        ('Exactly160 TRAIN plus40 CHECK call artifacts','Exactly180 TRAIN plus45 CHECK call artifacts',1),
        ("schema='current_v29_temporal_joint_capture_inventory_v1'",
         "schema='current_v29_temporal_projection_joint_capture_inventory_v1'",1),
        ("        train_complete_bills=train['trials'],check_complete_bills=record['trials'],",
         "        train_complete_bills=train['trials'],check_complete_bills=record['trials'],\n"
         "        train_projection_records=train['projection_records'],check_projection_records=record['projection_records'],\n"
         "        raw_host_semantics='raw_physical_unclipped',served_semantics='final_output_maximum_zero',\n"
         "        call_artifact_count=225,raw_arrays_per_call=64,downstream_training_authorized=False,\n"
         "        original_negative_acquisition_preserved=True,failed_projection_v1_preserved=True,",1),
    ])
    return changed,dict(owner_sha256=CAPTURE_WATCH_SHA,whole_source_inverse=True,
        original_packet_sha256=hashlib.sha256(packet.encode()).hexdigest(),
        derived_packet_sha256=hashlib.sha256(changed.encode()).hexdigest())


def derive_source():
    original,parent = owner().derive_source()
    changed = rewrite(original,[
        ("HERE/'capture_watch.py'","HERE/'projection_capture_watch.py'",1),
        ("HERE/'capture_worker.py'","HERE/'projection_capture_worker.py'",3),
        ("HERE/'CAPTURE_PROTOCOL.md'","HERE/'PROJECTION_CAPTURE_PROTOCOL.md'",1),
        ("HERE/'CAPTURE_WATCH_SPEC.md'","HERE/'PROJECTION_CAPTURE_WATCH_SPEC.md'",1),
        ("HERE/'adapter.py'","HERE/'projection_adapter.py'",2),
        ("HERE/'PROTOCOL.md':args.adapter_protocol_sha256","HERE/'PROJECTION_PROTOCOL.md':args.adapter_protocol_sha256",1),
        ("HERE/f'{args.stage}_capture_v1'","HERE/f'{args.stage}_projection_capture_v1'",1),
        ("HERE/f'{args.stage}_capture_watch_v1'","HERE/f'{args.stage}_projection_capture_watch_v1'",1),
        ("schema='current_v29_temporal_capture_watch_v1'","schema='current_v29_temporal_projection_capture_watch_v1'",1),
        ("record['schema'] == 'current_v29_temporal_capture_v1'","record['schema'] == 'current_v29_temporal_projection_capture_v1'",1),
        ('result.update(passed=True,completed=True,expanded_capture_verified=True,child_execution_completed=True,',
         'result.update(passed=True,completed=True,expanded_capture_verified=True,projected_capture_verified=True,\n'
         '            downstream_training_authorized=False,child_execution_completed=True,',1),
    ])
    packet,proof = joint_source()
    return changed,dict(parent=parent,owner_sha256=CAPTURE_WATCH_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),joint_inventory_source_proof=proof,
        stage_caps_s=dict(train=2460,check=660),actual_nine_artifacts_per_call=True,
        unchanged_live_resource_cleanup_and_three_seal_loop=True,
        unchanged_8GB_zero_swap_120s_progress_1s_actual_observations=True)


def execution_scope():
    value = owner(); packet,proof = joint_source()
    private = dict(vars(value))
    private.update(__name__='projection_capture_watch_private_metadata',__file__=str(Path(__file__).resolve()))
    exec(compile(packet,str(CAPTURE_WATCH)+':projected-joint-inventory-inverse','exec'),private)
    return dict(__name__='projection_capture_watch_inert',__file__=str(Path(__file__).resolve()),
        OWNER=value.SHARED_OWNER,OWNER_SHA=value.SHARED_SHA,
        _capture_watch=types.SimpleNamespace(**private))


if __name__ == '__main__':
    require('--run' in sys.argv, 'Inert. Explicit --run --numerical-slot-clear --stage and actual hashes required.')
    source,proof = derive_source()
    scope = execution_scope(); scope.update(__name__='__main__',_capture_watch_proof=proof)
    exec(compile(source,str(Path(__file__).resolve())+':projected-capture-watch','exec'),scope)
