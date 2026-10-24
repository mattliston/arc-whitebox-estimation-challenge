"""Exact actual native-parser namespace before versioned full40 admission."""
import hashlib
from pathlib import Path

_NW40V2_HERE=Path(__file__).resolve().parent
_NW40V2_OWNER=_NW40V2_HERE/'watch_native40.py'
_NW40V2_OWNER_SHA='454067244608c0a74bf372bd1e3d17e3ceebf78a33e267f5ed8eaa3b5014196c'
_NW40V2_PINS={_NW40V2_OWNER:_NW40V2_OWNER_SHA,
    _NW40V2_HERE/'NATIVE40_WATCH_SPEC.md':'8c8d50284c5e5d56a3c83ef8b3376d70a4d965d69e83e79625fe928521567fae'}
_NW40V2_RUNTIME_EDITS=(
    ("        panel = probe.panel_admission(args,result['input_bindings'],tick)",
     "        admitted_command = child_command(args)\n"
     "        worker_args = probe.parse_child_command(admitted_command)\n"
     "        panel = probe.panel_admission(worker_args,result['input_bindings'],tick)\n"
     "        result.update(admitted_worker_command=admitted_command,actual_worker_cli_namespace_qualified=True)",1),
    ('        argv = child_command(args)',
     "        argv = child_command(args)\n"
     "        require(argv == result['admitted_worker_command'], 'Actual spawned command exactly equals admitted native parser input')",1),
)


def version_source():
    for path,digest in _NW40V2_PINS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Immutable original negative-watch source')
    original=_NW40V2_OWNER.read_text()
    edits=(
        ("HERE/'watch_native40.py'","HERE/'watch_native40_v2.py'",1),
        ("HERE/'native40.py'","HERE/'native40_v2.py'",3),
        ("HERE/'NATIVE40_PROTOCOL.md'","HERE/'NATIVE40_V2_PROTOCOL.md'",1),
        ("HERE/'NATIVE40_WATCH_SPEC.md'","HERE/'NATIVE40_V2_WATCH_SPEC.md'",1),
        ("HERE/'native40_v1'","HERE/'native40_v2'",1),
        ("HERE/'native40_watch_v1'","HERE/'native40_watch_v2'",1),
        ("'current_comp_shared_gain_slice_native40_watch_v1'","'current_comp_shared_gain_slice_native40_watch_v2'",1),
        ("'current_comp_shared_gain_slice_native40_v1'","'current_comp_shared_gain_slice_native40_v2'",1),
        ('TRANSPORT:TRANSPORT_SHA,_G4W_OWNER:_G4W_OWNER_SHA}',
         'TRANSPORT:TRANSPORT_SHA,_G4W_OWNER:_G4W_OWNER_SHA,**_NW40V2_PINS}',1),
        ('    changed,spans=original,[]',
         '    edits=edits+_NW40V2_RUNTIME_EDITS\n    changed,spans=original,[]',1),
    )
    changed,spans=original,[]
    for before,after,count in edits:
        if changed.count(before)!=count:
            raise RuntimeError('Exact watch-v2 metadata boundary: '+repr((before,changed.count(before),count)))
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional watch-v2 source inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire frozen watch-v1 source restored byte-exact')
    return changed,dict(owner_sha256=_NW40V2_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_monitor_sampling_cleanup_and_limits_unchanged=True,
        actual_child_command_parsed_by_actual_native_main_prefix=True,old_negative_preserved=True)


_NW40V2_SOURCE,NATIVE40_WATCH_VERSION_SOURCE_PROOF=version_source()
exec(compile(_NW40V2_SOURCE,str(_NW40V2_OWNER)+':actual-native-parser-admission','exec'),globals())
