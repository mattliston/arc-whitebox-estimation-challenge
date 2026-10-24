"""DRAFT: complete qualified one-RPC120s/8GB supervisor, new unit metadata only."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_scratch_v1/watch.py'
OWNER_SHA='43cc5dbef197d92c4475494a58fd26449dee5ac571c1a577142e70f4e6042088'
OWNER_SPEC=OWNER.with_name('WATCH_SPEC.md')
OWNER_SPEC_SHA='e6f4980189a8c9a267599486c8ecd6763831ecbd29a55203fddb1173f5e03c7e'


def derive_adapter_source():
    for path,digest in ((OWNER,OWNER_SHA),(OWNER_SPEC,OWNER_SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact qualified compact-unit resource owner')
    spec=importlib.util.spec_from_file_location('_streamed_compact_watch_owner',OWNER)
    base=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original,parent_proof=base.derive_adapter_source()
    source,positions=original,[]
    edits=(
        ("'current_comp_compact_scratch_watch_v1'","'current_comp_streamed_scratch_watch_v1'",1),
        ("'current_comp_compact_scratch_unit_v1'","'current_comp_streamed_scratch_unit_v1'",1),
        ("'current_comp_compact_scratch_unit_v1_failure_ledger'",
         "'current_comp_streamed_scratch_unit_v1_failure_ledger'",1),
        ('fixed_D5_Gram_only','fixed_D4_streamed_scratch_only',2),
        ('Actual unchanged original and batch scalar source proof',
         'Actual admitted canonical/streamed scalar source and lifetime proof',1))
    for before,after,count in edits:
        if source.count(before)!=count:
            raise RuntimeError('Exact streamed-watch metadata splice count: '+before)
        for unused in range(count):
            at=source.index(before);positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional streamed-watch source inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified compact-unit resource adapter restored')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,
        owner_spec_sha256=OWNER_SPEC_SHA,whole_compact_resource_adapter_inverse=True,
        numerical_fixture_not_inherited=True,resource_and_retirement_loop_unchanged=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,STREAM_SOURCE_PROOF=derive_adapter_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_streamed_scratch_watch_inert')
exec(compile(_source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),_api)
main,derive_source=_api['main'],_api['derive_source']
del _source


if __name__=='__main__':
    raise SystemExit(main())
