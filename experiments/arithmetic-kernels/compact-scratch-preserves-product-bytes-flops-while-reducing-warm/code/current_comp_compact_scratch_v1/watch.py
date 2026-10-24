"""Exact qualified120s/8GB/no-added-swap supervisor; storage-question schema only."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_root_batch_gram_v1/watch.py'
OWNER_SHA='49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9'


def derive_adapter_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen complete resource owner')
    original=raw.decode()
    edits=(
        ("'current_comp_root_batch_gram_watch_v1'","'current_comp_compact_scratch_watch_v1'"),
        ("'current_comp_root_batch_gram_unit_v1'","'current_comp_compact_scratch_unit_v1'"),
        ("'current_comp_root_batch_gram_unit_v1_failure_ledger'",
         "'current_comp_compact_scratch_unit_v1_failure_ledger'"))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('One exact storage-question schema splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Complete qualified supervisor restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_V1_source_inverse=True,
        schema_only=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,COMPACT_SOURCE_PROOF=derive_adapter_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_scratch_watch_inert')
exec(compile(_source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),_api)
main,derive_source=_api['main'],_api['derive_source']
del _source


if __name__=='__main__':
    raise SystemExit(main())

