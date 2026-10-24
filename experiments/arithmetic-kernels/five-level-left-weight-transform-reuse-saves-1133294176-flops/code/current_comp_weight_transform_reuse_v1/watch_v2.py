"""V2 paths for the preserved W5 independent resource supervisor."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE/'watch.py'
OWNER_SHA='3ee6e653ee3a604078df9321337e4e77bc3e9495f8dda72bdb06bf6f37c0c71a'
raw=OWNER.read_bytes()
if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
    raise RuntimeError('Exact frozen V1 resource supervisor')
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_w5_watch_v2_inert')
exec(compile(raw,str(Path(__file__).resolve())+':preserved-resource-owner','exec'),_api)
_previous=_api['derive_source']


def derive_source():
    original,proof=_previous()
    source=original
    changes=[]
    for before,after in (
        ("HERE/'worker.py'","HERE/'worker_v2.py'"),
        ("HERE/'client.py'","HERE/'client_v2.py'"),
        ("HERE/'unit_watch_v1'","HERE/'unit_watch_v2'"),
        ("HERE/'unit_v1'","HERE/'unit_v2'"),
    ):
        count=source.count(before)
        if not count:
            raise RuntimeError('Exact V2 supervisor filename: '+before)
        for unused in range(count):
            at=source.index(before)
            changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact V2 resource source inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Every resource gate and numerical roster preserved')
    return source,dict(parent=proof,V1_watch_sha256=OWNER_SHA,paths_only=True,
        entry_repair_not_numerical_retry=True,derived_source_sha256=hashlib.sha256(source.encode()).hexdigest())


_api['derive_source']=derive_source
main=_api['main']


if __name__=='__main__':
    raise SystemExit(main())

