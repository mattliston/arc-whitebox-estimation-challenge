"""Fast metadata composition; same admitted V2 candidate and supervised pair."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE/'watch_rpc37.py'
OWNER_SHA='637ac9a1d93997adbd27f28a9916cd67a0582334322f969a619387613d27a7b9'
SPEC=HERE/'RPC37_PROTOCOL.md'
SPEC_SHA='2e82c292d58a9b98880f62a4e02b1cb5e4da5e7a1473913d6eb953d259e71e0c'


def _owner():
    for path,digest in ((OWNER,OWNER_SHA),(SPEC,SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Actual admitted V2 supervisor/spec')
    spec=importlib.util.spec_from_file_location('_compact_boxes_fast_watch_owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def derive_source():
    base=_owner()
    original,parent_proof=base.derive_source()
    source,positions=original,[]
    for before,after,count in (
        ("HERE/'watch_rpc37.py':args.source_sha256","HERE/'watch_fast_v1.py':args.source_sha256",1),
        ("HERE/'compose.py'","HERE/'compose_fast_v1.py'",2),
        ("HERE/'RPC37_PROTOCOL.md':args.protocol_sha256","HERE/'FAST_COMPOSITION_SPEC.md':args.protocol_sha256",1)):
        if source.count(before)!=count:
            raise RuntimeError('Exact fast-metadata path splice')
        for unused in range(count):
            at=source.index(before);positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact fast-metadata supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified V2 resource/gate source restored')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,
        whole_effective_V2_source_inverse=True,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        numerical_and_resource_program_unchanged=True)


def main():
    base=_owner()
    source,proof=derive_source()
    namespace=dict(base.main.__globals__,__file__=str(Path(__file__).resolve()),
                   __name__='_compact_boxes_fast_effective')
    prior_pins=dict(namespace['PINS'])
    exec(compile(source,str(Path(__file__).resolve())+':whole-V2-inverse','exec'),namespace)
    namespace['PINS'].update(prior_pins)
    namespace['PINS'].update({OWNER:OWNER_SHA,SPEC:SPEC_SHA})
    return namespace['main']()


if __name__=='__main__':
    raise SystemExit(main())

