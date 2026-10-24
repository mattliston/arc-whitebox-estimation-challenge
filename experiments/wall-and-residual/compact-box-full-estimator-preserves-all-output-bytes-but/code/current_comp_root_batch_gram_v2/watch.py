"""V2 version-only whole-source adapter of the frozen V1 unit supervisor.

No new resource architecture. Exact original120s/8GB/no-swap/1s observation,
one-context lifecycle, sealed tail and operational/scientific split remain.
"""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'current_comp_root_batch_gram_v1'
V1_SOURCES={
    V1/'kernel.py':'8a32612c3df91e5c24d1baa4f9c2f33352049f07ac38a8c2be6699f1b4a1e73c',
    V1/'worker.py':'ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38',
    V1/'client.py':'b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d',
    V1/'watch.py':'49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9',
    V1/'PROTOCOL.md':'704db621eaa27697367318371c043fc641b9bd12b6c87d425c13b1fb19ce3d0f',
    V1/'WATCH_SPEC.md':'de62c9d98b2dfa03924c4281b91668acbdce7637cadff3c97e34a432026c1bac'}


def derive_adapter_source():
    for path,digest in V1_SOURCES.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Immutable complete V1 supervisor owner: '+str(path))
    original=(V1/'watch.py').read_text()
    edits=(
        ("out,root = HERE/'unit_watch_v1',HERE/'unit_v1'", "out,root = HERE/'unit_watch_v2',HERE/'unit_v2'"),
        ("'current_comp_root_batch_gram_watch_v1'","'current_comp_root_batch_gram_watch_v2'"),
        ("'current_comp_root_batch_gram_unit_v1'","'current_comp_root_batch_gram_unit_v2'"),
        ("'current_comp_root_batch_gram_unit_v1_failure_ledger'","'current_comp_root_batch_gram_unit_v2_failure_ledger'"))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('One exact V2 supervisor version splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional V2 watch inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Complete frozen V1 watch source restored')
    return source,dict(owner_sha256=V1_SOURCES[V1/'watch.py'],whole_V1_source_inverse=True,
        paths_schema_only=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,V2_SOURCE_PROOF=derive_adapter_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_root_batch_v2_watch_inert')
exec(compile(_source,str(Path(__file__).resolve())+':version-only-V1-inverse','exec'),_api)
main,derive_source=_api['main'],_api['derive_source']
del _source


if __name__=='__main__':
    raise SystemExit(main())
