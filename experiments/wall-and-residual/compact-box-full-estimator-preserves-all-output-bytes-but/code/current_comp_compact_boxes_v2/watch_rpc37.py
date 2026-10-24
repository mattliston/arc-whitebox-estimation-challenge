"""V2 owner-complete supervisor; same fixed BOX37 math/runtime/cost gates."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
OWNER=NEW/'current_comp_compact_boxes_v1/watch_rpc37.py'
OWNER_SHA='9eef7acd753019a23292002643e8be7c28d09d76ba0441e241ede0be71835690'
EXTRA_OWNERS={
    OWNER:OWNER_SHA,
    NEW/'current_comp_compact_boxes_v1/RPC37_PROTOCOL.md':'0b0191b3d16411012223ded70393b388961c73b8d5654c8c8c310722eb30e2ed',
    NEW/'current_comp_compact_boxes_v1/client.py':'51051da837a11cd9b769eaa43a993fa05b611cccbca4291ebf75acbea9b38e02',
    NEW/'current_comp_compact_frames_v1/client.py':'de87726e4b1feae01e2a36d586c6103e68910f0f7f316c586b75ccd7e4a30e0e',
    NEW/'current_comp_padded_push_d6_v1/client.py':'46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',
    NEW/'current_comp_compact_frames_v1/CANONICAL_BOX_SOURCE_PRICE_20260913_UTC.md':
        '0e8426b64080ae2513fd42d980f017e28afb076b0975b1591260fac4eb7f8521',
    NEW/'current_comp_compact_frames_v1/CANONICAL_BOX_SOURCE_PRICE_20260913_UTC.json':
        'fa069c8d4403b0ff4497960002ac04bcb700e98e1403f68732c824cba40df735'}


def source_adapter():
    for path,digest in EXTRA_OWNERS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Complete factual-repair/client/census source owner')
    original=OWNER.read_text()
    edits=(
        ("CLIENT: '51051da837a11cd9b769eaa43a993fa05b611cccbca4291ebf75acbea9b38e02'",
         "CLIENT: 'e815c9c82b7730598a63832855b357f2ea6d825fe16b1b8942135a1706f25622'"),
        ("'current_comp_compact_boxes37_v1'","'current_comp_compact_boxes37_v2'"),
        ("'current_comp_compact_boxes_rpc37_v1'","'current_comp_compact_boxes_rpc37_v2'"))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('One V2 client/source-version splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact V2 source adapter inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Whole original V1 supervisor restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_V1_source_inverse=True,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_resource_and_all_gates_unchanged=True)


_source,V2_SOURCE_PROOF=source_adapter()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_boxes_v2_watch_inert')
exec(compile(_source,str(Path(__file__).resolve())+':V1-version-inverse','exec'),_api)
del _source
main,derive_source=_api['main'],_api['derive_source']
main.__globals__['PINS'].update(EXTRA_OWNERS)


if __name__=='__main__':
    raise SystemExit(main())

