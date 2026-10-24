"""Fresh output tag only; the fixed two-call real-client body is unchanged."""
import hashlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE/'client.py'
OWNER_SHA = 'dbdd9d28b0bd3c15bb83e1fe8669e593aba925b576b27b8f80bb650143f72fad'


def load_client():
    raw = OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest() != OWNER_SHA:
        raise RuntimeError('Exact original fixed-pair client')
    original = raw.decode()
    old, new = 'current_comp_shared_gain_slice_pair_v1', 'current_comp_shared_gain_slice_pair_v2'
    if original.count(old) != 1 or new in original:
        raise RuntimeError('One fresh output-tag splice')
    source = original.replace(old,new)
    if source.replace(new,old).encode() != raw:
        raise RuntimeError('Full client source inverse')
    scope = dict(__name__='gain_pair_client_v2_inert',__file__=str(Path(__file__).resolve()))
    exec(compile(source,str(OWNER)+':fresh-output-tag','exec'),scope)
    scope['PINS'][OWNER] = OWNER_SHA
    return types.SimpleNamespace(**scope)


if __name__ == '__main__':
    raise SystemExit(load_client().main())
else:
    _client = load_client()
    PINS = _client.PINS
    source_and_proof = _client.source_and_proof
