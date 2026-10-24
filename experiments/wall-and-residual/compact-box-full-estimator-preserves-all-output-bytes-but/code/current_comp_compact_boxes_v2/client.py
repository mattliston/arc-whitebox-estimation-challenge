"""V2 identity-source adapter of the fixed two-context diagnostic client."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_boxes_v1/client.py'
OWNER_SHA='51051da837a11cd9b769eaa43a993fa05b611cccbca4291ebf75acbea9b38e02'


def _api():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen V1 NumPy-free client')
    namespace=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_boxes_v2_client_inert')
    exec(compile(raw,str(Path(__file__).resolve())+':identity-V1-source','exec'),namespace)
    return namespace


def source_and_proof():
    source,proof=_api()['source_and_proof']()
    return source,dict(parent=proof,identity_V1_owner_sha256=OWNER_SHA)


def main():
    return _api()['main']()


if __name__=='__main__':
    raise SystemExit(main())

