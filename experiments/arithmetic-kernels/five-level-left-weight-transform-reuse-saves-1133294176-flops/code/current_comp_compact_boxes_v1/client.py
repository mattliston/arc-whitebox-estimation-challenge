"""Identity source adapter: fixed RPC37 pair and scalar binding diagnostics."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_frames_v1/client.py'
OWNER_SHA='de87726e4b1feae01e2a36d586c6103e68910f0f7f316c586b75ccd7e4a30e0e'


def _api():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen NumPy-free diagnostic client source')
    namespace=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_boxes_client_inert')
    exec(compile(raw,str(Path(__file__).resolve())+':identity-source','exec'),namespace)
    return namespace


def source_and_proof():
    source,proof=_api()['source_and_proof']()
    return source,dict(parent=proof,identity_source_owner_sha256=OWNER_SHA)


def main():
    return _api()['main']()


if __name__=='__main__':
    raise SystemExit(main())

