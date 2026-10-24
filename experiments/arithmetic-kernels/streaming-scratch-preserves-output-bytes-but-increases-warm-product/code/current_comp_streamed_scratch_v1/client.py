"""DRAFT identity-source reuse of the qualified NumPy-free RPC client entry."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_scratch_v1/client.py'
OWNER_SHA='1c1c0bc42e589398424af6990d63904bc2b4f206c6c9f7e6309e5dc8ed1259eb'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact qualified NumPy-free client owner')
    return raw


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_streamed_scratch_client_inert')
    exec(compile(derive_source(),str(Path(__file__).resolve())+':identity-source','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
