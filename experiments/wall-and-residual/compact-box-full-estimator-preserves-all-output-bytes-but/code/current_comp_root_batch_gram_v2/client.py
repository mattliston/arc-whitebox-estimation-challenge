"""V2 identity-source adapter of the frozen NumPy-free client entry.

Only __file__ ownership changes to this V2 directory. The admitted worker,
client and all six caller-supplied source hashes are the V2 owners.
"""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_root_batch_gram_v1/client.py'
OWNER_SHA='b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen complete V1 NumPy-free entry')
    return raw


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_root_batch_v2_client_inert')
    exec(compile(derive_source(),str(Path(__file__).resolve())+':identity-V1-source','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
