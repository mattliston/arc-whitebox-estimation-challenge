"""V2 file-name adapter of the unchanged NumPy-free W5 client."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_root_batch_gram_v1/client.py'
OWNER_SHA='b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d'


def main():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact NumPy-free client owner')
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_w5_client_v2_inert')
    exec(compile(raw,str(Path(__file__).resolve())+':identity-source','exec'),env)
    derive=env['derive_source']
    def fixed():
        original=derive()
        before,after=b"HERE/'worker.py'",b"HERE/'worker_v2.py'"
        if original.count(before)!=1 or after in original:
            raise RuntimeError('One actual client worker filename')
        return original.replace(before,after)
    env['derive_source']=fixed
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())

