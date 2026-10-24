"""DRAFT exact NumPy-free NOISE IPC entry; the new local worker owns the law."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_cubic_innovation_noise_v1/client.py'
OWNER_SHA='01ace9b814a052c68a260ccc111232bfef4ea02996d2e767556641f95e249a3a'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact frozen NOISE NumPy-free client')
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_distinct_client_source_owner')
    exec(compile(raw,str(OWNER)+':inert-client-owner','exec'),env)
    source,parent=env['derive_source']()
    return source,dict(owner_sha256=OWNER_SHA,parent=parent,
        entire_effective_client_unchanged=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest())


def main():
    source,proof=derive_source()
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_distinct_client_inert_owner')
    exec(compile(source,str(Path(__file__).resolve())+':whole-client-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
