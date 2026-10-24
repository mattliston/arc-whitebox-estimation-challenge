"""Version-only adapter of the unchanged strict execution supervisor."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE/'supervise.py'
OWNER_SHA='67e2f94d732007681beb917f7edce8f00fc5081c3a73948bf258d882822e5518'


def derive():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Preserved v1 source supervisor')
    spec=importlib.util.spec_from_file_location('qualification_original_supervisor',OWNER)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    original=parent.derive();source=original;changes=[]
    for before,after in (
        ("choices=('qualification',)","choices=('qualification_v2',)"),
        ("output = HERE/(args.phase+'_v1'); directory=HERE/(args.phase+'_watch_v1')",
         "output = HERE/'qualification_v2'; directory=HERE/'qualification_watch_v2'"),
        ("and 'supervise.py' in pins,","and 'supervise.py' in pins and 'supervise_v2.py' in pins,"),
    ):
        if source.count(before)!=1:raise RuntimeError('Exact version-only supervisor splice')
        at=source.index(before);changes.append((at,before,after));source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:raise RuntimeError('Exact version-only supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:raise RuntimeError('Every120s/8GB/no-swap/observation/retirement check unchanged')
    return source,parent


def main():
    source,parent=derive()
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_qualification_supervisor_v2',
        _qualification_owner=parent.OWNER,_qualification_owner_sha=parent.OWNER_SHA)
    exec(compile(source,str(Path(__file__).resolve())+':version-only','exec'),env)
    return env['main']()


if __name__=='__main__':raise SystemExit(main())
