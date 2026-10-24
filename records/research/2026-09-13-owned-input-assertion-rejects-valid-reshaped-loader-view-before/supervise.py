"""Inherited strict resource supervisor, fixed three-call qualification roster."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_native_trajectory_adjoint_v1/supervise.py'
OWNER_SHA='648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'


def derive():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact preserved120s/8GB/no-swap supervisor source')
    original=raw.decode();source=original;changes=[]
    for before,after in (
        ("choices=('native','adjoint')","choices=('qualification',)"),
        ("records[0]['full_model_calls']==0","records[0]['full_model_calls']==3 and records[0]['backwards']==2"),
        ("'Actual target-free retired local qualifier, not a fitted estimator'",
         "'Actual target-free three-card/two-backward qualifier, not a fit'"),
        ("        worker='run_'+args.phase+'.py'",
         "        require(sha(_qualification_owner)==_qualification_owner_sha,'Supervisor owner at admission')\n"
         "        worker='run_'+args.phase+'.py'"),
        ("        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')",
         "        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')\n"
         "        require(sha(_qualification_owner)==_qualification_owner_sha,'Supervisor owner at finalization')\n"
         "        result['inherited_source_bindings']={str(_qualification_owner):_qualification_owner_sha}"),
    ):
        if source.count(before)!=1:raise RuntimeError('Exact qualification roster/admission splice')
        at=source.index(before);changes.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:raise RuntimeError('Exact supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:raise RuntimeError('All original resource/retirement/publication checks preserved')
    return source


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_qualification_supervisor_effective',
        _qualification_owner=OWNER,_qualification_owner_sha=OWNER_SHA)
    exec(compile(derive(),str(Path(__file__).resolve())+':resource-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':raise SystemExit(main())
