"""Qualified120s/8GB/no-swap supervisor with fixed roster/source-owner checks."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_native_trajectory_adjoint_v1/supervise.py'
OWNER_SHA='648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'


def derive():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact preserved independent supervisor source')
    original=raw.decode();source=original;changes=[]
    for before,after in (
        ("choices=('native','adjoint')","choices=('pair',)"),
        ("if args.phase=='native'","if args.phase=='pair'"),
        ("records[0]['partial_model_calls']==records[0]['retired_calls']==3",
         "records[0]['function_calls']==4"),
        ("'Actual N/plus/minus native calls'","'Actual original/fused pair and terminal calls'"),
        ("        worker='run_'+args.phase+'.py'",
         "        require(sha(_fusion_owner)==_fusion_owner_sha,'Preserved supervisor owner at admission')\n"
         "        worker='run_'+args.phase+'.py'"),
        ("        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')",
         "        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')\n"
         "        require(sha(_fusion_owner)==_fusion_owner_sha,'Preserved supervisor owner at finalization')\n"
         "        result['inherited_source_bindings']={str(_fusion_owner):_fusion_owner_sha}"),
    ):
        count=source.count(before)
        if count<1:raise RuntimeError('Exact block-supervisor metadata splice')
        for unused in range(count):
            at=source.index(before);changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:raise RuntimeError('Exact positional supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:raise RuntimeError('Every resource/lifecycle gate preserved')
    return source


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_pair_supervisor_effective',
        _fusion_owner=OWNER,_fusion_owner_sha=OWNER_SHA)
    exec(compile(derive(),str(Path(__file__).resolve())+':resource-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':raise SystemExit(main())
