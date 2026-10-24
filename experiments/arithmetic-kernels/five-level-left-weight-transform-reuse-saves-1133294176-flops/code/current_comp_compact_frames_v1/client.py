"""NumPy-free fixed37/repeat client with scalar binding-bill diagnostics."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
OWNER=NEW/'current_comp_padded_push_d6_v1/client.py'
OWNER_SHA='46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3'
SERIES=NEW/'check_remote_series_v1.py'
SERIES_SHA='f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def source_and_proof():
    require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest()==OWNER_SHA,
            'Frozen complete fixed37 NumPy-free client')
    spec=importlib.util.spec_from_file_location('_compact_frames_fixed_client_owner',OWNER)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original,parent_proof=base.source_and_proof()
    edits=(
        ('    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f"{args.tag}-{index}") as ctx:',
         '    compact_before=(mod._CS_BINDING_FLOPS,mod._CS_BINDING_CALLS)\n'
         '    with flops.BudgetContext(flop_budget=2**41,quiet=True,namespace=f"{args.tag}-{index}") as ctx:'),
        ('                timing={key:value for key,value in summary.items() if "time" in key})',
         '                timing={key:value for key,value in summary.items() if "time" in key},\n'
         '                compact_binding_flops=mod._CS_BINDING_FLOPS-compact_before[0],\n'
         '                compact_binding_events=mod._CS_BINDING_CALLS-compact_before[1])'))
    source,positions=original,[]
    for before,after in edits:
        require(source.count(before)==1,'One exact scalar-only diagnostic splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        require(inverse[at:at+len(after)]==after,'Exact client inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original,'Entire fixed context/output arithmetic unchanged')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        full_source_inverse_verified=True,scalar_shape_counter_diagnostics_only=True,
        ranks=[37,37],actual_context_and_output_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--tag')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    if not args.candidate or not args.tag:
        parser.error('--run requires --candidate and --tag')
    require(hashlib.sha256(SERIES.read_bytes()).hexdigest()==SERIES_SHA,'Original fixture/series owner')
    source,_=source_and_proof()
    name='_current_comp_compact_frames_client_body'
    require(name not in sys.modules,'One fresh persistent client')
    runtime=types.ModuleType(name)
    runtime.__file__=str(SERIES);runtime.__package__=None
    sys.modules[name]=runtime
    saved=sys.argv
    try:
        sys.argv=[str(SERIES),'--candidate',args.candidate,'--tag',args.tag]
        exec(compile(source,str(SERIES),'exec'),vars(runtime))
    finally:
        sys.argv=saved
    return 0


if __name__=='__main__':
    raise SystemExit(main())

