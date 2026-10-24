"""Versioned full40 metadata repair; unchanged c2/OpenBLAS paid loop."""
import ast
import hashlib
from pathlib import Path
import textwrap

_N40V2_HERE=Path(__file__).resolve().parent
_N40V2_OWNER=_N40V2_HERE/'native40.py'
_N40V2_OWNER_SHA='630e4128eeb29478b2e53f337c26242202deeeb276dcbaf1f7fcbc0d5c193173'
_N40V2_PINS={
    _N40V2_OWNER:_N40V2_OWNER_SHA,
    _N40V2_HERE/'watch_native40.py':'454067244608c0a74bf372bd1e3d17e3ceebf78a33e267f5ed8eaa3b5014196c',
    _N40V2_HERE/'NATIVE40_PROTOCOL.md':'9906881601418253a6ccb769bf6660cbf4e408b1b2f5ef91d1fbed0fc1293235',
    _N40V2_HERE/'NATIVE40_WATCH_SPEC.md':'8c8d50284c5e5d56a3c83ef8b3376d70a4d965d69e83e79625fe928521567fae',
}
_N40V2_NEGATIVE=('fd214643a2f09961bae8e1ace8bb7d8f685935a593fe61fcbc4b60d0786d2eb2',
    'b69ffc3040166db63c370adeaf3984a2836b009721b5f88bb5ba0a2badd8f7b3',
    'a3639e0ff6786802ac6dd7ccaf7b30220c8f12107547de56b2d97d77587a8a0f')


def _n40v2_negative(inputs,tick):
    paths=[HERE/'native40_watch_v1'/name for name in ('result.json','completion.json','finalization.json')]
    for path,digest in zip(paths,_N40V2_NEGATIVE):
        require(sha(path,tick)==digest,'Exact preserved negative namespace-failure seal')
        _panel_merge(inputs,{str(path):digest})
    r,e,f=map(read,paths)
    require(all(x.get('passed') is False and x.get('completed') is False for x in (r,e,f))
        and e['result_sha256']==f['result_sha256']==_N40V2_NEGATIVE[0]
        and f['completion_sha256']==_N40V2_NEGATIVE[1]
        and r['schema']=='current_comp_shared_gain_slice_native40_watch_v1'
        and r['retired_calls']==0 and r['markers']==[] and 'child_pid' not in r and 'command' not in r
        and r['cleanup_passed'] and r['whole_owned_tree_retired'] and r['observational_groups_retired']
        and r['numerical_imports'] is False and r['truth_accessed'] is False
        and "has no attribute 'watch_sha256'" in r['error']
        and not (HERE/'native40_v1').exists(),
        'Earlier source-admission failure remains negative, zero model calls, no worker folder')
    for group in ('source_bindings','input_bindings','artifacts'):
        _panel_merge(inputs,r[group])
    return dict(zip(('result_sha256','completion_sha256','finalization_sha256'),_N40V2_NEGATIVE))


def parse_child_command(command):
    """Execute only the exact actual native-main parser prefix, with explicit argv."""
    require(command[:4]==[str(OFFPY),'-B','-u',str(HERE/'native40_v2.py')],
        'Exact future native child executable/source prefix')
    nodes=[n for n in ast.parse(_G40_DERIVED).body if isinstance(n,ast.FunctionDef) and n.name=='main']
    require(len(nodes)==1,'Unique actual native-main parser owner')
    body=nodes[0].body
    require(len(body)>4 and isinstance(body[3],ast.Assign)
        and len(body[3].targets)==1 and isinstance(body[3].targets[0],ast.Name)
        and body[3].targets[0].id=='args','Exact parser/argument/for-loop/parse prefix')
    lines=_G40_DERIVED.splitlines(keepends=True)
    original=textwrap.dedent(''.join(lines[body[0].lineno-1:body[3].end_lineno]))
    require(original.count('parser.parse_args()')==1,'Only explicit-argv parser adaptation')
    adapted=original.replace('parser.parse_args()','parser.parse_args(argv)')
    require(adapted.replace('parser.parse_args(argv)','parser.parse_args()')==original,
        'Entire actual parser prefix inverse')
    scope=dict(globals(),argv=list(command[4:]))
    exec(compile(adapted,'<exact-native40-v2-parser-prefix>','exec'),scope)
    args=scope['args']
    require(args.run and set(vars(args))=={'run',*(key+'_sha256' for key in KEYS)},
        'Only original17 native authority fields; no watcher aliases or stage calls')
    return args


def version_source():
    for path,digest in _N40V2_PINS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Immutable original full40 source: '+str(path))
    original=_N40V2_OWNER.read_text()
    edits=(
        ("HERE/'native40.py'","HERE/'native40_v2.py'",1),
        ("HERE/'watch_native40.py'","HERE/'watch_native40_v2.py'",1),
        ("HERE/'NATIVE40_PROTOCOL.md'","HERE/'NATIVE40_V2_PROTOCOL.md'",1),
        ("HERE/'NATIVE40_WATCH_SPEC.md'","HERE/'NATIVE40_V2_WATCH_SPEC.md'",1),
        ("HERE/'native40_v1'","HERE/'native40_v2'",2),
        ("'current_comp_shared_gain_slice_native40_v1'","'current_comp_shared_gain_slice_native40_v2'",2),
        ('**EXTRA_SOURCES,**_G40_SOURCE_PINS}','**EXTRA_SOURCES,**_G40_SOURCE_PINS,**_N40V2_PINS}',1),
        ("PANEL_LINEAGE_KEYS=('current_panel_prerequisite_seals'",
         "PANEL_LINEAGE_KEYS=('previous_native40_watch_seals','previous_native40_watch_still_failed','current_panel_prerequisite_seals'",1),
        ('    records,seals={},{}','    negative=_n40v2_negative(inputs,tick)\n    records,seals={},{}',1),
        ('return dict(current_panel_prerequisite_seals=seals,controls=controls,control_seals=control_seals,',
         'return dict(previous_native40_watch_seals=negative,previous_native40_watch_still_failed=True,\n'
         '        current_panel_prerequisite_seals=seals,controls=controls,control_seals=control_seals,',1),
        ('return dict(operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,',
         'return dict(native40_version_source_proof=NATIVE40_VERSION_SOURCE_PROOF,\n'
         '        previous_native40_watch_still_failed=True,operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,',1),
    )
    changed,spans=original,[]
    for before,after,count in edits:
        if changed.count(before)!=count:
            raise RuntimeError('Exact native-v2 metadata boundary: '+repr((before,changed.count(before),count)))
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional native-v2 source inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire frozen native-v1 source restored byte-exact')
    return changed,dict(owner_sha256=_N40V2_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        only_version_authority_negative_preservation_and_parser_metadata_changed=True,
        numerical_paid_loop_and_resource_gates_unchanged=True,negative_v1_not_relabelled=True)


_N40V2_SOURCE,NATIVE40_VERSION_SOURCE_PROOF=version_source()
exec(compile(_N40V2_SOURCE,str(_N40V2_OWNER)+':versioned-current-authority','exec'),globals())
