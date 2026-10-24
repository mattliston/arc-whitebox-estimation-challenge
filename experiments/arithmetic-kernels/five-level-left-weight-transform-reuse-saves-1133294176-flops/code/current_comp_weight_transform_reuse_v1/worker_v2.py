"""V2 entry repair only: original W5 arithmetic is unchanged and unexecuted."""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
V1=HERE/'worker.py'
V1_SHA='0de0944af72154196fda4c5cf40cbd206a95c8d260dd1d3d9030ab981c89e1eb'
ROOT=HERE.parent/'current_comp_root_batch_gram_v1/worker.py'
ROOT_SHA='ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38'


def load(path,digest,name):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Exact V2 lifecycle source owner')
    spec=importlib.util.spec_from_file_location(name,path)
    api=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def edit(source,edits):
    original=source
    positions=[]
    for before,after in edits:
        count=source.count(before)
        if not count:
            raise RuntimeError('One or more explicit V2 source splice: '+before[:100])
        for unused in range(count):
            at=source.index(before)
            positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('V2 inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Whole pre-repair source recovered')
    return source


_old=load(V1,V1_SHA,'w5_entry_v1_inert')
_root=load(ROOT,ROOT_SHA,'w5_original_main_inert')
_source,SOURCE_V1_PROOF=_old.derive_source()
_main,ROOT_MAIN_PROOF=_root.derive_main()
# V1 renamed an old search-anchor inside derive_main, before the old owner
# could match it. Here generate the fully qualified main FIRST, then rename
# only its resulting metadata. No numerical fixture or resource gate changes.
STAGE2_MAIN_SOURCE=edit(_main,(
    ('unit_v1','unit_v2'),
    ('fixed_D5_Gram_only','fixed_D5_weight_pair_only'),
    ('helper_repeats_exact','two_rhs_parity_exact'),
    ('all_independent_references_passed','original_source_reference_parity'),
    ('repeat_claimed=True','repeat_claimed=False'),
))
STAGE2_MAIN_PROOF=dict(parent=ROOT_MAIN_PROOF,entry_generation_order_repaired=True,
    numerical_fragment_unchanged=True,source_sha256=hashlib.sha256(STAGE2_MAIN_SOURCE.encode()).hexdigest())
_nodes=[n for n in ast.parse(_source).body if isinstance(n,ast.FunctionDef) and n.name=='derive_main']
if len(_nodes)!=1:
    raise RuntimeError('One exact faulty metadata generator')
_before=ast.get_source_segment(_source,_nodes[0])
_source=edit(_source,(
    (_before,'def derive_main():\n    return STAGE2_MAIN_SOURCE,STAGE2_MAIN_PROOF'),
    ("HERE/'worker.py'","HERE/'worker_v2.py'"),
    ("HERE/'client.py'","HERE/'client_v2.py'"),
    ("HERE/'watch.py'","HERE/'watch_v2.py'"),
    ("HERE/'unit_v1/remote/call_00.json'","HERE/'unit_v2/remote/call_00.json'"),
))
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_weight_reuse_unit_v2_inert',
    UNIT_SOURCE_SHA=_old.UNIT_SOURCE_SHA,STAGE2_MAIN_SOURCE=STAGE2_MAIN_SOURCE,
    STAGE2_MAIN_PROOF=STAGE2_MAIN_PROOF)
exec(compile(_source,str(Path(__file__).resolve())+':entry-order-repair','exec'),_api)
_api['PINS'].update({V1:V1_SHA,ROOT:ROOT_SHA,HERE/'unit_arithmetic.py':_old.UNIT_SOURCE_SHA})
for _name in ('main','admit','owner_map','helper_source','primitive_namespace','derive_main','PINS',
              'numerical_modules','sha','read','write','verify_call','call_records','compare',
              'SCHEMA','CALL_BUDGET','ROSTER'):
    globals()[_name]=_api[_name]
del _source,_main,_before,_nodes


if __name__=='__main__':
    raise SystemExit(main())

