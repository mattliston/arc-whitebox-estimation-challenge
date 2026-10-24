"""Inert fixed W5 two-RHS unit using the qualified closed-RPC lifecycle."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_root_batch_gram_v1/worker.py'
OWNER_SHA='ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38'
UNIT_SOURCE_SHA='bdbb412a316b5d4c7f6773f9936097f6195b50ed6d91ab560150a6dee517c1be'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact qualified closed-RPC lifecycle owner')
    unit=(HERE/'unit_arithmetic.py').read_text()
    if hashlib.sha256(unit.encode()).hexdigest()!=UNIT_SOURCE_SHA:
        raise RuntimeError('Exact reviewed new numerical fragment')
    original=raw.decode()
    source,positions=original,[]
    def change(before,after):
        nonlocal source
        count=source.count(before)
        if count<1:
            raise RuntimeError('Required W5 source splice: '+before[:100])
        for unused in range(count):
            at=source.index(before)
            positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    tree=ast.parse(unit)
    functions={n.name:ast.get_source_segment(unit,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    if set(functions)!={'owner_map','helper_source','primitive_namespace','_exercise','verify_call','compare'}:
        raise RuntimeError('Exact six reviewed function replacements')
    for name,replacement in functions.items():
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:
            raise RuntimeError('Exactly one lifecycle function: '+name)
        change(ast.get_source_segment(source,nodes[0]),replacement)
    change("SCHEMA='current_comp_root_batch_gram_unit_v1'","SCHEMA='current_comp_weight_reuse_unit_v1'")
    change("ROSTER=('serial_first','batch_first','serial_warm','batch_warm')",
           "ROSTER=('original_R0','original_R1','creator_R0','hit_R1')")
    change('fixed_D5_Gram_only','fixed_D5_weight_pair_only')
    change('helper_repeats_exact','two_rhs_parity_exact')
    change('all_independent_references_passed','original_source_reference_parity')
    change('repeat_claimed=True','repeat_claimed=False')
    change('root-batch-gram-unit','weight-reuse-pair-unit')
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact W5 positional source inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Complete qualified closed-RPC lifecycle restored')
    return source,dict(owner_sha256=OWNER_SHA,new_arithmetic_sha256=UNIT_SOURCE_SHA,
        complete_original_source_inverse=True,original_single_context_failure_retirement_retained=True,
        new_fixed_two_RHS_fixture=True,old_Gram_fixture_or_result_claimed=False,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,UNIT_SOURCE_PROOF=derive_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_weight_reuse_unit_inert',
          UNIT_SOURCE_SHA=UNIT_SOURCE_SHA)
exec(compile(_source,str(Path(__file__).resolve())+':qualified-lifecycle','exec'),_api)
_api['PINS'].update({OWNER:OWNER_SHA,HERE/'unit_arithmetic.py':UNIT_SOURCE_SHA})
for _name in ('main','admit','owner_map','helper_source','primitive_namespace','derive_main','PINS',
              'numerical_modules','sha','read','write','verify_call','call_records','compare',
              'SCHEMA','CALL_BUDGET','ROSTER'):
    globals()[_name]=_api[_name]
del _source


if __name__=='__main__':
    raise SystemExit(main())
