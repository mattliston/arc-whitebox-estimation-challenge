"""Original five-call NumPy-free client, current candidate and fresh outputs."""
import ast
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
PARENT=HERE/'client.py'
PARENT_SHA='dbdd9d28b0bd3c15bb83e1fe8669e593aba925b576b27b8f80bb650143f72fad'


def load_client():
    raw=PARENT.read_bytes()
    if PARENT.is_symlink() or hashlib.sha256(raw).hexdigest()!=PARENT_SHA:
        raise RuntimeError('Frozen fixed-client metadata owner')
    original=raw.decode(); lines=original.splitlines(keepends=True)
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='source_and_proof')
    old_function=''.join(lines[node.lineno-1:node.end_lineno])
    new_function="""def source_and_proof():
    raw = OWNER.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OWNER_SHA256, 'Original exact production-five body')
    return raw.decode(), dict(owner=str(OWNER),owner_sha256=OWNER_SHA256,
        derived_source_sha256=OWNER_SHA256,whole_source_inverse_verified=True,
        source_splices=0,ranks=[35,100,500,3016500,35],full_model_calls=5,
        persistent_estimator=True,original_here_and_fixture_loading_preserved=True,
        original_context_and_output_arithmetic_unchanged=True,
        original_repeat_output_hash_check_unchanged=True,no_runner_or_inner_monitor=True,
        input_preparation_convention='original fnp.load inside participant context; actual complete bill')"""
    edits=[
        (old_function,new_function),
        ("TAG = 'current_comp_shared_gain_slice_pair_v1'","TAG = 'current_comp_shared_gain_slice_production_v1'"),
        ("*(NEW / f'{TAG}_{i}_r35.f32' for i in range(2))",
         "*(NEW / f'{TAG}_{i}_r{r}.f32' for i,r in enumerate((35,100,500,3016500,35)))"),
    ]
    source=original; history=[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('Unique production-client metadata span')
        at=source.index(before); history.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(history):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact production-client inverse span')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse.encode()!=raw:
        raise RuntimeError('Complete client metadata source inverse')
    scope=dict(__name__='gain_production_client_inert',__file__=str(Path(__file__).resolve()))
    exec(compile(source,str(PARENT)+':original-five','exec'),scope)
    scope['PINS'][PARENT]=PARENT_SHA
    return types.SimpleNamespace(**scope)


if __name__=='__main__':
    raise SystemExit(load_client().main())
else:
    _client=load_client()
    PINS=_client.PINS
    source_and_proof=_client.source_and_proof
