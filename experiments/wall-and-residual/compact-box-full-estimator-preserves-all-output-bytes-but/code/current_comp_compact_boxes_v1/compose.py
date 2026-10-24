"""Inert shared power-of-two scratch boxes; no numerical work at import."""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_frames_v1/compose.py'
OWNER_SHA='0270b21e17117073a99436ee1f2b409f7be90bd8e678d7d1fb725e78f6d93766'

NEW_VIEWS="""def _cs_compact_views(levels,m,k,n,views):
    global _CS_BINDING_FLOPS,_CS_BINDING_CALLS
    if _SCRATCH is None:
        return views
    selected=None
    for index,depth,specs in ((1,4,_PACKED_PUSH),(2,4,_PACKED_READ),
            (3,3,_PACKED_SQUARE3),(4,4,_PACKED_SQUARE4),(5,5,_PACKED_SQUARE5)):
        if views is _SCRATCH[index]:
            assert levels==depth
            selected=specs
            break
    if selected is None:
        return views
    assert min(m,k,n)>0
    cm,ck,cn=(1<<(value-1).bit_length() for value in (m,k,n))
    key=(id(views),levels,cm,ck,cn)
    found=_CS_CAPACITY_VIEWS.get(key)
    if found is not None:
        return found
    compact,cost,changed=_cs_compact_specs(selected,levels,cm,ck,cn)
    if not changed:
        return views
    bound=_cp3_bind_arena_views(_SCRATCH[0],compact)
    _CS_BINDING_FLOPS+=cost
    _CS_BINDING_CALLS+=1
    if len(_CS_CAPACITY_VIEWS)>=128:
        _CS_CAPACITY_VIEWS.pop(next(iter(_CS_CAPACITY_VIEWS)))
    _CS_CAPACITY_VIEWS[key]=bound
    return bound
"""


def _owner():
    if OWNER.is_symlink() or hashlib.sha256(OWNER.read_bytes()).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Unchanged source-only exact-every-geometry draft')
    spec=importlib.util.spec_from_file_location('_compact_boxes_source_owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def derive_runtime(base):
    original=base.RUNTIME
    nodes=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='_cs_compact_views']
    if len(nodes)!=1:
        raise RuntimeError('One original view-only helper')
    old=ast.get_source_segment(original,nodes[0])
    edits=(('_CS_BINDING_CALLS = 0','_CS_BINDING_CALLS = 0\n_CS_CAPACITY_VIEWS = {}'),
           (old,NEW_VIEWS.rstrip()))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('Unique shared-capacity source splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact capacity policy inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Whole original exact-prefix binder restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_runtime_inverse=True,
        original_runtime_sha256=hashlib.sha256(original.encode()).hexdigest(),
        shared_runtime_sha256=hashlib.sha256(source.encode()).hexdigest(),
        canonical_ceil_power_of_two_only=True,capacity_cache_limit=128,
        original_frame_cache_limit=128,all_6d_views_paid=True,
        original_family_identity_retained=True,FBX_P1_aliases_unchanged=True)


def metadata_api(tick=None):
    base=_owner()
    base.RUNTIME,proof=derive_runtime(base)
    return base.metadata_api(tick)


def authenticate(tick=None):
    base=_owner()
    pins=base.authenticate(tick)
    pins[str(OWNER)]=OWNER_SHA
    return pins


def derive_source(tick=None):
    base=_owner()
    base.RUNTIME,box_proof=derive_runtime(base)
    text,proof=base.derive_source(tick)
    proof.update(schema='current_comp_compact_boxes_composition_v1',
        capacity_box_proof=box_proof,exact_interval_prefixes=True,
        exact_actual_geometry_binding=False,canonical_power_of_two_capacity=True,
        separate_128_capacity_cache=True,all_6d_views_paid=True,
        cold37_extra_binding_source_price=3488967424,
        warm37_extra_binding_source_price=0,
        planned_cold37_flops=411230669135,planned_warm37_flops=404454713141,
        price_conditional_on_saved_geometries=True)
    proof['source_bindings'][str(OWNER)]=OWNER_SHA
    return text,proof


if __name__=='__main__':
    print('Inert canonical capacity-box source, no estimator or test runs.')

