"""Inert full BOX-source inverse; geometry-compact regular frames only."""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
PARENT=NEW/'current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py'
PARENT_SHA='378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
C2=NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
C2_SHA='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
UNIT=NEW/'current_comp_compact_scratch_v1'
PINS={
    str(PARENT):PARENT_SHA,str(C2):C2_SHA,
    str(UNIT/'kernel.py'):'0de00f81c8fe772e8ed89b0fbadc9b2786be4db58a1980775890c079237b43fc',
    str(UNIT/'DESIGN.md'):'fba209fe46e4ab0421a396f2befbe9cf0b54c1514ab42d488aad85de42977e8b',
    str(UNIT/'unit_watch_v1/result.json'):'28eca2b2e269dc6db6bd2726c2a4b9cd738485d01f03644bce07142a7f67d9d0',
    str(UNIT/'unit_watch_v1/completion.json'):'aa12ad5c62a55488eea74963340b02336ce76736c2e2c1494fa6f3a0ce7f4fa9',
    str(UNIT/'unit_watch_v1/finalization.json'):'7b0cd4d22f2c0761176160152e1a2e21f78b5301e09344136bfe84095089da50'}
LAYOUT_NAMES=('_PACKED_PUSH','_PACKED_READ','_PACKED_SQUARE3','_PACKED_SQUARE4','_PACKED_SQUARE5')
# All scalar bookkeeping uses public shape metadata. No numerical values are
# read or manipulated outside FNP. The counter lets the client reconcile each
# call's actual complete bill to every executed reshape, including rebindings.
RUNTIME="""

# BEGIN EXACT-GEOMETRY REGULAR SCRATCH V1
_CS_BINDING_FLOPS = 0
_CS_BINDING_CALLS = 0


def _cs_shape_axes(levels,m,k,n):
    axes = []
    for level in range(1,levels+1):
        axes.extend(((m//2**level,k//2**level),(k//2**level,n//2**level)))
    axes.extend((m//2**level,k//2**level) for level in range(1,levels+1))
    axes.extend((k//2**level,n//2**level) for level in range(1,levels+1))
    axes.append((m//2**levels,n//2**levels))
    axes.extend((m//2**(level-1),n//2**(level-1)) for level in range(levels,1,-1))
    axes.extend((m//2**level,n//2**level) for level in range(levels,0,-1))
    return tuple(axes)


def _cs_compact_specs(specs,levels,m,k,n):
    assert len(specs)==6*levels and min(m,k,n)>0
    assert m%2**levels==k%2**levels==n%2**levels==0
    axes=_cs_shape_axes(levels,m,k,n)
    assert len(axes)==len(specs)
    compact=[]
    total=0
    changed=False
    for (start,stop,old_shape),(rows,columns) in zip(specs,axes):
        assert 0<rows<=old_shape[-2] and 0<columns<=old_shape[-1]
        shape=(*old_shape[:-2],rows,columns)
        count=1
        for extent in shape:
            count*=extent
        end=start+count
        assert 0<=start<end<=stop<=292225024
        changed=changed or shape!=old_shape
        compact.append((start,end,shape))
        total+=count
    return tuple(compact),total,changed


def _cs_compact_views(levels,m,k,n,views):
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
    # FBX/P1 frontier and child aliases MUST retain their original layout.
    # Their separately owned flat aliases are not remapped by this proposal.
    if selected is None:
        return views
    compact,cost,changed=_cs_compact_specs(selected,levels,m,k,n)
    if not changed:
        return views
    bound=_cp3_bind_arena_views(_SCRATCH[0],compact)
    _CS_BINDING_FLOPS+=cost
    _CS_BINDING_CALLS+=1
    return bound
# END EXACT-GEOMETRY REGULAR SCRATCH V1
"""


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def read_bytes(path,digest,tick=None):
    path=Path(path)
    if tick:tick()
    require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<3*1024**2,'Regular bounded owner')
    before=path.stat()
    raw=path.read_bytes()
    after=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(after) and hashlib.sha256(raw).hexdigest()==digest,'Exact owner bytes')
    if tick:tick()
    return raw


def authenticate(tick=None):
    for path,digest in PINS.items():
        read_bytes(path,digest,tick)
    records=[json.loads(read_bytes(UNIT/'unit_watch_v1'/name,PINS[str(UNIT/'unit_watch_v1'/name)],tick))
             for name in ('result.json','completion.json','finalization.json')]
    r,c,f=records
    require(all(x['passed'] and x['completed'] for x in records)
        and c['result_sha256']==f['result_sha256']==PINS[str(UNIT/'unit_watch_v1/result.json')]
        and f['completion_sha256']==PINS[str(UNIT/'unit_watch_v1/completion.json')]
        and r['component_gate_passed'] and r['comparison']['byte_exact']
        and r['whole_owned_tree_retired'] and r['maximum_swap_growth_bytes']==0,
        'Actual retired fixed compact unit, not a full-card claim')
    return dict(PINS)


def metadata_api(tick=None):
    source=read_bytes(PARENT,PARENT_SHA,tick).decode()
    tree=ast.parse(source)
    layouts={}
    for name in LAYOUT_NAMES:
        assignments=[node for node in tree.body if isinstance(node,ast.Assign)
            and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
        require(bool(assignments),'Actual last-executed layout: '+name)
        layouts[name]=ast.literal_eval(assignments[-1].value)
    # Only source/shape definitions execute. These helpers cannot construct an
    # array unless the numerical _cs_compact_views function is explicitly called.
    namespace={}
    exec(compile(RUNTIME,'compact-frames:inert-shape-metadata','exec'),namespace)
    return layouts,namespace


def derive_source(tick=None):
    pins=authenticate(tick)
    original=read_bytes(PARENT,PARENT_SHA,tick)
    text=original.decode()
    require('_cs_compact_' not in text and '_CS_BINDING_' not in text,'New private metadata namespace')
    tree=ast.parse(text)
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_packed_frame']
    require(len(functions)==1,'One effective cached frame owner')
    frame=ast.get_source_segment(text,functions[0])
    before='    if found is not None:\n        return found\n    cursor = 0'
    after='    if found is not None:\n        return found\n    views = _cs_compact_views(levels,m,k,n,views)\n    cursor = 0'
    require(frame.count(before)==1 and text.count(frame)==1,'One exact cache-miss-only view binding splice')
    new_frame=frame.replace(before,after)
    at=text.index(frame)
    changed=text[:at]+new_frame+text[at+len(frame):]+RUNTIME
    inverse=changed[:-len(RUNTIME)]
    require(inverse[at:at+len(new_frame)]==new_frame,'Exact added frame block')
    inverse=inverse[:at]+frame+inverse[at+len(new_frame):]
    require(inverse.encode()==original,'Entire BOX estimator restored byte-exactly')
    require(new_frame.replace(after,before)==frame and RUNTIME.isascii(),'Only view binding changed')
    proof=dict(schema='current_comp_compact_frames_composition_v1',
        source_bindings=pins,parent_sha256=PARENT_SHA,c2_ancestor_sha256=C2_SHA,
        candidate_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_bytes=len(changed.encode()),
        full_source_inverse_verified=True,original_source_edits=1,
        appendix_sha256=hashlib.sha256(RUNTIME.encode()).hexdigest(),
        original_frame_sha256=hashlib.sha256(frame.encode()).hexdigest(),
        original_cache_key_and_128_eviction_retained=True,original_bindings_paid=True,
        original_D6_dispatch_and_flat_aliases_unchanged=True,regular_D3_D4_D5_only=True,
        exact_interval_prefixes=True,additional_backing_arenas=0,
        all_other_arithmetic_schedules_state_heads_literals_byte_exact=True,
        per_call_added_cost_is_executed_binding_shape_sum=True,
        model_calls=0,parameter_updates=0,truth_accessed=False,
        actual_complete_bills_measured=False,full_runtime_measured=False,
        candidate_materialized=False,submission_authorized=False)
    return changed,proof


if __name__=='__main__':
    print('Inert exact-geometry source; no estimator or numerical stage runs.')

