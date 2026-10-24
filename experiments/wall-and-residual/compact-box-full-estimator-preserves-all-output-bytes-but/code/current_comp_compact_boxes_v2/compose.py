"""V2 factual depth3-name repair; canonical boxes/arithmetic/gates unchanged."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_compact_boxes_v1/compose.py'
OWNER_SHA='e985c62dca01878b502df4ffdee0b36653d536ffbaea0bfe5eb54146030840a4'


def source_adapter():
    original=OWNER.read_text()
    if OWNER.is_symlink() or hashlib.sha256(original.encode()).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Frozen source-admission-negative V1 composer')
    edits=(
        ('(3,3,_PACKED_SQUARE3),(4,4,_PACKED_SQUARE4)','(3,3,_PACKED_SQUARE),(4,4,_PACKED_SQUARE4)'),
        ('    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n    return module',
         '    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n'
         '    if module.RUNTIME.count("_PACKED_SQUARE3")!=1:\n'
         '        raise RuntimeError("Exact original depth3 runtime typo owner")\n'
         '    module.RUNTIME=module.RUNTIME.replace("_PACKED_SQUARE3","_PACKED_SQUARE")\n'
         '    if module.LAYOUT_NAMES.count("_PACKED_SQUARE3")!=1:\n'
         '        raise RuntimeError("Exact original depth3 metadata typo owner")\n'
         '    module.LAYOUT_NAMES=tuple("_PACKED_SQUARE" if n=="_PACKED_SQUARE3" else n for n in module.LAYOUT_NAMES)\n'
         '    return module'),
        ("schema='current_comp_compact_boxes_composition_v1'","schema='current_comp_compact_boxes_composition_v2'"))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('One factual-name/source-version splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact factual repair inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire V1 source restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_V1_source_inverse=True,
        depth3_actual_name='_PACKED_SQUARE',repaired_name='_PACKED_SQUARE3',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        no_geometry_cache_arithmetic_or_gate_change=True)


_source,V2_SOURCE_PROOF=source_adapter()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_boxes_v2_source_inert')
exec(compile(_source,str(Path(__file__).resolve())+':factual-V1-inverse','exec'),_api)
del _source
metadata_api=_api['metadata_api']


def authenticate(tick=None):
    pins=_api['authenticate'](tick);pins[str(OWNER)]=OWNER_SHA
    return pins


def derive_source(tick=None):
    text,proof=_api['derive_source'](tick)
    proof['source_bindings'][str(OWNER)]=OWNER_SHA
    proof['factual_depth3_name_repair']=V2_SOURCE_PROOF
    return text,proof


if __name__=='__main__':
    print('Inert V2 source repair; no numerical run or candidate file.')

