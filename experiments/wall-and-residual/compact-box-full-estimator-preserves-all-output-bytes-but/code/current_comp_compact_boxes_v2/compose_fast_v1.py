"""Linear-time source-span adapter; exact emitted candidate bytes are fixed."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
OWNER=HERE/'compose.py'
OWNER_SHA='46424dca2aa802b6313bce3d3a7d99987cf53c7c4118cf7ddd055b8ac2677d0f'
CANDIDATE_SHA='b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'


def _linear_frame_segment(source,node,**kwargs):
    if not (isinstance(node,ast.FunctionDef) and node.name=='_packed_frame' and not kwargs):
        raise RuntimeError('Only exact ASCII packed-frame source-span lookup is adapted')
    lines=source.splitlines(keepends=True)
    first,last=node.lineno-1,node.end_lineno-1
    if node.col_offset!=0 or not 0<=first<=last<len(lines):
        raise RuntimeError('Actual direct top-level frame AST span')
    selected=lines[first:last+1]
    if not all(line.isascii() for line in selected):
        raise RuntimeError('ASCII source columns equal AST UTF8 byte columns')
    if first==last:
        value=selected[0][node.col_offset:node.end_col_offset]
    else:
        value=selected[0][node.col_offset:]+''.join(selected[1:-1])+selected[-1][:node.end_col_offset]
    if not value.startswith('def _packed_frame(') or not value.endswith('return frame'):
        raise RuntimeError('Exact complete frame source boundaries')
    return value


def _owner():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact admitted V2 source owner')
    spec=importlib.util.spec_from_file_location('_compact_boxes_fast_source_owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_owner=module._api['_owner']
    def frame_source_owner():
        base=original_owner()
        # A private metadata namespace, not a stdlib/global or numerical API patch.
        # Every other AST operation remains its original function.
        facade=types.SimpleNamespace(**vars(ast))
        facade.get_source_segment=_linear_frame_segment
        base.ast=facade
        return base
    module._api['_owner']=frame_source_owner
    return module


def authenticate(tick=None):
    pins=_owner().authenticate(tick);pins[str(OWNER)]=OWNER_SHA
    return pins


def metadata_api(tick=None):
    return _owner().metadata_api(tick)


def derive_source(tick=None):
    text,proof=_owner().derive_source(tick)
    if hashlib.sha256(text.encode()).hexdigest()!=CANDIDATE_SHA:
        raise RuntimeError('Whole emitted candidate identical to actual V2 inert admission')
    proof['source_bindings'][str(OWNER)]=OWNER_SHA
    proof['linear_frame_source_span']=dict(exact_emitted_candidate_sha256=CANDIDATE_SHA,
        original_inverse_still_executed=True,metadata_lookup_only=True,
        candidate_or_arithmetic_change=False)
    return text,proof


if __name__=='__main__':
    print('Inert exact-source lookup accelerator; no numerical work.')

