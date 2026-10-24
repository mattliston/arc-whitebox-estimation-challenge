"""V2 metadata-only repair: admit actual LAST direct schedule assignment.

No kernel arithmetic, layout, helper roster, tariff or numerical gate changes.
All six frozen V1 files remain byte-authenticated and untouched.
"""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'current_comp_root_batch_gram_v1'
V1_SOURCES={
    V1/'kernel.py':'8a32612c3df91e5c24d1baa4f9c2f33352049f07ac38a8c2be6699f1b4a1e73c',
    V1/'worker.py':'ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38',
    V1/'client.py':'b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d',
    V1/'watch.py':'49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9',
    V1/'PROTOCOL.md':'704db621eaa27697367318371c043fc641b9bd12b6c87d425c13b1fb19ce3d0f',
    V1/'WATCH_SPEC.md':'de62c9d98b2dfa03924c4281b91668acbdce7637cadff3c97e34a432026c1bac'}


def read_owners():
    for path,digest in V1_SOURCES.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen V1 root-batch source: '+str(path))


def derive_source():
    read_owners()
    original=(V1/'kernel.py').read_text()
    before="""    def literal(name):
        nodes = [n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        require(len(nodes)==1, 'One fixed layout/schedule literal: '+name)
        return ast.literal_eval(nodes[0].value)
"""
    after="""    assignment_ownership={}
    def literal(name):
        expected={'_SQUARE_EQUAL_DEPTHS':(1078,1641),
                  '_SQUARE_CROSS_DEPTHS':(1079,1643),'_PACKED_SQUARE5':(992,)}
        require(name in expected,'Only the three fixed literal owners')
        nodes = [n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        require(tuple(n.lineno for n in nodes)==expected[name]
            and all(n.col_offset==0 and n.end_lineno==n.lineno and len(n.targets)==1
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name
                    and isinstance(n.value,ast.Tuple) for n in nodes),
            'Exact observed ordered direct assignment structure: '+name)
        assignment_ownership[name]=dict(count=len(nodes),lines=list(expected[name]),
            selected_line=nodes[-1].lineno,selection='last direct top-level assignment',
            assignment_sha256=[hashlib.sha256(span(n).encode()).hexdigest() for n in nodes])
        return ast.literal_eval(nodes[-1].value)
    require(len(literal('_SQUARE_CROSS_DEPTHS'))==32,
            'Also verify actual two-assignment cross schedule, unused by this geometry')
"""
    edits=(
        (before,after),
        ('        whole_frame_inverse=True,whole_product_inverse=True,first_forward_ops_per_root=11,',
         '        literal_assignment_ownership=assignment_ownership,\n'
         '        metadata_repair_only=True,whole_V1_kernel_inverse=True,\n'
         '        whole_frame_inverse=True,whole_product_inverse=True,first_forward_ops_per_root=11,'))
    source,positions=original,[]
    for before,after in edits:
        if source.count(before)!=1:
            raise RuntimeError('One exact V2 literal metadata splice')
        at=source.index(before);positions.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact V2 positional source inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire frozen V1 kernel restored')
    return source,dict(owner_sha256=V1_SOURCES[V1/'kernel.py'],whole_V1_source_inverse=True,
        metadata_selection_only=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,V2_SOURCE_PROOF=derive_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_root_batch_v2_kernel_inert')
exec(compile(_source,str(Path(__file__).resolve())+':metadata-only-V1-inverse','exec'),_api)
_api['PINS'].update(V1_SOURCES)
for _name in ('PINS','source_bundle','cost_certificate','namespace','setup','batched_gram','clear',
              'ARENA_ELEMENTS','WARM_FLOPS','SERIAL_BIND_FLOPS','BATCH_BIND_FLOPS'):
    globals()[_name]=_api[_name]
del _source
