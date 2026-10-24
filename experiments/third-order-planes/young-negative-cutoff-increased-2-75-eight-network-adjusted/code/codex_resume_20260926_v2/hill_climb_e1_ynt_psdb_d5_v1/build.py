"""Stdlib-only append/inverse and fixed depth/arena proof; no model imports."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent/'hill_climb_e1_gnbatch_young_trace_v1/candidate.py'
PARENT_SHA = 'ab2a51309b24dfc65ad46439e4d7c54d6d90c49e3fbc895b2822cc56cad2431a'
ACTIVE = {(0,128,1024),(1,1024,128),(1,1024,256),(2,128,128),(2,128,256)}
KEYS = ((0,64,1024),(0,128,1024),(1,1024,64),(1,1024,128),
        (1,1024,256),(2,64,64),(2,64,128),(2,128,256),(2,128,128))


def pin(path):
    p=Path(path).resolve(); raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    assert pin(PARENT)['sha256']==PARENT_SHA
    parent=PARENT.read_text(); fragment=(D/'fragment.py').read_text()
    child=parent+'\n'+fragment
    assert child[:-(len(fragment)+1)]==parent
    pt,ct=ast.parse(parent),ast.parse(child)
    assert [ast.dump(x) for x in ct.body[:len(pt.body)]]==[ast.dump(x) for x in pt.body]
    functions={n.name:n for n in pt.body if isinstance(n,ast.FunctionDef)}
    lines=parent.splitlines()
    def source(n):return '\n'.join(lines[n.lineno-1:n.end_lineno])
    assert '_psdb_side(kind, rows, cols)' in source(functions['_pb_bind'])
    assert '_PSDB_OLD_BIND(t0)' in source(functions['_pb_bind'])
    group=source(functions['_psdb_group'])
    for term in ('sa.forward(a, keep)','sb.forward(b, keep)',
                 'fnp.matmul(sa.T, sb.T, out=sc.T)','sc.unwind(parts, keep)'):
        assert term in group
    assert '_psdb_group(A4, B4, out, offset, rows, cols)' in source(functions['_psdb_sym_upper'])
    assert '_PSDB_GROUPS = ((0, 128, 256), (128, 128, 128))' in parent
    assert parent.endswith((D.parent/'hill_climb_young_negative_trace_v1/fragment.py').read_text())
    ft=ast.parse(fragment)
    assert {n.name for n in ft.body if isinstance(n,ast.FunctionDef)}=={'_psdb_fits','_psdb_side','_psdb_group'}
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.AugAssign)) for n in ast.walk(ft))
    region=(90354432,120472576,120472576);starts=(0,region[0],region[0]+region[1])
    layout=[]
    for kind,rows,cols in KEYS:
        depth=5 if (kind,rows,cols) in ACTIVE else 6;u=1<<depth
        assert rows%u==cols%u==0
        size=4*7**depth*(rows//u)*(cols//u)
        bind=4*(7**depth+7*4**(depth-1))*(rows//u)*(cols//u)
        assert size<=region[kind] and starts[kind]+size<=391602176
        layout.append(dict(key=[kind,rows,cols],depth=depth,start_elements=starts[kind],
                           tree_elements=size,bind_flops=bind))
    assert sum(x['depth']==5 for x in layout)==5 and len(layout)==9
    assert sum(x['bind_flops'] for x in layout)==59136140
    pricing=pin(D.parent/'teammate_s33_psdb_depth5_pricing_v1.json')
    assert pricing['sha256']=='e472778820dd347404c9d4f3ff878a316e028c3757f80ac6e3081914202ddde0'
    with (D/'candidate.py').open('x') as f:f.write(child)
    history=['hill_climb_e1_gnbatch_young_trace_v1/native2_assessment.json',
        'hill_climb_e1_gnbatch_young_trace_v1/boundary_assessment.json',
        'hill_climb_e1_gnbatch_young_trace_v1/boundary_assess_guard.json',
        'hill_climb_e1_gnbatch_young_trace_v1/boundary_assessment_extension_v2.json',
        'hill_climb_e1_gnbatch_young_trace_v1/boundary_assess_guard_extension_v2.json',
        'hill_climb_e1_gnbatch_young_trace_v1/boundary_guard.json',
        'hill_climb_e1_ynt_screen_extension_v2/extension_protocol.json',
        'hill_climb_precision_g3_plane_e1_gnbatch_v2/native8_assessment.json',
        'hill_climb_young_negative_trace_v1/native2_assessment.json']
    record=dict(parent=pin(PARENT),parent_build=pin(PARENT.parent/'build.json'),
        candidate=pin(D/'candidate.py'),builder=pin(__file__),fragment=pin(D/'fragment.py'),
        primitive=pin(D/'probe.py'),protocol=pin(D/'PROTOCOL.md'),pricing=pricing,
        native_driver=pin(D/'native_probe.py'),policy=pin(D.parent/'hill_climb_policy_20260928_v2.json'),
        history=[pin(D.parent/x) for x in history],append_only=True,source_inverse_exact=True,
        suffix_bytes=len(('\n'+fragment).encode()),parent_all_bytes_unchanged=True,
        setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),side_layout=layout,
        batch_side_bind_flops=59136140,arena_elements=391602176,
        source_proof=dict(dynamic_bind=True,unchanged_group_core=True,unchanged_ynt_and_e1=True,
                          private_mixed_depth_group_rejected=True),
        source_only=True,numerical_execution=False,execution_authorized=False)
    with (D/'build.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({key:record[key] for key in ('candidate','builder','fragment','primitive','protocol')}))


if __name__=='__main__':main()
