"""Stdlib-only immutable source concatenation and integer tariff metadata."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
H2 = D.parent
PARENT = H2/'hill_climb_precision_g4_symdiag_v1/candidate.py'
KERNEL = H2/'hill_climb_hk15_blocked_v2/kernel_blocked.py'
PARENT_SHA = '09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd'
KERNEL_SHA = 'c9c87cb4680d165c3c908e086dcd49cac2ccab5b552a9e31e7036cb6d76a3f3c'
KEYS = ((0,64,1024),(0,128,1024),(1,1024,64),(1,1024,128),
        (1,1024,256),(2,64,64),(2,64,128),(2,128,256))


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(KERNEL)['sha256'] == KERNEL_SHA
    for name in ('fragment.py','candidate.py','pricing.json','build.json'):
        assert not (D/name).exists(), name
    source = KERNEL.read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Side']
    assert len(nodes)==1
    original = ast.get_source_segment(source,nodes[0])
    renamed = original.replace('class Side:', 'class _PSDBSide:', 1)
    assert renamed.replace('class _PSDBSide:', 'class Side:', 1)==original
    wrapper = (D/'wrapper.py').read_text()
    fragment = ('# Generic binary Side only, extracted from pinned blocked kernel.\n'
                + renamed + '\n\n' + wrapper)
    candidate = PARENT.read_bytes()+b'\n'+fragment.encode()
    ast.parse(fragment);ast.parse(candidate)
    assert candidate[:-len(fragment.encode())-1] == PARENT.read_bytes()
    with (D/'fragment.py').open('x') as f:f.write(fragment)
    with (D/'candidate.py').open('xb') as f:f.write(candidate)
    region = (90354432,120472576,120472576)
    starts = (0,region[0],region[0]+region[1])
    sides=[]
    for kind,rows,cols in KEYS:
        elements=4*7**6*(rows//64)*(cols//64)
        bind=4*(7**6+7*4**5)*(rows//64)*(cols//64)
        assert elements <= region[kind]
        assert starts[kind]+elements <= 391602176
        sides.append(dict(key=[kind,4,6,rows,cols],start_elements=starts[kind],
            tree_elements=elements,bind_flops=bind,region_capacity_elements=region[kind]))
    groups=[]
    for offset,m,n in ((0,128,256),(128,64,128),(192,64,64)):
        tree=[4*7**6*x*y//64**2 for x,y in ((m,1024),(1024,n),(m,n))]
        groups.append(dict(offset=offset,shape=[4,m,1024,n],tree_elements=tree,
            tree_bytes=4*sum(tree),fresh_result_bytes=4*4*m*n))
    assert sum(s['bind_flops'] for s in sides)==85374828
    price=dict(parent=pin(PARENT),kernel=pin(KERNEL),wrapper=pin(D/'wrapper.py'),
        source_only=True,model_executed=False,numerical_arrays_read=False,
        parent_warm_per_helper=707798476,child_warm_per_helper=709895628,
        split_axis_view_flops_per_helper=2097152,delta_flops_14_helpers=29360128,
        batch_side_bind_flops=85374828,side_keys=sides,groups=groups,
        setup=dict(parent_bound=673,parent_single_fused=649,new_batch_fused=8,
                   bound_total=681,fused_total=657,old_keys_preserved=True),
        calls=dict(parent_diagonal_cores=12,child_diagonal_cores=3,
            explicit_fnp_calls_per_core=83,positive_cost_calls_per_core=76,
            saved_explicit_core_calls_per_helper=747,saved_explicit_core_calls_14=10458,
            saved_positive_cost_core_calls_per_helper=684,
            saved_positive_cost_core_calls_14=9576,
            note='Exact arithmetic-core source counts. Extra split-axis reshape/moveaxis, result unstack, allocation/assembly views and worker requests are separate; no RPC saving is certified.'),
        memory=dict(arena_elements=391602176,arena_bytes=1566408704,
            maximum_batch_tree_bytes=max(g['tree_bytes'] for g in groups),
            maximum_batch_result_bytes=max(g['fresh_result_bytes'] for g in groups),
            unchanged_largest_outer_result_bytes=786432,final_output_bytes=4194304,
            separate_tree_allocation=False,packing_copy=False,
            note='The entire old arena remains allocated. Cached sides are views at old fixed offsets. This is component accounting, not a process RSS certificate.'),
        cold_predict_note='All8 new sides are intended to bind during setup. A lost setup arena triggers the existing reset/lazy path; no cold prediction saving is promised.',
        numerical_note='Per-product order is preserved by source; exact parent output bytes remain mandatory in the root primitive and native parity check.',
        runtime_note='Existing failed boundary remains immutable. Source-call savings are not a residual/wall qualification.')
    write_json(D/'pricing.json',price)
    result=dict(parent=pin(PARENT),candidate=pin(D/'candidate.py'),builder=pin(__file__),
        protocol=pin(D/'PROTOCOL.md'),fragment=pin(D/'fragment.py'),wrapper=pin(D/'wrapper.py'),
        pricing=pin(D/'pricing.json'),side_dependency=pin(KERNEL),
        parent_build=pin(PARENT.parent/'build.json'),append_only=True,source_inverse_exact=True,
        suffix_bytes=len(fragment.encode())+1,
        generic_side_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generic_side_only_change='class Side -> class _PSDBSide',
        no_hk_circuit_or_dispatch=True,model_executed=False)
    write_json(D/'build.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
