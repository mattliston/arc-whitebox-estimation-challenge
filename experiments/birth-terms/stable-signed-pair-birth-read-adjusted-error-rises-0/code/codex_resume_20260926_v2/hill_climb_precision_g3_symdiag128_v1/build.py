"""Stdlib-only fixed source edits, inverse proof and rational cost metadata."""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent/'hill_climb_precision_g4_symdiag_batched_v2/candidate.py'
PARENT_SHA = 'e82f7bf1fbe736d37816492e42c456198df07006152436a299b36eee757e2341'
PARENT_BUILD_SHA = 'ed53ec46a59caedea66601f40ef39ccc9a01fc6f3dc913bfb0f53860d90b29d9'
EDITS = (
    ('gaussian_pair_slices',
     'Fr16=Fr[1:5], Gr16=Gr[1:5], P16=PB[1:5], P6=PB[1:5][:, None]',
     'Fr16=Fr[1:4], Gr16=Gr[1:4], P16=PB[1:4], P6=PB[1:4][:, None]'),
    ('gaussian_divisors', 'DIV6=fnp.arange(1, 5, dtype=f32)',
     'DIV6=fnp.arange(1, 4, dtype=f32)'),
    ('gaussian_x1_view', 'X1=fnp.reshape(arena[0:12 * nn], (4, 3, n, n))',
     'X1=fnp.reshape(arena[0:9 * nn], (3, 3, n, n))'),
    ('comment_only', '# fixed Gaussian R4: rho^r/r!, r=1..4',
     '# fixed Gaussian R3: rho^r/r!, r=1..3'),
)


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def save(name, value):
    with (D/name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def product(m, k, n):
    assert all(x % 64 == 0 for x in (m,k,n))
    g = Fraction(7,4)**6
    cost = ((g+Fraction(9,4))*(m*k+k*n)
            +(2*g-Fraction(3,4)+Fraction(35,16))*m*n
            +7**6*(m//64)*(n//64)*(2*(k//64)-1))
    assert cost.denominator == 1
    return int(cost)


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(PARENT.parent/'build.json')['sha256'] == PARENT_BUILD_SHA
    parent_build = json.loads((PARENT.parent/'build.json').read_text())
    for key in ('candidate','builder','protocol','fragment','wrapper','pricing',
                'side_dependency','parent_build','parent'):
        assert pin(parent_build[key]['path']) == parent_build[key]
    for name in ('candidate.py','pricing.json','build.json'):
        assert not (D/name).exists(), name
    original = PARENT.read_text(); source = original
    edits = []
    for label, before, after in EDITS:
        assert source.count(before) == 1, label
        source = source.replace(before,after,1)
        edits.append(dict(label=label,before=before,after=after))
    inverse = source
    for edit in reversed(edits):
        assert inverse.count(edit['after']) == 1, edit['label']
        inverse = inverse.replace(edit['after'],edit['before'],1)
    assert inverse == original
    fragment = (D/'fragment.py').read_text()
    candidate = source + '\n' + fragment
    ast.parse(candidate)
    old_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body
                     if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    new_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                     if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    changed = sorted(k for k in old_functions if old_functions[k] != new_functions[k])
    assert changed == ['_e2_pair_arena','_e2_pair_state'], changed
    assert old_functions['_nonlinear_pair_batch'] == new_functions['_nonlinear_pair_batch']
    with (D/'candidate.py').open('x') as f: f.write(candidate)
    old_keys = ((0,64,1024),(0,128,1024),(1,1024,64),(1,1024,128),
                (1,1024,256),(2,64,64),(2,64,128),(2,128,256))
    keys = old_keys + ((2,128,128),)
    region = (90354432,120472576,120472576)
    starts = (0,region[0],region[0]+region[1])
    sides = []
    for kind,rows,cols in keys:
        size = 4*7**6*(rows//64)*(cols//64)
        bind = 4*(7**6+7*4**5)*(rows//64)*(cols//64)
        assert size <= region[kind] and starts[kind]+size <= 391602176
        sides.append(dict(key=[kind,4,6,rows,cols],start_elements=starts[kind],
                          tree_elements=size,bind_flops=bind,region_elements=region[kind]))
    outer = sum(product(256,1024,n) for n in (768,512,256))
    groups = []
    for offset,m,n in ((0,128,256),(128,128,128)):
        trees = [4*7**6*(r//64)*(c//64) for r,c in ((m,1024),(1024,n),(m,n))]
        groups.append(dict(offset=offset,shape=[4,m,1024,n],warm_flops=4*product(m,1024,n),
                           tree_elements=trees,tree_bytes=4*sum(trees),
                           fresh_result_bytes=4*4*m*n))
    warm = outer + sum(g['warm_flops'] for g in groups) + 3*1024**2
    assert warm == 717329488
    assert sides[-1]['bind_flops'] == 1997072
    assert sum(s['bind_flops'] for s in sides) == 87371900
    geometry_debit = 14*(warm-709895628)
    gaussian_warm = 14*11*1024**2
    gaussian_x1_bind = 3*1024**2
    save('pricing.json',dict(parent=pin(PARENT),parent_pricing=parent_build['pricing'],
        candidate=pin(D/'candidate.py'),fragment=pin(D/'fragment.py'),source_only=True,
        child_warm_per_helper=warm,parent_warm_per_helper=709895628,
        outer_product_flops=outer,assembly_copy_flops=1024**2,split_axis_view_flops=2*1024**2,
        groups=groups,side_keys=sides,batch_side_bind_flops=87371900,new_side_bind_flops=1997072,
        setup=dict(bound_total=682,single_fused=649,batch_fused=9,fused_total=658,ks=24,
                   every_old_key_preserved=True,cache_caps_unchanged=True),
        calls=dict(parent_operations_per_helper=537,child_source_operations_per_helper=444,
                   removed_per_helper=93,removed_14_helpers=1302,
                   note='Source operation count, not an RPC or measured runtime claim.'),
        direct_fixed_shape_forecast=dict(helpers=14,geometry_debit_flops=geometry_debit,
            gaussian_warm_saving_flops=gaussian_warm,gaussian_x1_one_bind_saving_flops=gaussian_x1_bind,
            net_warm_saving_flops=gaussian_warm-geometry_debit,
            net_including_one_x1_and_new_side_bind=gaussian_warm-geometry_debit+gaussian_x1_bind-1997072,
            unpriced=['one arange entry','conditional downstream route and cache changes'],
            note='Fixed source shapes only; setup and prediction binding contexts differ. Native full bills are authoritative.'),
        memory=dict(arena_elements=391602176,arena_bytes=1566408704,
            maximum_batch_tree_bytes=max(g['tree_bytes'] for g in groups),
            maximum_batch_result_bytes=max(g['fresh_result_bytes'] for g in groups),
            largest_outer_result_bytes=786432,final_output_bytes=4194304,
            new_tree_allocation=False,new_input_pack=False,
            note='Component extents only, not process RSS qualification.'),
        numerical_note='Partition may change rounded values; Gaussian order is approximate. Primitive reference gates and native endpoint gates are mandatory.'))
    save('build.json',dict(parent=pin(PARENT),candidate=pin(D/'candidate.py'),builder=pin(__file__),
        protocol=pin(D/'PROTOCOL.md'),fragment=pin(D/'fragment.py'),pricing=pin(D/'pricing.json'),
        parent_build=pin(PARENT.parent/'build.json'),side_dependency=parent_build['side_dependency'],
        edits=edits,edited_functions=changed,source_inverse_exact=True,
        append_only=False,edited_prefix_plus_append=True,suffix_bytes=len(fragment.encode())+1,
        all_unlisted_source_bytes_unchanged=True,model_executed=False))
    print(json.dumps({name:pin(D/name) for name in ('candidate.py','build.json','pricing.json','PROTOCOL.md')}))


if __name__ == '__main__': main()
