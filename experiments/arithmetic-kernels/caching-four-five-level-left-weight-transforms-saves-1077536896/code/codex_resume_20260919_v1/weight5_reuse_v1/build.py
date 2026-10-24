"""Append an invertible, explicitly owned D5 weight-transform cache to Read5."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'read5_packet_v1/candidate.py'
EXPECTED = 'd7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b'
ORIGIN = HERE.parent.parent / 'astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/port.py'
ORIGIN_SHA = '741e6ed41b7cbdf338cb85690f463fa2d813056a1650fa5081ebbb2a5731567b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


source = PARENT.read_text()
assert sha(source.encode()) == EXPECTED
tree = ast.parse(source)
origin = ORIGIN.read_text()
assert sha(origin.encode()) == ORIGIN_SHA
assert '_W5_PORT_SCOPE' not in source


def last(name):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return ast.get_source_segment(source, nodes[-1])


literal = [n for n in ast.parse(origin).body if isinstance(n, ast.Assign)
           and any(isinstance(v, ast.Name) and v.id == 'RUNTIME' for v in n.targets)]
assert len(literal) == 1
runtime = ast.literal_eval(literal[0].value)
# The current arena owns exact contiguous left frames. Match their strides.
runtime = once(runtime, 'cm,ck = 1<<(m-1).bit_length(),1<<(k-1).bit_length()', 'cm,ck = m,k')
assert 'na = _exact_view(arena, a_specs[index], (m2, k2))' in last('_a_frame')
assert 'key = (id(views), L, m, k)' in last('_a_frame')
assert 'return fnp.reshape(arena[start:start + numel], exact)' in last('_exact_view')
assert 'aparts, leaf_a = _a_frame(views, levels, m, k)' in last('_packed_frame')

old_product = last('_packed_product_cached')
left_start = old_product.index('        fnp.add(a21,a22,out=aa[4])\n')
left_stop = old_product.index('        fnp.subtract(b12,b11,out=bb[4])\n')
left = old_product[left_start:left_stop]
assert left.count('fnp.') == 6
guarded = '        if cache._leaf is None:\n' + ''.join('    ' + line for line in left.splitlines(keepends=True))
old_header = 'def _packed_product_cached(a,b,out,views,levels):'
new_header = 'def _w5_port_product_body(a,b,out,views,levels,cache):'
leaf_line = '    fnp.matmul(leaf_a,leaf_b,out=leaf)'
copy = '''    if cache._leaf is None:
        cached = fnp.empty(cache._shape,dtype=fnp.float32)
        cache._own_leaf(cached,cached)
        fnp.copyto(cached,leaf_a)
    else:
        leaf_a = cache._leaf
'''
product = once(once(once(old_product, old_header, new_header), left, guarded), leaf_line, copy + leaf_line)
assert product.replace(copy + leaf_line, leaf_line, 1).replace(guarded, left, 1).replace(new_header, old_header, 1) == old_product

sites = {
    '_cov_compact': ('result = _mm(wa.T, _mm(ca, wa))',
        "result = _w5_port_call(_W5_PORT_SCOPE,w,indices,'cov_outer',_mm,wa.T,_mm(ca,wa))"),
    '_star_compact': ('pushed = _mm(wp.T, aa)',
        "pushed = _w5_port_call(_W5_PORT_SCOPE,w,previous,'star_push',_mm,wp.T,aa)"),
    '_push_bank_shared_newborn': ('common = _mm(wp,base[previous,:])',
        "common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])"),
    '_shared_old384_push': ('U = _shared_old384_mm(w32[previous, :].T, active_basis)',
        "U = _w5_port_call(_W5_PORT_SCOPE,w32,previous,'shared_basis',_shared_old384_mm,w32[previous, :].T,active_basis)"),
}
definitions = []
for name, (before, after) in sites.items():
    original = last(name)
    changed = once(original, before, after)
    assert changed.replace(after, before, 1) == original
    definitions.append(changed)

old_d5 = last('_packed_product_5')
assert old_d5 == 'def _packed_product_5(a,b,out,arena_views):\n    return _packed_product_cached(a,b,out,arena_views,5)'
definitions.append('''def _packed_product_5(a,b,out,arena_views):
    scope = _W5_PORT_SCOPE
    if scope is None or scope['grant'] is None:
        return _packed_product_cached(a,b,out,arena_views,5)
    return _w5_port_product(a,b,out,arena_views,scope)''')

original = last('_predict_binary')
start = '        w32 = fnp.asarray(W[l], dtype=fnp.float32)\n'
begin = '        _w5_port_begin(_w5_context,mlp,l,w32,previous)\n'
star = '        star_pre = fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else _star_compact(w32, star, previous, previous_alive, live_indices)\n'
end = '        _w5_port_end_layer(_w5_context)\n'
body = once(once(original, start, start + begin), star, star + end)
assert body.replace(start + begin, start, 1).replace(star + end, star, 1) == original
header, inside = body.split('\n', 1)
indented = ''.join('    ' + line if line.strip() else line for line in inside.splitlines(keepends=True))
assert ''.join(line[4:] if line.strip() else line for line in indented.splitlines(keepends=True)) == inside
definitions.append(header + '''
    global _W5_PORT_SCOPE
    if _W5_PORT_SCOPE is not None:
        raise RuntimeError('W5 no concurrent or nested network prediction')
    _w5_context = _w5_port_new(mlp)
    _W5_PORT_SCOPE = _w5_context
    try:
''' + indented + '''
    except BaseException:
        _w5_context['failed'] = True
        raise
    finally:
        try:
            _w5_port_finish(_w5_context)
        finally:
            _W5_PORT_SCOPE = None''')

extension = '\n\n# Exact contiguous four-site W-left D5 reuse, 2026-09-19.\n' + runtime
extension += '\n\n' + '\n\n'.join([product, *definitions]) + '\n'
ast.parse(extension)
candidate = source + extension
assert candidate[:-len(extension)] == source
with (HERE / 'candidate.py').open('x') as stream:
    stream.write(candidate)
proof = dict(parent_sha256=EXPECTED, origin_sha256=ORIGIN_SHA,
             candidate_sha256=sha(candidate.encode()), builder_sha256=sha(Path(__file__).read_bytes()),
             source_bytes=len(candidate.encode()), parent_byte_prefix_exact=True,
             product_inverse_exact=True, physical_sites_inverse_exact=True,
             prediction_body_inverse_exact=True, current_contiguous_left_layout_proved=True,
             explicit_sites=list(sites), cache_max_bytes=68841472,
             parent_arena_unchanged=True, original_dispatch_and_numeric_order_unchanged=True,
             original_primitives_omitted_per_hit=30, copy_primitives_per_creator=1,
             truth_accessed=False, numerical_calls=0)
with (HERE / 'build.json').open('x') as stream:
    json.dump(proof, stream, indent=2)
print(json.dumps(proof))
