"""DRAFT: inert, fixed D5 copied-left unit; not a model or full-card adapter.

Only namespace/setup/product entry points can execute supplied FNP operations.
Import and source_bundle use stdlib source/literal processing only. This file
has not been executed or qualified. DESIGN.md remains the frozen authority.
"""
import ast
import hashlib
import math
from pathlib import Path
import weakref

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'current_comp_compact_boxes_v2/rpc37_v1/estimator.py'
PARENT_SHA = 'b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'
DESIGN = HERE / 'DESIGN.md'
DESIGN_SHA = '63aaf29623a3f729bac9c6827bac7bea5a267141ab3048f2356e81610adfc06e'
PINS = {PARENT: PARENT_SHA, DESIGN: DESIGN_SHA}
DEPTH = 5
LAYER = 5
LEFT_SHAPE = (1024, 992)
RIGHT_SHAPES = ((992, 1024), (992, 384))
ROLES = ('covariance_outer', 'shared_basis_push')
ARENA_ELEMENTS = 292225024
CACHE_SHAPE = (7, 7, 7, 7, 7, 32, 32)
LEAF_SHAPE = (7, 7, 7, 7, 7, 32, 31)
BASE_BIND_FLOPS = 172392448
COMPACT_BIND_FLOPS = 115826176
SETUP_FLOPS = 288218624
PRODUCT_FLOPS = (1160608064, 458060764)
LEFT_TRANSFORM_FLOPS = 36532384
CACHE_COPY_FLOPS = 16672544
REUSE_PRODUCT_FLOPS = (1177280608, 421528380)
_FUNCTION_NAMES = ('_cp3_bind_arena_views', '_cs_shape_axes',
                   '_cs_compact_specs', '_cs_compact_views',
                   '_packed_frame', '_packed_product_cached')
_LITERAL_COUNTS = {'_PACKED_ELEMENTS': 3, '_PACKED_PUSH': 3,
                   '_PACKED_READ': 3, '_PACKED_SQUARE': 1,
                   '_PACKED_SQUARE4': 1, '_PACKED_SQUARE5': 1}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, digest, tick=None):
    if tick:
        tick()
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 4 * 1024 * 1024,
            'Bounded regular source owner: ' + str(path))
    raw = path.read_bytes()
    require(_sha(raw) == digest, 'Exact source owner: ' + str(path))
    if tick:
        tick()
    return raw.decode('utf-8')


def _function(source, name):
    """Parse only the selected small definition, never the giant learned literal."""
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith('def ' + name + '(')]
    require(len(starts) == 1, 'One exact function definition: ' + name)
    start = starts[0]
    stop = start + 1
    while stop < len(lines) and (not lines[stop].strip() or lines[stop][0].isspace()):
        stop += 1
    text = ''.join(lines[start:stop]).rstrip('\n')
    tree = ast.parse(text)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == name and not tree.body[0].decorator_list,
            'Only the complete undecorated source function: ' + name)
    return text


def _literal(source, name, count):
    lines = [line for line in source.splitlines() if line.startswith(name + ' = ')]
    require(len(lines) == count, 'Exact original assignment roster: ' + name)
    line = lines[-1]
    return line, ast.literal_eval(line.split(' = ', 1)[1])


def _compact_price(specs, m, k, n):
    # Independent integer witness of the unchanged source helper's 6*d axes.
    axes = []
    for level in range(1, DEPTH + 1):
        axes.extend(((m // 2**level, k // 2**level),
                     (k // 2**level, n // 2**level)))
    axes.extend((m // 2**level, k // 2**level) for level in range(1, DEPTH + 1))
    axes.extend((k // 2**level, n // 2**level) for level in range(1, DEPTH + 1))
    axes.append((m // 2**DEPTH, n // 2**DEPTH))
    axes.extend((m // 2**(level - 1), n // 2**(level - 1))
                for level in range(DEPTH, 1, -1))
    axes.extend((m // 2**level, n // 2**level) for level in range(DEPTH, 0, -1))
    require(len(specs) == len(axes) == 30, 'All original dependency bindings included')
    total = 0
    for (start, stop, old_shape), (rows, columns) in zip(specs, axes):
        require(0 < rows <= old_shape[-2] and 0 < columns <= old_shape[-1]
                and math.prod(old_shape) == stop - start,
                'Legal exact prefix of the original binding')
        count = math.prod((*old_shape[:-2], rows, columns))
        require(0 <= start < start + count <= stop <= ARENA_ELEMENTS,
                'Original interval and arena preserved')
        total += count
    return total


def cost_certificate():
    d, m, k = DEPTH, *LEFT_SHAPE
    gamma_numerator = sum(7**j * 4**(d-j) for j in range(1, d+1))
    left = gamma_numerator * m * k // 4**d
    copied = 7**d * m * k // 4**d
    prices = []
    for _, n in RIGHT_SHAPES:
        leaf = 7**d * (m // 2**d) * (n // 2**d) * (2*k // 2**d - 1)
        transforms = gamma_numerator * (m*k + k*n + m*n) // 4**d
        prices.append(leaf + transforms)
    require(tuple(prices) == PRODUCT_FLOPS and left == LEFT_TRANSFORM_FLOPS
            and copied == CACHE_COPY_FLOPS, 'Independent exact fixed product prices')
    require((prices[0] + copied, prices[1] - left) == REUSE_PRODUCT_FLOPS,
            'Creator copy and hit omission both charged')
    return dict(depth=d, layer=LAYER, left_shape=list(LEFT_SHAPE),
        right_shapes=[list(shape) for shape in RIGHT_SHAPES],
        original_product_flops=prices, reuse_product_flops=list(REUSE_PRODUCT_FLOPS),
        original_pair_flops=sum(prices), reuse_pair_flops=sum(REUSE_PRODUCT_FLOPS),
        net_pair_saving=left-copied, original_arithmetic_calls=182,
        reused_arithmetic_calls=153, removed_left_calls=30, added_copy_calls=1,
        cache_copy_flops=copied, cache_capacity_bytes=4*math.prod(CACHE_SHAPE),
        actual_copied_bytes=4*copied, logical_copy_read_write_bytes=8*copied,
        original_arena_bytes=4*ARENA_ELEMENTS, setup_flops=SETUP_FLOPS,
        original_binding_flops=BASE_BIND_FLOPS, compact_binding_flops=COMPACT_BIND_FLOPS,
        input_snapshot_flops=7237629, four_output_snapshot_flops=8650748,
        fixed_subtotal_before_fixture_flops=3521584817,
        fixture_and_any_extra_diagnostics_must_be_paid=True,
        metadata_rpc_calls_included_in_arithmetic_counts=False,
        full_card_or_runtime_qualified=False)


_OLD_HEADER = 'def _packed_product_cached(a,b,out,views,levels):'
_NEW_HEADER = 'def _w5_product_body(a,b,out,views,levels,cache):'
_LEAF_LINE = '    fnp.matmul(leaf_a,leaf_b,out=leaf)'
_COPY_PREFIX = '''    if cache._leaf is None:
        backing = fnp.empty((7,7,7,7,7,32,32),dtype=fnp.float32)
        cached = backing[..., :32, :31]
        cache._own_leaf(backing,cached)
        fnp.copyto(cached,leaf_a)
        del backing,cached
    else:
        leaf_a = cache._leaf
'''


def source_bundle(tick=None):
    source = _read(PARENT, PARENT_SHA, tick)
    _read(DESIGN, DESIGN_SHA, tick)
    functions = {name: _function(source, name) for name in _FUNCTION_NAMES}
    literals = {name: _literal(source, name, count)
                for name, count in _LITERAL_COUNTS.items()}
    require(literals['_PACKED_ELEMENTS'][1] == ARENA_ELEMENTS, 'Exact actual arena')
    specs = literals['_PACKED_SQUARE5'][1]
    require(_compact_price(specs, 1024, 1024, 1024) == BASE_BIND_FLOPS
            and _compact_price(specs, 1024, 1024, 512) == COMPACT_BIND_FLOPS
            and specs[8][2] == CACHE_SHAPE, 'Exact full/compact setup and A5 layout')
    product = functions['_packed_product_cached']
    start_text = '        fnp.add(a21,a22,out=aa[4])\n'
    end_text = '        fnp.subtract(b12,b11,out=bb[4])\n'
    require(product.count(start_text) == product.count(end_text) == 1
            and product.count(_OLD_HEADER) == product.count(_LEAF_LINE) == 1,
            'Unique complete source-splice anchors')
    start, stop = product.index(start_text), product.index(end_text)
    left = product[start:stop]
    require(left.count('fnp.') == 6 and left.splitlines()[-1].strip()
            == 'fnp.copyto(aa[3],a22)', 'All six original left primitives')
    guarded = '        if cache._leaf is None:\n' + ''.join(
        '    ' + line for line in left.splitlines(keepends=True))
    adapted = product.replace(_OLD_HEADER, _NEW_HEADER, 1)
    adapted = adapted.replace(left, guarded, 1)
    adapted = adapted.replace(_LEAF_LINE, _COPY_PREFIX + _LEAF_LINE, 1)
    inverse = adapted.replace(_COPY_PREFIX + _LEAF_LINE, _LEAF_LINE, 1)
    inverse = inverse.replace(guarded, left, 1).replace(_NEW_HEADER, _OLD_HEADER, 1)
    require(inverse == product, 'Complete original product recovered by positional inverse')
    # Parse declarations only: no estimator execution, imports, arrays or calls.
    tree = ast.parse(adapted)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
            'Inert adapted function declaration only')
    bundle = '\n'.join(line for line, _ in literals.values()) + '\n\n'
    bundle += '\n\n'.join(functions.values()) + '\n\n' + adapted + '\n'
    proof = dict(parent_sha256=PARENT_SHA, design_sha256=DESIGN_SHA,
        source_sha256=_sha(bundle.encode()),
        source_bindings={str(path): digest for path, digest in PINS.items()},
        original_function_sha256={name: _sha(text.encode())
                                  for name, text in functions.items()},
        original_product_sha256=_sha(product.encode()),
        adapted_product_sha256=_sha(adapted.encode()),
        original_left_transform_sha256=_sha(left.encode()),
        complete_original_product_inverse_verified=True,
        original_frame_and_compact_hook_byte_exact=True,
        unchanged_right_leaf_and_backward_text=True,
        copied_leaf_before_creator_matmul_and_unwind=True,
        no_left_transpose_side_substitution=True,
        cached_data_excluded_from_persistent_frames=True,
        fixed_unit_only=True, physical_gathered_W_integration_qualified=False,
        runtime_strides_not_inferred_from_remote_strides_property=True,
        cost_certificate=cost_certificate(), numerical_execution=False)
    if tick:
        tick()
    return bundle, proof


class _Views(tuple):
    """Original bound array handles, with a non-numerical unit namespace owner."""
    def __new__(cls, values, env):
        result = super().__new__(cls, values)
        result.env = env
        return result


def _track_arrays(value, refs):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        refs.append(weakref.ref(value))
    elif isinstance(value, dict):
        for item in value.values():
            _track_arrays(item, refs)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _track_arrays(item, refs)


def namespace(fnp):
    source, proof = source_bundle()
    env = dict(fnp=fnp, _SCRATCH=None, _PACKED_FRAME_CACHE={},
               _CS_CAPACITY_VIEWS={}, _CS_BINDING_FLOPS=0, _CS_BINDING_CALLS=0,
               _W5_VIEWS=None, _W5_BUSY=False, _W5_ACTIVE_CACHE=None,
               _W5_READY=False, _W5_RETIRED=False, _W5_ARRAY_REFS=[])
    # Only the authenticated literals/functions above are installed. No model.
    exec(compile(source, str(HERE / 'fixed_d5_source.py'), 'exec'), env)
    return env, proof


def setup(fnp, env):
    require(env['fnp'] is fnp and env['_SCRATCH'] is None
            and not env['_W5_READY'] and not env['_W5_RETIRED'], 'One fresh unit setup')
    arena = fnp.empty((ARENA_ELEMENTS,), dtype=fnp.float32)
    env['_W5_ARRAY_REFS'].append(weakref.ref(arena))
    env['_SCRATCH'] = [arena, None, None, None, None, None, None]
    try:
        views = _Views(env['_cp3_bind_arena_views'](arena, env['_PACKED_SQUARE5']), env)
        env['_SCRATCH'][5] = env['_W5_VIEWS'] = views
        # These are actual basic views and one compact binding, not transforms.
        for n in (1024, 384):
            env['_packed_frame'](DEPTH, *LEFT_SHAPE, n, views)
        require(len(env['_PACKED_FRAME_CACHE']) == 2
                and len(env['_CS_CAPACITY_VIEWS']) == 1
                and env['_CS_BINDING_FLOPS'] == COMPACT_BIND_FLOPS
                and env['_CS_BINDING_CALLS'] == 1, 'Exact two prepared CB2 frames')
        _track_arrays(views, env['_W5_ARRAY_REFS'])
        _track_arrays(env['_CS_CAPACITY_VIEWS'], env['_W5_ARRAY_REFS'])
        _track_arrays(env['_PACKED_FRAME_CACHE'], env['_W5_ARRAY_REFS'])
        env['_W5_READY'] = True
        return arena, views
    except BaseException:
        clear(env)
        raise


class _LayerCache:
    """One creator/hit pair. All retained numerical values are layer-local.

    For this synthetic unit w_owner IS the immutable L input, not an unloaded
    physical W matrix. previous_owner and network_token are opaque generation
    owners. Full-card separately gathered-W provenance is not implemented.
    """
    def __init__(self, network_token, layer, w_owner, previous_owner):
        self._network = network_token
        self._layer = layer
        self._left = w_owner
        self._previous = previous_owner
        self._env = None
        self._backing = self._leaf = None
        self.state = 'new'
        self.events = []
        self.array_refs = []

    def _own_leaf(self, backing, leaf):
        require(self.state == 'creating' and self._leaf is None
                and tuple(backing.shape) == CACHE_SHAPE and tuple(leaf.shape) == LEAF_SHAPE
                and str(backing.dtype) == str(leaf.dtype) == 'float32',
                'Exactly one same-stride copied leaf owner')
        self._backing, self._leaf = backing, leaf
        self.array_refs.extend((weakref.ref(backing), weakref.ref(leaf)))


def begin_layer(network_token, layer, w_owner, previous_owner):
    require(network_token is not None and previous_owner is not None
            and network_token is not w_owner and previous_owner is not w_owner
            and type(layer) is int and layer == LAYER,
            'Fixed unit layer and explicit distinct provenance owners')
    require(tuple(w_owner.shape) == LEFT_SHAPE and str(w_owner.dtype) == 'float32',
            'The fixed synthetic L itself is the immutable unit source owner')
    return _LayerCache(network_token, layer, w_owner, previous_owner)


def product_d5(a, b, out, views, cache, *, role, event):
    require(isinstance(views, _Views) and views.env is not None,
            'Actual live setup-owned original view tuple')
    env = views.env
    require(isinstance(cache, _LayerCache) and cache.state in ('new', 'ready')
            and callable(event) and env['_W5_READY'] and not env['_W5_RETIRED']
            and not env['_W5_BUSY'] and views is env['_SCRATCH'][5]
            and views is env['_W5_VIEWS'], 'Live serial fixed-unit ownership')
    ordinal = 0 if cache.state == 'new' else 1
    require(role == ROLES[ordinal] and a is cache._left
            and cache._network is not None and cache._previous is not None
            and cache._layer == LAYER and tuple(a.shape) == LEFT_SHAPE
            and tuple(b.shape) == RIGHT_SHAPES[ordinal]
            and tuple(out.shape) == (1024, RIGHT_SHAPES[ordinal][1])
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and a is not b and out is not a and out is not b,
            'Exact source owner, fixed role, orientation, geometry and FP32 dtype')
    require((ordinal == 0 and cache._env is None and env['_W5_ACTIVE_CACHE'] is None)
            or (ordinal == 1 and cache._env is env and env['_W5_ACTIVE_CACHE'] is cache
                and cache._leaf is not None), 'One creator then one same-namespace hit')
    require(len(env['_PACKED_FRAME_CACHE']) == 2
            and len(env['_CS_CAPACITY_VIEWS']) == 1, 'No frame recreation in the fixed pair')
    if ordinal == 0:
        cache._env = env
        env['_W5_ACTIVE_CACHE'] = cache
    env['_W5_BUSY'] = True
    cache.state = 'creating' if ordinal == 0 else 'using'
    try:
        result = env['_w5_product_body'](a, b, out, views, DEPTH, cache)
        require(result is out and cache._leaf is not None, 'Original output and retained leaf')
        entry = dict(kind='creator' if ordinal == 0 else 'hit',
            generation=id(cache._network), layer=LAYER, role=role, ordinal=ordinal,
            left_owner=id(cache._left), previous_owner=id(cache._previous),
            source_owner=id(a), shape=[1024, 992, RIGHT_SHAPES[ordinal][1]],
            depth=DEPTH, side='left', layout='original_SQ5_A5_capacity_1024x1024',
            leaf_shape=list(LEAF_SHAPE), backing_shape=list(CACHE_SHAPE),
            capacity_bytes=4*math.prod(CACHE_SHAPE),
            copied_elements=CACHE_COPY_FLOPS if ordinal == 0 else 0,
            skipped_left_arithmetic_calls=0 if ordinal == 0 else 30,
            expected_product_flops=REUSE_PRODUCT_FLOPS[ordinal])
        cache.events.append(entry)
        event(dict(entry))  # Metadata-only callback; no FNP, scans or strong arrays.
        cache.state = 'ready' if ordinal == 0 else 'complete'
        return result
    except BaseException:
        cache.state = 'failed'
        raise
    finally:
        env['_W5_BUSY'] = False


def close_layer(cache):
    require(isinstance(cache, _LayerCache), 'Actual layer cache')
    if cache.state == 'closed':
        return
    env = cache._env
    require(env is None or not env['_W5_BUSY'], 'Cannot retire an executing product')
    if env is not None:
        require(env['_W5_ACTIVE_CACHE'] is cache, 'Exact active cache owner')
        env['_W5_ACTIVE_CACHE'] = None
    cache._leaf = cache._backing = None
    cache._left = cache._previous = cache._network = cache._env = None
    cache.state = 'closed'


def clear(env):
    require(not env['_W5_BUSY'], 'Cannot clear an executing arena')
    cache = env['_W5_ACTIVE_CACHE']
    if cache is not None:
        close_layer(cache)
    env['_PACKED_FRAME_CACHE'].clear()
    env['_CS_CAPACITY_VIEWS'].clear()
    views = env['_W5_VIEWS']
    if views is not None:
        views.env = None
    env['_W5_VIEWS'] = None
    env['_SCRATCH'] = None
    env['_W5_READY'] = False
    env['_W5_RETIRED'] = True
    # The caller must also drop setup's returned arena/views and all outputs.
    # Only scalar counters and weak refs intentionally survive in this env.
