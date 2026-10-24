"""DRAFT inert CB2 source composer: four explicitly granted W-left D5 sites.

Only derive_source() returns text; no candidate is written/imported/executed.
The actual fixed unit is supporting evidence, not full-card qualification.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'current_comp_compact_boxes_v2/rpc37_v1/estimator.py'
PARENT_SHA = 'b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'
HELPER = HERE / 'helper.py'
HELPER_SHA = 'd44114c432758b7be5a082990de0356060d3be7dc449818babfe3fe3526aa8d7'
UNIT = HERE / 'unit_v2/remote'
WATCH = HERE / 'unit_watch_v2'
PINS = {
    PARENT: PARENT_SHA, HELPER: HELPER_SHA,
    HERE / 'DESIGN.md': '63aaf29623a3f729bac9c6827bac7bea5a267141ab3048f2356e81610adfc06e',
    UNIT / 'result.json': '60761f0a2151f62c22a9a1dcef1f18a3c91db0d18b7ce7d3001dc44b17d27beb',
    UNIT / 'completion.json': '2cf5c677d0b81410255d936d71c1329972842edb43a05744da4b7325d3a0baf5',
    UNIT / 'finalization.json': '47fa106df2d53549c1c26cdfdbe96b16d24b23e62d86821462fc34ed7b03410f',
    WATCH / 'result.json': '997216d8f5238e1c5ee705a26f2a7ac8d6bae57d87fa3b4e1cb92e9022ab73f2',
    WATCH / 'completion.json': '609c2cb11df268d86220a5dbc4df8990ec07ffb04b2141024432012b9788fae5',
    WATCH / 'finalization.json': '0b5d0026046b0a0f88ef86de5a71b5dfca5794258914a49bbe71abb7305004bf',
    UNIT / 'call_00.json': '15d69fe68c168b013536b0f98d73349dedb718f210789d525fe2cdcd29c1dd44',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read(path, digest, tick=None):
    if tick:
        tick()
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 4 * 1024**2, 'Bounded regular pinned owner')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'Exact owner: ' + str(path))
    if tick:
        tick()
    return raw.decode('utf-8')


def authenticate(tick=None):
    """Direct source/unit evidence only; no ancestral replay or numerical import."""
    for path, digest in PINS.items():
        _read(path, digest, tick)
    return {str(path): digest for path, digest in PINS.items()}


def _function_span(source, name, last=False):
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith('def ' + name + '(')]
    require(starts and (last or len(starts) == 1), 'Exact source definition: ' + name)
    start = starts[-1]
    stop = start + 1
    while stop < len(lines) and (not lines[stop].strip() or lines[stop][0].isspace()):
        stop += 1
    text = ''.join(lines[start:stop]).rstrip('\n')
    tree = ast.parse(text)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == name, 'Complete selected function, not a prefix')
    offset = sum(map(len, lines[:start]))
    return offset, offset + len(text), text


def _replace_once(text, before, after):
    require(text.count(before) == 1, 'One exact source splice: ' + before[:80])
    return text.replace(before, after, 1)


def _helper_literals(source):
    tree = ast.parse(source)
    names = ('_OLD_HEADER', '_NEW_HEADER', '_LEAF_LINE', '_COPY_PREFIX')
    result = {}
    for name in names:
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == name]
        require(len(nodes) == 1, 'Actual qualified helper literal: ' + name)
        result[name] = ast.literal_eval(nodes[0].value)
    return result


def _unit_admission(texts):
    worker = json.loads(texts[UNIT / 'result.json'])
    watch = json.loads(texts[WATCH / 'result.json'])
    call = json.loads(texts[UNIT / 'call_00.json'])
    require(worker['passed'] is True and worker['completed'] is True
            and worker['source_bindings'][str(HELPER)] == HELPER_SHA
            and worker['source_bindings'][str(PARENT)] == PARENT_SHA,
            'Actual positive fixed-unit source ownership')
    require(watch['passed'] is True and watch['completed'] is True
            and watch['whole_owned_tree_retired'] is True
            and watch['observational_groups_retired'] is True
            and watch['worker_seals'] == {
                'result_sha256': PINS[UNIT / 'result.json'],
                'completion_sha256': PINS[UNIT / 'completion.json'],
                'finalization_sha256': PINS[UNIT / 'finalization.json']},
            'Direct qualified unit supervisor and worker join')
    require(call['passed'] is True and call['complete_flops'] == 3603611321
            and [len(row['primitive_calls']) for row in call['rows']] == [91, 91, 92, 61]
            and call['rows'][0]['raw_sha256'] == call['rows'][2]['raw_sha256']
            and call['rows'][1]['raw_sha256'] == call['rows'][3]['raw_sha256'],
            'Actual fixed-unit two-RHS parity and full own bill')


# This fragment contains declarations/scalar metadata only at module scope.
# Numeric cache owners exist solely inside a participant's paid prediction.
RUNTIME = '''# BEGIN EXPLICIT FOUR-SITE W-LEFT D5 REUSE V1
_W5_PORT_SCOPE = None
_W5_PORT_LAST = None


class _W5_PortCache:
    def __init__(self, key, m, k, cm, ck):
        self._key = key
        self._capacity = (7,7,7,7,7,cm//32,ck//32)
        self._shape = (7,7,7,7,7,m//32,k//32)
        self._backing = self._leaf = None
        self._state = 'new'
        self.creators = self.hits = self.copy_flops = self.removed_flops = 0

    def _own_leaf(self, backing, leaf):
        if (self._state != 'creating' or self._leaf is not None
                or tuple(backing.shape) != self._capacity
                or tuple(leaf.shape) != self._shape
                or str(backing.dtype) != 'float32' or str(leaf.dtype) != 'float32'):
            raise RuntimeError('W5 exact typed copied-leaf owner')
        self._backing, self._leaf = backing, leaf

    def close(self):
        self._leaf = self._backing = None
        self._state = 'closed'


def _w5_port_new(network):
    global _W5_PORT_LAST
    _W5_PORT_LAST = None
    return dict(network=network, generation=object(), layer=None,
                weight=None, previous=None, cache=None, grant=None,
                roles={}, records=[], failed=False)


def _w5_port_begin(scope, network, layer, weight, previous):
    if (scope is None or scope['network'] is not network
            or scope['layer'] is not None or scope['cache'] is not None
            or scope['grant'] is not None or type(layer) is not int
            or tuple(weight.shape) != (1024,1024)
            or str(weight.dtype) != 'float32' or len(previous.shape) != 1):
        raise RuntimeError('W5 physical network/layer/source owner')
    scope['layer'], scope['weight'], scope['previous'] = layer, weight, previous
    scope['roles'] = {}


def _w5_port_call(scope, weight, previous, role, function, left, right):
    # Called only at four explicit source sites. Shape equality is not a grant.
    if scope is None:
        return function(left,right)
    expected = _shared_old384_mm if role == 'shared_basis' else _mm
    if (scope['layer'] is None or scope['weight'] is not weight
            or scope['previous'] is not previous or scope['generation'] is None
            or scope['network'] is None or scope['grant'] is not None
            or role not in ('cov_outer','shared_basis','newborn_common','star_push')
            or function is not expected or len(left.shape) != 2 or len(right.shape) != 2
            or tuple(left.shape) != (1024,int(previous.shape[0]))
            or int(right.shape[0]) != int(previous.shape[0])):
        raise RuntimeError('W5 explicit immutable W/index provenance')
    scope['roles'][role] = scope['roles'].get(role,0)+1
    # No changed tiling, padding, dtype or schedule. Original dispatch decides.
    if (str(left.dtype) != 'float32' or str(right.dtype) != 'float32'
            or max(int(left.shape[0]),int(left.shape[1]),int(right.shape[1])) > 1024):
        return function(left,right)
    scope['grant'] = dict(left=left,right=right,role=role,used=False)
    try:
        return function(left,right)
    finally:
        # Do not retain any separately gathered input or result in a cache.
        scope['grant'] = None


def _w5_port_product(a,b,out,views,scope):
    grant = scope['grant']
    m,k,n = int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    if (grant is None or grant['used'] or a is not grant['left'] or b is not grant['right']
            or views is not _SCRATCH[5] or m != 1024 or k != int(scope['previous'].shape[0])
            or min(m,k,n) <= 0 or max(m,k,n) > 1024 or m%32 or k%32 or n%32
            or tuple(out.shape) != (m,n)
            or str(a.dtype) != 'float32' or str(b.dtype) != 'float32'
            or str(out.dtype) != 'float32'):
        raise RuntimeError('W5 original unpadded square-D5 dispatch identity')
    cm,ck = 1<<(m-1).bit_length(),1<<(k-1).bit_length()
    key = (id(scope['generation']),id(scope['weight']),id(scope['previous']),
           scope['layer'],id(views),5,m,k,cm,ck)
    cache = scope['cache']
    if cache is None:
        cache = scope['cache'] = _W5_PortCache(key,m,k,cm,ck)
    elif cache._key != key or cache._state != 'ready':
        raise RuntimeError('W5 no cross-owner/layer/layout reuse')
    creator = cache._leaf is None
    cache._state = 'creating' if creator else 'using'
    grant['used'] = True
    try:
        result = _w5_port_product_body(a,b,out,views,5,cache)
        if result is not out or cache._leaf is None:
            raise RuntimeError('W5 original returned output and copied leaf')
        if creator:
            cache.creators += 1
            cache.copy_flops += 16807*(m//32)*(k//32)
        else:
            cache.hits += 1
            cache.removed_flops += 36827*m*k//1024
        cache._state = 'ready'
        return result
    except BaseException:
        cache._state = 'failed'
        raise


def _w5_port_end_layer(scope):
    if scope['layer'] is None:
        return
    cache = scope['cache']
    try:
        if scope['grant'] is not None:
            raise RuntimeError('W5 outstanding call grant at layer retirement')
        row = dict(layer=scope['layer'],previous_count=int(scope['previous'].shape[0]),
                   roles=dict(scope['roles']),creators=0,hits=0,copy_flops=0,
                   removed_flops=0,capacity_bytes=0)
        if cache is not None:
            capacity_elements = 1
            for extent in cache._capacity:
                capacity_elements *= extent
            row.update(creators=cache.creators,hits=cache.hits,copy_flops=cache.copy_flops,
                       removed_flops=cache.removed_flops,capacity_bytes=4*capacity_elements)
        scope['records'].append(row)
    finally:
        if cache is not None:
            cache.close()
        scope['cache'] = scope['grant'] = None
        scope['weight'] = scope['previous'] = None
        scope['layer'] = None


def _w5_port_finish(scope):
    global _W5_PORT_LAST
    try:
        _w5_port_end_layer(scope)
        records = scope['records']
        _W5_PORT_LAST = dict(completed=not scope['failed'],layers=records,
            creators=sum(row['creators'] for row in records),
            hits=sum(row['hits'] for row in records),
            copied_flops=sum(row['copy_flops'] for row in records),
            removed_flops=sum(row['removed_flops'] for row in records),
            maximum_cache_bytes=max([0]+[row['capacity_bytes'] for row in records]),
            numeric_cache_retired=True,additional_initialization_flops=0)
    finally:
        scope['network'] = scope['generation'] = None
        scope['weight'] = scope['previous'] = scope['grant'] = scope['cache'] = None
# END EXPLICIT FOUR-SITE W-LEFT D5 REUSE V1
'''


def derive_source(tick=None):
    """Return full candidate TEXT and exact inverse proof; never materialize it."""
    texts = {path: _read(path, digest, tick) for path, digest in PINS.items()}
    _unit_admission(texts)
    parent, helper = texts[PARENT], texts[HELPER]
    require('_W5_PORT_SCOPE' not in parent and '_w5_port_' not in parent,
            'Fresh additive names, not a second application')
    literals = _helper_literals(helper)
    _, _, product = _function_span(parent, '_packed_product_cached')
    old_header, leaf_line = literals['_OLD_HEADER'], literals['_LEAF_LINE']
    require(product.startswith(old_header), 'Same original product as qualified unit')
    a = product.index('        fnp.add(a21,a22,out=aa[4])\n')
    b = product.index('        fnp.subtract(b12,b11,out=bb[4])\n')
    left = product[a:b]
    require(left.count('fnp.') == 6, 'All six exact left-transform primitives')
    guarded = '        if cache._leaf is None:\n' + ''.join(
        '    '+line for line in left.splitlines(keepends=True))
    copy = literals['_COPY_PREFIX']
    dynamic_copy = _replace_once(copy,
        'fnp.empty((7,7,7,7,7,32,32),dtype=fnp.float32)',
        'fnp.empty(cache._capacity,dtype=fnp.float32)')
    dynamic_copy = _replace_once(dynamic_copy,
        'cached = backing[..., :32, :31]', 'cached = backing[..., :m//32, :k//32]')
    require(dynamic_copy.replace('fnp.empty(cache._capacity,dtype=fnp.float32)',
        'fnp.empty((7,7,7,7,7,32,32),dtype=fnp.float32)',1).replace(
        'cached = backing[..., :m//32, :k//32]', 'cached = backing[..., :32, :31]',1)
        == copy, 'Exact qualified-unit copy generalization inverse')
    new_header = 'def _w5_port_product_body(a,b,out,views,levels,cache):'
    adapted = _replace_once(product,old_header,new_header)
    adapted = _replace_once(adapted,left,guarded)
    adapted = _replace_once(adapted,leaf_line,dynamic_copy+leaf_line)
    inverse_product = adapted.replace(dynamic_copy+leaf_line,leaf_line,1)
    inverse_product = inverse_product.replace(guarded,left,1).replace(new_header,old_header,1)
    require(inverse_product == product, 'Complete original packed product inverse')

    # Check actual A-capacity construction rather than the remote .strides property.
    axes = _function_span(parent,'_cs_shape_axes')[2]
    frame = _function_span(parent,'_packed_frame')[2]
    require('axes.extend(((m//2**level,k//2**level),(k//2**level,n//2**level)))' in axes
            and 'av = views[cursor:cursor+2*levels:2]' in frame
            and 'na = av[index][..., :m2, :k2]' in frame
            and 'frame = (forward, old_a, old_b, leaf, backward)' in frame,
            'Original left shapes and leaf depend on m/k, not RHS width')
    sq5_lines = [line for line in parent.splitlines() if line.startswith('_PACKED_SQUARE5 = ')]
    require(len(sq5_lines) == 1, 'Exact current SQ5 capacity owner')
    specs = ast.literal_eval(sq5_lines[0].split(' = ',1)[1])
    require(all(specs[2*(level-1)][2] == (7,)*level+(1024//2**level,1024//2**level)
                for level in range(1,6)) and specs[8][2] == (7,7,7,7,7,32,32),
            'One n-independent original-capacity A layout at every level')

    replacements = {
        '_cov_compact': (
            'result = _mm(wa.T, _mm(ca, wa))',
            "result = _w5_port_call(_W5_PORT_SCOPE,w,indices,'cov_outer',_mm,wa.T,_mm(ca,wa))"),
        '_star_compact': (
            'pushed = _mm(wp.T, aa)',
            "pushed = _w5_port_call(_W5_PORT_SCOPE,w,previous,'star_push',_mm,wp.T,aa)"),
        '_push_bank_shared_newborn': (
            'common = _mm(wp,base[previous,:])',
            "common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])"),
        '_shared_old384_push': (
            'U = _shared_old384_mm(w32[previous, :].T, active_basis)',
            "U = _w5_port_call(_W5_PORT_SCOPE,w32,previous,'shared_basis',_shared_old384_mm,w32[previous, :].T,active_basis)"),
    }
    edits = []
    for name, (before, after) in replacements.items():
        start, stop, old = _function_span(parent,name)
        edits.append((start,stop,old,_replace_once(old,before,after)))
    start,stop,old = _function_span(parent,'_packed_product_5',last=True)
    require(old == 'def _packed_product_5(a,b,out,arena_views):\n'
            '    return _packed_product_cached(a,b,out,arena_views,5)',
            'Actual effective D5 entry, not an overwritten ancestor')
    new = '''def _packed_product_5(a,b,out,arena_views):
    scope = _W5_PORT_SCOPE
    if scope is None or scope['grant'] is None:
        return _packed_product_cached(a,b,out,arena_views,5)
    return _w5_port_product(a,b,out,arena_views,scope)'''
    edits.append((start,stop,old,new))

    start,stop,old = _function_span(parent,'_predict_binary',last=True)
    body = _replace_once(old,
        '        w32 = fnp.asarray(W[l], dtype=fnp.float32)\n',
        '        w32 = fnp.asarray(W[l], dtype=fnp.float32)\n'
        '        _w5_port_begin(_w5_context,mlp,l,w32,previous)\n')
    star = '        star_pre = fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else _star_compact(w32, star, previous, previous_alive, live_indices)\n'
    body = _replace_once(body,star,star+'        _w5_port_end_layer(_w5_context)\n')
    header, original_body = body.split('\n',1)
    new = header+'''
    global _W5_PORT_SCOPE
    if _W5_PORT_SCOPE is not None:
        raise RuntimeError('W5 no concurrent or nested network prediction')
    _w5_context = _w5_port_new(mlp)
    _W5_PORT_SCOPE = _w5_context
    try:
'''+''.join('    '+line if line.strip() else line
            for line in original_body.splitlines(keepends=True))+'''
    except BaseException:
        _w5_context['failed'] = True
        raise
    finally:
        try:
            _w5_port_finish(_w5_context)
        finally:
            _W5_PORT_SCOPE = None'''
    edits.append((start,stop,old,new))

    changed, applied = parent, []
    for start,stop,before,after in sorted(edits,reverse=True):
        require(changed[start:stop] == before, 'Exact disjoint source function splice')
        changed = changed[:start]+after+changed[stop:]
        applied.append((start,before,after))
    suffix = '\n\n'+RUNTIME+'\n\n'+adapted+'\n'
    changed += suffix
    restored = changed[:-len(suffix)]
    for start,before,after in reversed(applied):
        require(restored[start:start+len(after)] == after, 'Exact positional inverse owner')
        restored = restored[:start]+before+restored[start+len(after):]
    require(restored == parent, 'Entire CB2 restored byte-for-byte')
    # No compile/exec here; only parse the new small definitions for source validity.
    for _,_,_,after in edits:
        ast.parse(after)
    ast.parse(RUNTIME+'\n'+adapted)
    for name in ('_mm','_square_product','_packed_frame','_packed_product_cached',
                 '_cs_shape_axes','_cs_compact_specs','_cs_compact_views','_shared_old384_mm'):
        require(_function_span(parent,name,last=True)[2] ==
                _function_span(changed,name,last=True)[2], 'Unchanged full owner: '+name)
    proof = dict(schema='current_comp_weight_D5_port_source_v1',
        parent_sha256=PARENT_SHA,candidate_sha256=sha(changed),source_bytes=len(changed.encode()),
        source_bindings={str(path):digest for path,digest in PINS.items()},
        full_source_inverse_verified=True,original_product_inverse_verified=True,
        qualified_copy_generalization_inverse_verified=True,
        source_splice_count=len(edits),explicit_granted_sites=list(replacements),
        original_dispatch_tiling_padding_and_guards_unchanged=True,
        all_original_gathers_heads_selectors_and_consumers_preserved=True,
        covariance_inner_evaluated_before_grant=True,
        depth_other_than_5_unchanged=True,original_arena_and_all_bindings_preserved=True,
        n_independent_left_capacity_proved_from_source=True,
        cache_keys_per_layer_at_most=1,maximum_cache_backing_bytes=68841472,
        cache_retired_after_star_before_nonlinear_and_in_call_finally=True,
        module_scope_numerical_work=False,additional_initialization_flops=0,
        conditional_rank37_creators=15,conditional_rank37_hits=41,
        conditional_rank37_copy_flops=230188672,
        conditional_rank37_removed_flops=1363482848,
        conditional_rank37_net_saved_flops=1133294176,
        conditional_rank37_cold_flops=410097374959,
        conditional_rank37_warm_flops=403321418965,
        conditional_rank37_initialization_flops=6775955994,
        expected_add_sub_stack_copy_calls_saved=1215,
        actual_full_card_bytes_bills_domains_and_resource_gates_pending=True,
        full_call_runtime_retention_ratio_at_most=.99,
        release_wall_cap_s=45,release_historical_ratio_at_most=.80,
        runtime_gain_measured=False,raw_value_measured=False,
        model_calls=0,parameter_changes=0,truth_accessed=False,
        materialization_or_submission_authorized=False)
    return changed, proof
