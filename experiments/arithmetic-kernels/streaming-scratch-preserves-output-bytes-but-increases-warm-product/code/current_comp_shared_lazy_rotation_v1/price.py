"""DRAFT source/JSON-only complete conditional tariff, never an estimator bill.

Explicit calculate()/main reads scalar metadata and prints JSON. It creates no
file, candidate, stage or process and imports no numerical package or estimator.
The selected frozen price-owner FUNCTIONS, not its main/calculate, are reused.
"""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
SOURCE = NEW/'current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py'
SOURCE_SHA = '378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
FRAGMENT = HERE/'fnp.py'
FRAGMENT_SHA = 'e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92'
OWNER = NEW/'current_comp_shared_old384_v1/price_metadata.py'
OWNER_SHA = '5538dba5a46169009eafe0b3d51cfa80e6b7cb1f2eb8e51c70dadfbc247fa530'
NATIVE = NEW/'current_comp_shared_gain_slice_v1/native40_v2/result.json'
NATIVE_SHA = '95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e'
WATCH = NEW/'current_comp_shared_gain_slice_v1/native40_watch_v2/result.json'
WATCH_SHA = '1bc1135f275609a4ea64bd7414de2cae9881f9396af3b561f6af6e1fe2bc7bfb'
C2_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
BOX_EXTRA_COLD = 1342018272
INITIALIZATION_FLOPS = 3286988570


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path, digest, tick=None):
    path = Path(path)
    if tick is not None:
        tick()
    require(path.is_file() and not path.is_symlink(), 'Regular metadata/source')
    before = path.stat()
    require(0 < before.st_size <= 16*1024**2, 'Bounded metadata/source size')
    parts, hashed = [], hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            parts.append(block)
            hashed.update(block)
            require(sum(map(len,parts)) <= 16*1024**2, 'Read remains bounded')
            if tick is not None:
                tick()
    after = path.stat()
    ident = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    raw = b''.join(parts)
    require(ident(before)==ident(after) and len(raw)==before.st_size
            and hashed.hexdigest()==digest, 'Exact unchanged source/metadata bytes')
    return raw


def tariffs(tick=None):
    owner = ast.parse(read(OWNER,OWNER_SHA,tick).decode('utf-8'))
    selected = [node for node in owner.body if isinstance(node,ast.FunctionDef)
                and node.name in ('dense','packed','matrix_price')]
    require([node.name for node in selected]==['dense','packed','matrix_price'],
            'Exact original integer tariff functions only')
    namespace = dict(Fraction=Fraction)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(OWNER),'exec'),namespace)
    tree = ast.parse(read(SOURCE,SOURCE_SHA,tick).decode('utf-8'))
    tables = {}
    for node in tree.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in ('_SQUARE_EQUAL_DEPTHS','_SQUARE_CROSS_DEPTHS')):
            tables[node.targets[0].id] = ast.literal_eval(node.value)
    require(len(tables)==2, 'Actual LAST effective frozen square schedules')
    if tick is not None:
        tick()
    return lambda m,k,n: namespace['matrix_price'](m,k,n,tables,row_tiles=True)


def consumer(mm, s, R, j):
    """_slr_reconstruct + _slr_inner, against the materialized parent."""
    n, r, N = 1024, 384, R+j
    h = lambda z:2*z-1
    return dict(matrix=(mm(n,r,r)+mm(n,r,R)+mm(n,r,j)-mm(n,r,N)
                        +mm(s,R,r)+mm(s,j,r)+mm(s,r,r)-mm(s,N,r)),
                scalar=n*N+s*r+h(n*r)+h(s*r))


def layer_price(mm, layer, s, R, j, group):
    """Code-order tariff of the one fixed schedule; no alternative selection."""
    r = 384
    h = lambda z:2*z-1
    matrix = scalar = 0
    kind = 'unchanged'
    if layer in (8,10,12,14):
        require(group is None, 'One lazy map only')
        kind, group = 'lazy', (R,j)
        matrix -= mm(r,r,R)
        # Remove original factor concat and full factor finite scan; add FULL
        # finite scans of both actual blocks and map (_slr_finish_factors).
        scalar += -r*(R+j)-h(r*(R+j))+h(r*R)+h(r*j)+h(r*r)
        # Virtual-rotation envelope: two abs/max scans, two FP64 casts,
        # two typed FP64 full(1,1) constants, two multiplies, three comparisons,
        # and two Boolean conjunctions, including BOTH individual input maxima.
        # The final Boolean all of one element has zero reduction FLOPs.
        scalar += h(r*R)+h(r*r)+20
    elif layer in (9,11,13):
        require(group is not None and sum(group)==R, 'Next-join consolidation')
        old, previous_join = group
        kind, group = 'consolidate', None
        matrix += (mm(r,r,r)+mm(r,r,old)+mm(r,r,previous_join)-mm(r,r,R))
        scalar += h(r*r)  # composed map guard; factor concat/scan unchanged
    else:
        require(layer <= 7 or layer == 15, 'Fixed source schedule')
        if layer == 15:
            require(group is not None, 'Final L14 map used, not dropped')
            kind = 'terminal_lazy'
    if group is not None:
        used = consumer(mm,s,*group)
        matrix += used['matrix']
        scalar += used['scalar']
    return dict(layer=layer,kind=kind,matrix_delta=matrix,scalar_delta=scalar,
                delta=matrix+scalar,retained_group=list(group) if group else None), group


def calculate(tick=None):
    mm = tariffs(tick)
    fragment = read(FRAGMENT,FRAGMENT_SHA,tick)
    require(b"_SLR_LAZY_LAYERS = (8, 10, 12, 14)" in fragment
            and b"_SLR_CONSOLIDATE_LAYERS = (9, 11, 13)" in fragment,
            'Exact code-owned fixed schedule')
    native = json.loads(read(NATIVE,NATIVE_SHA,tick))
    watch = json.loads(read(WATCH,WATCH_SHA,tick))
    require(native['passed'] and native['completed'] and watch['passed']
            and watch['completed'] and native['full_model_calls']==41
            and native['candidate_sha256']==C2_SHA
            and native['initialization_flops']==1944970298,
            'Actual current-source scalar geometry/bill boundary')
    rows = []
    for call, trial in enumerate(native['trials'][:40]):
        item = native['diagnostic_records'][call]
        path = Path(item['path'])
        require(path.parent==NATIVE.parent and path.name==f'call_{call:02d}_diagnostics.json',
                'Exact scalar diagnostic path')
        diagnostic = json.loads(read(path,item['sha256'],tick))
        require(diagnostic['call']==call and diagnostic['rank']==trial['rank'],
                'Original call/rank metadata')
        cells = diagnostic['domains']['selection_cells']
        counts = [[cells[f'{l}:0']['count'], cells[f'{l}:1']['count'],
                   cells[f'{l}:2']['count'] if l<15 else 0] for l in range(16)]
        group, layers = None, []
        for l in range(5,16):
            R = 2*sum(c[2] for c in counts[:l-5 if l<15 else 10])
            j = 2*counts[l-5][2] if l<15 else 0
            item_price, group = layer_price(mm,l,counts[l][1],R,j,group)
            layers.append(item_price)
        delta = sum(item['delta'] for item in layers)
        cold = BOX_EXTRA_COLD if trial['includes_initialization'] else 0
        rows.append(dict(call=call,rank=trial['rank'],counts=counts,
            diagnostic_path=str(path),diagnostic_sha256=item['sha256'],
            actual_c2_complete_flops=trial['flops'],retained_BOX_cold_binding_flops=cold,
            conditional_BOX_baseline_flops=trial['flops']+cold,
            conditional_delta_flops=delta,
            conditional_lazy_BOX_complete_flops=trial['flops']+cold+delta,layers=layers))
    return dict(schema='current_comp_shared_lazy_rotation_source_price_v1',
        source_sha256=SOURCE_SHA,fragment_sha256=FRAGMENT_SHA,tariff_owner_sha256=OWNER_SHA,
        current_native40_sha256=NATIVE_SHA,current_native40_watch_sha256=WATCH_SHA,
        conditional_on_current_c2_geometry=True,complete_bill_measured=False,
        numerical_arrays_read=False,estimator_executed=False,truth_accessed=False,
        additional_cold_binding_flops=0,retained_initialization_flops=INITIALIZATION_FLOPS,
        mean_delta_flops=sum(row['conditional_delta_flops'] for row in rows)/40,
        conditional_mean_complete_flops=sum(row['conditional_lazy_BOX_complete_flops']
                                            for row in rows)/40,rows=rows)


if __name__ == '__main__':
    print(json.dumps(calculate(),sort_keys=True,indent=2,allow_nan=False))
