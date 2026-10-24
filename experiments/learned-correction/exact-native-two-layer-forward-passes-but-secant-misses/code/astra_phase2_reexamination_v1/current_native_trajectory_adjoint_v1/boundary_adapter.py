"""Inert native compact-bank -> existing Torch14 boundary coordinates.

No files, numerical imports, execution, targets or fitting at import. The
caller authenticates the actual producer/source/P/fixture/seals before passing
its manifest and detached bytes. This module verifies descriptor bytes and
coordinates; it does NOT grant freshness or scientific authority to a packet.

describe(manifest, 'entry8'|'end8') is stdlib-only. decode(np, torch, manifest,
blob, boundary, refs=...) returns (state14, plain_metadata) in an owned numerical
scope. entry8 uses ONLY head7.state; end8 uses ONLY head8.state and is entry9.
Required end8 p/Q/F/core must actually be saved: old fingerprint-only records
fail closed. No historical end state, reference or supplied moment is filled in.

State order is exactly CAUSAL/model.py:
 (m,C,p_full,lam,star,b,previous_alive,radial,hstate,Q,F,core,stops,previous).
CPU scalar state/mask is FP64; p/lam/star/H/Q/F/core are FP32; indices int64;
stops is an immutable scalar tuple. Boolean masks/indices/stops are branch
metadata, not differentiable coordinates. All other eleven tensor paths remain
explicit. Decode creates fresh values; it cannot differentiate through bytes.
The caller's native-forward/local-VJP node must supply the connected adjoint.

For selected row s_j, native base B[s_j,j] is exactly zero: _predict_binary
zeros bp's diagonal and _bd_update zeros both direction and change diagonals.
The native newborn tuple stores B[:,selected], t32=(t*alive)[selected], and
s32=(s*alive)[selected], already masked/rounded. Thus x=B+E*t, y=B+E*s, with
E[s_j,j]=1. expand_newborn performs a functional diagonal overwrite (equal
on this constrained domain), then concatenates inherited p,x,y. No regating,
coefficient recomputation, signed-pair averaging, or QF reconstruction occurs.
newborn_pullback returns Gp, offdiag(Gx+Gy), diag_s(Gx), diag_s(Gy). The removed
base diagonal is constrained zero, not an omitted physical degree of freedom.

This coordinate identity does NOT make a dense Torch push byte-equivalent to
native common+directions*t FP32 association. Forward must remain the actual
fresh native program. At a new P, every downstream native state is recomputed;
saved states are not a frozen teacher trajectory. Head pending coefficients
are consumed by coefficient_hook before entry/end boundaries; only H survives.

All copies, expansion, finite/domain checks and pullback arithmetic are caller-
owned research work. refs optionally records weak references for retirement.
These functions do not create contexts, bypass accounting, or authorize runs.
"""
import hashlib
import math
import re
import sys
import weakref

CANDIDATE_SHA256 = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
SHARED_MODEL_SHA256 = '65805a9ffbdb310fb374b82279b79d2f0d0931b00f45191fbb3d377b10802311'
SHARED_OPERATOR_SHA256 = '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2'
SCHEMA = 'current_native_trajectory_boundary_v1'
STATE_NAMES = ('m','C','p_full','lam','star','b','previous_alive','radial',
               'hstate','Q','F','core','stops','previous')
DIFFERENTIABLE_INDICES = (0,1,2,3,4,5,7,8,9,10,11)
WIDTH, SHARED_RANK, HIDDEN = 1024, 384, 64
MAX_PACKET_BYTES = 512 * 1024**2
ENTRY_COHORTS = ((0,2048),(2048,2048),(4096,2048),(6144,1984),
                 (8128,1856),(9984,1728),(11712,1664),(13376,1536))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _track(refs, *values):
    if refs is not None:
        for value in values:
            refs.append(weakref.ref(value))


def _plain(value):
    """Decode scalar chronology only; never array values or aliases."""
    if set(value) == {'scalar'}:
        answer = value['scalar']
        require(answer is None or type(answer) in (bool,int,str), 'Integer/bool/string/null metadata only')
        return answer
    if set(value) == {'mapping'}:
        return {key:_plain(v) for key,v in value['mapping'].items()}
    if set(value) == {'sequence','tuple'}:
        answer = [_plain(v) for v in value['sequence']]
        return tuple(answer) if value['tuple'] else answer
    raise RuntimeError('No numerical array/alias in scalar chronology')


def _row(value, path, packet_bytes, shape, dtype):
    require(set(value) == {'array'}, 'Actual stored array required: '+path)
    row = value['array']; sizes = {'float32':4,'float64':8,'int64':8,'bool':1}
    require(row.get('dtype') == dtype and row.get('shape') == list(shape),
            'Fixed native boundary dtype/shape: '+path)
    require(row.get('bytes') == math.prod(shape)*sizes[dtype]
            and row.get('byteorder') == sys.byteorder
            and isinstance(row.get('sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',row['sha256']), 'Typed native bytes: '+path)
    require(type(row.get('offset')) is int and row['offset'] >= 0
            and row['offset']+row['bytes'] <= packet_bytes
            and row.get('fingerprint_only',False) is False,
            'Required boundary bytes must be retained, not fingerprint-only: '+path)
    require(row.get('original_type_module') == 'flopscope._ndarray'
            and row.get('original_type_name') == 'FlopscopeArray'
            and (row.get('original_c_contiguous') or row.get('original_f_contiguous')),
            'Actual plain compact native boundary owner: '+path)
    return dict(row)


def describe(manifest, boundary):
    """Validate/return selected descriptor inventory and exact scalar chronology.

    No array decoding, file read or numerical API. This accepts a newly sealed
    additive recorder with the same packet descriptor convention, not a guessed
    future file path/hash or automatic reuse of the old negative/partial stage.
    """
    require(boundary in ('entry8','end8'), 'Only fixed entry8 or end8 boundary')
    require(manifest.get('candidate_sha256') == CANDIDATE_SHA256
            and manifest.get('rank') == 35 and manifest.get('layer') == 8
            and manifest.get('consumer') == 9 and manifest.get('truth_accessed') is False,
            'Fixed target-blind current-c2 rank35 cell8->9 source frame')
    size = manifest.get('bytes')
    require(type(size) is int and 0 < size <= MAX_PACKET_BYTES
            and isinstance(manifest.get('packet_sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',manifest['packet_sha256']), 'Complete packet byte inventory')
    next_layer, head = (8,'head7') if boundary == 'entry8' else (9,'head8')
    native = manifest['packets'][boundary]
    shared = native['shared_old']['mapping']
    cohorts = tuple(tuple(v) for v in _plain(shared['cohorts']))
    require(len(cohorts) == next_layer and cohorts[:8] == ENTRY_COHORTS,
            'Unchanged first eight cohorts, fresh ninth cohort only at end8')
    stops = []; end = 0
    for start,width in cohorts:
        require(type(start) is int and type(width) is int and start == end
                and 0 < width <= 2*WIDTH and width % 2 == 0, 'Intact chronological signed cohorts')
        end = start+width; stops.append(end)
    stops = tuple(stops); old_columns = stops[next_layer-6]
    require(_plain(shared['columns']) == old_columns and _plain(shared['next_column']) == end
            and _plain(shared['pre_basis']) is None, 'Post-commit shared boundary, no pending transported basis')
    joins = _plain(shared['joins'])
    require(len(joins) == next_layer-5, 'All and only completed original shared joins')
    for j,row in enumerate(joins):
        require(row['cohort'] == j and row['layer'] == j+5
                and row['start'] == (stops[j-1] if j else 0) and row['stop'] == stops[j]
                and type(row['previous_active']) is int and 384 <= row['previous_active'] <= WIDTH,
                'Exact original layer/cohort join chronology')
    nb = native['newborn']
    require(set(nb) == {'sequence','tuple'} and nb['tuple'] is True
            and len(nb['sequence']) == 5, 'Original five-field newborn tuple')
    nb = nb['sequence']; inherited = _plain(nb[0]); births = cohorts[-1][1]//2
    require(inherited == stops[-2], 'Native p ends before BOTH pending newborn legs')
    previous_shape = native['previous']['array']['shape']
    require(len(previous_shape) == 1 and type(previous_shape[0]) is int
            and SHARED_RANK <= previous_shape[0] <= WIDTH, 'Actual ordered previous-active count')
    rows = {}
    def add(key, value, shape, dtype):
        rows[key] = _row(value,boundary+'.'+key,size,shape,dtype)
    for key,shape,dtype in (('m',(WIDTH,),'float64'),('C',(WIDTH,WIDTH),'float64'),
            ('p',(WIDTH,inherited),'float32'),('lam',(end,),'float32'),
            ('star',(WIDTH,WIDTH),'float32'),('b',(WIDTH,),'float64'),
            ('previous_alive',(WIDTH,),'bool'),('radial',(),'float64'),
            ('previous',tuple(previous_shape),'int64')):
        add(key,native[key],shape,dtype)
    for key,shape in (('Q',(WIDTH,SHARED_RANK)),('F',(SHARED_RANK,old_columns)),
                      ('core',(SHARED_RANK,SHARED_RANK))):
        add(key,shared[key],shape,'float32')
    add('selected',nb[1],(births,),'int64'); add('base',nb[2],(WIDTH,births),'float32')
    add('t',nb[3],(births,),'float32'); add('s',nb[4],(births,),'float32')
    add('hstate',manifest['packets'][head]['state'],(WIDTH,HIDDEN),'float32')
    ranges = sorted((r['offset'],r['offset']+r['bytes']) for r in rows.values())
    require(all(b <= c for (a,b),(c,d) in zip(ranges,ranges[1:])),
            'No overlapping selected boundary descriptors or hidden aliases')
    return dict(schema=SCHEMA,rank=35,boundary=boundary,next_layer=next_layer,head_stage=head,
        candidate_sha256=CANDIDATE_SHA256,rows=rows,packet_bytes=size,
        packet_sha256=manifest['packet_sha256'],cohort_stops=stops,
        old_columns=old_columns,inherited_columns=inherited,newborn_columns=2*births,
        full_columns=end,previous_active=previous_shape[0],state_names=list(STATE_NAMES),
        differentiable_indices=list(DIFFERENTIABLE_INDICES),branch_indices=[6,12,13],
        head_pending_not_carried=True,native_strides_not_replayed_in_Torch=True,
        caller_must_bind_fresh_producer_P_and_fixture=True,truth_accessed=False)


def _tensor(torch, value, shape, dtype, name):
    require(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == dtype and tuple(value.shape) == tuple(shape)
            and bool(torch.isfinite(value).all()), 'Finite explicit CPU boundary tensor: '+name)


def expand_newborn(torch, p, selected, base, t, s, *, refs=None):
    """Pure differentiable expansion; zero selected diagonal is a strict premise."""
    require(isinstance(p,torch.Tensor) and p.ndim == 2 and p.device.type == 'cpu'
            and p.dtype in (torch.float32,torch.float64), 'Explicit real inherited bank')
    n = p.shape[0]; count = selected.numel()
    _tensor(torch,p,tuple(p.shape),p.dtype,'inherited bank')
    _tensor(torch,selected,(count,),torch.int64,'selected')
    _tensor(torch,base,(n,count),p.dtype,'newborn base')
    _tensor(torch,t,(count,),p.dtype,'stored t'); _tensor(torch,s,(count,),p.dtype,'stored s')
    require(count > 0 and count <= n and bool(((selected >= 0)&(selected < n)).all())
            and torch.unique(selected).numel() == count, 'Unique in-range selected rows in original order')
    col = torch.arange(count,dtype=torch.int64,device='cpu')
    require(bool((base[selected,col] == 0).all()), 'Original BD stored selected diagonal is exactly zero')
    x = base.index_put((selected,col),t)
    y = base.index_put((selected,col),s)
    answer = torch.cat((p,x,y),dim=1)
    _track(refs,col,x,y,answer)
    return answer


def newborn_pullback(torch, gradient, inherited_columns, selected, *, refs=None):
    """Independent coordinate pullback; base's constrained diagonal gets zero."""
    require(isinstance(gradient,torch.Tensor) and gradient.ndim == 2
            and gradient.device.type == 'cpu' and gradient.dtype in (torch.float32,torch.float64)
            and type(inherited_columns) is int and inherited_columns >= 0, 'Full-coordinate cotangent')
    n,total = gradient.shape; count = selected.numel()
    _tensor(torch,gradient,(n,total),gradient.dtype,'newborn cotangent')
    _tensor(torch,selected,(count,),torch.int64,'selected')
    require(total == inherited_columns+2*count and count > 0 and count <= n
            and bool(((selected >= 0)&(selected < n)).all())
            and torch.unique(selected).numel() == count, 'Exact three-block cotangent ownership')
    col = torch.arange(count,dtype=torch.int64,device='cpu')
    gp = gradient[:,:inherited_columns]
    gx,gy = gradient[:,inherited_columns:inherited_columns+count],gradient[:,inherited_columns+count:]
    gb = (gx+gy).index_put((selected,col),torch.zeros(count,dtype=gradient.dtype,device='cpu'))
    gt,gs = gx[selected,col],gy[selected,col]
    _track(refs,col,gp,gx,gy,gb,gt,gs)
    return gp,gb,gt,gs


def assemble(torch, values, metadata, *, refs=None):
    """Map caller-provided tensors without detaching any continuous coordinate."""
    require(metadata.get('schema') == SCHEMA and metadata.get('candidate_sha256') == CANDIDATE_SHA256
            and metadata.get('boundary') in ('entry8','end8') and metadata.get('next_layer') in (8,9)
            and set(values) == set(metadata['rows']), 'Described fixed boundary and complete tensor inventory')
    for key,row in metadata['rows'].items():
        _tensor(torch,values[key],row['shape'],getattr(torch,row['dtype']),key)
    previous,alive = values['previous'],values['previous_alive']
    selected,t,s = values['selected'],values['t'],values['s']
    require(torch.equal(torch.sort(previous).values,torch.nonzero(alive,as_tuple=True)[0]),
            'Ordered previous IDs have exactly the actual Boolean alive membership')
    require(bool((values['base'][~alive] == 0).all())
            and bool((values['p'][~alive] == 0).all()) and bool((values['Q'][~alive] == 0).all())
            and bool((t[~alive[selected]] == 0).all()) and bool((s[~alive[selected]] == 0).all()),
            'Native row masking already paid/applied; do not gate the boundary again')
    start = metadata['inherited_columns']; count = selected.numel(); lam = values['lam']
    require(torch.equal(lam[start:start+count],-lam[start+count:]),
            'Actual stored pending signed coefficients, no rederived a')
    full = expand_newborn(torch,values['p'],selected,values['base'],t,s,refs=refs)
    mask = alive.to(torch.float64); _track(refs,mask)
    state = (values['m'],values['C'],full,lam,values['star'],values['b'],mask,
             values['radial'],values['hstate'],values['Q'],values['F'],values['core'],
             tuple(metadata['cohort_stops']),previous)
    require(len(state) == 14 and full.shape == (WIDTH,metadata['full_columns']), 'Exact Torch14 state')
    return state


def decode(np, torch, manifest, blob, boundary, *, refs=None):
    """Decode only the selected boundary in the caller's owned numerical scope.

    NumPy frombuffer/reshape are representation views; Torch tensor copies,
    checks and expansion are research arithmetic. All decoded tensors are new
    owned values, not aliases to a mutable native arena or a previous graph.
    """
    metadata = describe(manifest,boundary)
    require(type(blob) is bytes and len(blob) == metadata['packet_bytes']
            and hashlib.sha256(blob).hexdigest() == metadata['packet_sha256'],
            'Exact immutable complete caller-provided packet bytes')
    values = {}
    for key,row in metadata['rows'].items():
        data = memoryview(blob)[row['offset']:row['offset']+row['bytes']]
        require(hashlib.sha256(data).hexdigest() == row['sha256'], 'Exact boundary member bytes: '+key)
        view = np.frombuffer(data,dtype=row['dtype']).reshape(row['shape'])
        answer = torch.tensor(view,dtype=getattr(torch,row['dtype']),device='cpu')
        require(answer.detach().numpy().tobytes() == data.tobytes(), 'Lossless owned tensor copy: '+key)
        _track(refs,view,answer); values[key] = answer
    state = assemble(torch,values,metadata,refs=refs)
    metadata = dict(metadata,stored_member_bytes=sum(r['bytes'] for r in metadata['rows'].values()),
        full_bank_output_bytes=int(state[2].numel()*state[2].element_size()),
        newborn_zero_diagonal_qualified=True,alive_membership_qualified=True,
        stored_signed_coefficients_qualified=True,bytes_decoded_in_caller_owned_scope=True,
        actual_full_or_local_VJP_qualified=False)
    return state,metadata
