"""Inert current-c2 L8 capture/restart source adapter. No numerical imports.

Only derive_capture/derive_restart produce source strings. Neither imports or
executes the estimator. Sink is called solely inside its caller's paid context.
"""
import ast
import hashlib
import json
from pathlib import Path
import weakref

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
DESIGN_SHA = '360add35b70970cba3af6181e7e786ceb28187c89a07813cba8a4143823a6f31'
PACKET_CAP = 512*1024**2
COPY_CAP = 480*1024**2
TEMP_CAP = 96*1024**2
DIAGNOSTIC_CAP = 2_099_703_203
N, R, MAX_COLUMNS = 1024, 384, 16384


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source():
    raw = CANDIDATE.read_bytes()
    require(digest(raw) == CANDIDATE_SHA, 'Exact frozen current-c2 source')
    require(digest((HERE/'DESIGN.md').read_bytes()) == DESIGN_SHA, 'Frozen fixed-cell design')
    return raw.decode()


def function_span(text, name):
    """UTF8 offsets indexed once; never rescan giant learned literals per node."""
    lines = text.encode().splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(nodes, 'Actual function owner: '+name)
    node = nodes[-1]  # Python's actual final definition, not an earlier namesake.
    return offsets[node.lineno-1]+node.col_offset, offsets[node.end_lineno-1]+node.end_col_offset


def replace_function(text, name, edits):
    start, stop = function_span(text, name)
    raw = text.encode(); original = raw[start:stop].decode(); result = original
    spans = []
    for old, new in edits:
        require(result.count(old) == 1, 'Unique '+name+' splice: '+old[:70])
        pos = result.index(old); result = result[:pos]+new+result[pos+len(old):]
        spans.append((pos, old, new))
    inverse = result
    for pos, old, new in reversed(spans):
        require(inverse[pos:pos+len(new)] == new, 'Exact positional inverse')
        inverse = inverse[:pos]+old+inverse[pos+len(new):]
    require(inverse == original, 'Complete function inverse')
    answer = raw[:start]+result.encode()+raw[stop:]
    require(answer[:start] == raw[:start] and answer[start+len(result.encode()):] == raw[stop:],
            'Every other parent byte remains identical')
    return answer.decode(), dict(function=name,original_sha256=digest(original.encode()),
        derived_sha256=digest(result.encode()),whole_function_inverse=True,splices=len(spans))


def derive_capture():
    original = source(); text = original; proofs = []
    edits = [
        ('    for l in range(depth):\n',
         '    for l in range(depth):\n        if l == 8: _jca_emit("entry8", locals())\n'),
        ('        pair_map = _nonlinear_first_gaussian',
         '        if l in (8, 9): _jca_emit("pre"+str(l), locals())\n'
         '        pair_map = _nonlinear_first_gaussian'),
        ('        host = m.copy()\n',
         '        if l in (8, 9): _jca_emit("pair"+str(l), locals())\n        host = m.copy()\n'),
        ('        means.append(m.copy())\n',
         '        if l in (8, 9): _jca_emit("postP"+str(l), locals())\n        means.append(m.copy())\n'),
        ('        radial = radial_new\n',
         '        radial = radial_new\n        if l == 8: _jca_emit("end8", locals())\n')]
    text, proof = replace_function(text, '_predict_binary', edits); proofs.append(proof)
    text, proof = replace_function(text, '_predict_learned_compact', [
        ('        return fnp.asarray(delta,dtype=fnp.float64)',
         '        if l in (7, 8, 9): _jca_emit("head"+str(l), locals())\n'
         '        return fnp.asarray(delta,dtype=fnp.float64)')]); proofs.append(proof)
    # The hook's current layer selects the two batch packets, not invocation count.
    text, proof = replace_function(text, '_nonlinear_pair_batch', [
        ('    rr=fnp.ones_like(rho)',
         '    _jca_emit("batch", locals())\n    rr=fnp.ones_like(rho)')]); proofs.append(proof)
    return text, dict(parent_sha256=CANDIDATE_SHA,derived_sha256=digest(text.encode()),
        source_inverse=True,functions=proofs,original_physical_arithmetic_unchanged=True,
        capture_only=True,candidate_materialized=False)


ENTRY_KEYS = ('m','C','radial','b','p','lam','star','previous','previous_alive','newborn','shared_old')
PRE_KEYS = ('mu','Cpre','Bpre','bpre','vv','K21','k4','k22','radial','cp_pre','star_pre',
            'diagonal','live_indices','alive','response_indices','response_mask','omitted')
PAIR_KEYS = ('m','C','target','radial_new','aux')
HEAD_KEYS = ('state','delta','_bdnh_pending','_bdnh_layer')
BATCH_KEYS = ('F','G','rho','T21','T22','T31','t3','t4')
POST_KEYS = ('m','C','host','dm','radial_new')
END_KEYS = ('m','C','radial','b','lam','star','previous','previous_alive','newborn','shared_old',
            'p','Bpost','trace_post','cp_post','born','inherited','rest','a','t','s','birth','beta','u','delta')
STAGES = ('head7','entry8','pre8','batch8','pair8','head8','postP8','end8',
          'pre9','batch9','pair9','head9','postP9')


class Sink:
    """Borrowed arrays never escape. Each copy/finite/export runs in actual ctx.

    end8's large redundant p/F/Q/core are fingerprint-only: a paid copied byte
    packet is hashed immediately, not retained. Entry8 keeps their restart bytes.
    Aux pre_cov aliases the already saved pre{8,9}.Cpre by explicit manifest link.
    """
    def __init__(self, fnp):
        self.fnp = fnp; self.ctx = None; self.packets = {}; self.blocks = []
        self.spans = []; self.bytes = 0; self.copied_bytes = 0; self.layer = None
        self.flops = 0; self.operations = 0; self.refs = []

    def array(self, value, path, keep=True):
        shape = tuple(int(x) for x in value.shape); dtype = str(value.dtype)
        require(dtype in ('float32','float64','int32','int64','bool'), 'Explicit physical/counter dtype')
        size = int(value.nbytes)
        require(size <= 64*1024**2 and self.copied_bytes+size <= COPY_CAP,
                'Single copied bank<=64MiB, aggregate paid copies<=480MiB')
        require(not keep or self.bytes+size <= PACKET_CAP, 'Reject packet overflow BEFORE copy')
        # Public function also meters an immediate NumPy scalar; a borrowed
        # scalar's own .copy() method would not establish that billing owner.
        copy = self.fnp.copy(value)
        try:
            ref = weakref.ref(copy); self.refs.append(ref)
        except TypeError:
            require(shape == () and size <= 8, 'Only an immediate scalar may lack weakref')
            ref = None
        # copy + full finite bitmap + scalar reduction fit <=80MiB for largest bank.
        require(size+int(copy.size) <= TEMP_CAP, 'Explicit callback temporaries<=96MiB')
        finite = self.fnp.all(self.fnp.isfinite(copy))
        require(bool(finite), 'Finite raw captured state: '+path)
        raw = copy.tobytes(); require(len(raw) == size, 'Full typed detached bytes')
        row = dict(shape=list(shape),dtype=dtype,bytes=size,sha256=digest(raw))
        self.copied_bytes += size
        if keep:
            row['offset'] = self.bytes; self.blocks.append(raw); self.bytes += size
        else:
            row['fingerprint_only'] = True
        del finite, copy, raw
        require(ref is None or ref() is None, 'Every explicit callback copy released immediately')
        return row

    def encode(self, value, path, keep=True):
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            return dict(array=self.array(value,path,keep))
        if value is None or isinstance(value,(str,int,float,bool)):
            # JSON finite scalar enforcement occurs in the paid callback too.
            json.dumps(value, allow_nan=False); return dict(scalar=value)
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if path.endswith('.aux') and key == 'pre_cov':
                    result[key] = dict(alias='pre'+str(self.layer)+'.Cpre')
                else:
                    result[key] = self.encode(item,path+'.'+str(key),keep and not
                        (path == 'end8.shared_old' and key in ('F','Q','core')))
            return dict(mapping=result)
        if isinstance(value,(list,tuple)):
            return dict(sequence=[self.encode(x,path+'.'+str(i),keep) for i,x in enumerate(value)],
                        tuple=isinstance(value,tuple))
        raise RuntimeError('No opaque/borrowed packet owner: '+path)

    def __call__(self, stage, local):
        if stage == 'batch':
            if self.layer not in (8,9) or 'pre'+str(self.layer) not in self.packets:
                return
            stage = 'batch'+str(self.layer)
            if stage in self.packets:
                return
        if stage.startswith('pre'):
            self.layer = int(stage[-1])
        require(stage == STAGES[len(self.packets)] and self.ctx is not None,
                'Fixed13 target-blind packet boundaries inside actual paid context')
        keys = (ENTRY_KEYS if stage=='entry8' else END_KEYS if stage=='end8' else
                PRE_KEYS if stage.startswith('pre') else PAIR_KEYS if stage.startswith('pair') else
                HEAD_KEYS if stage.startswith('head') else BATCH_KEYS if stage.startswith('batch') else POST_KEYS)
        require(all(key in local for key in keys), 'Actual current-c2 locals, no guessed fallback')
        if stage == 'entry8':
            require(local['newborn'] is not None and local['shared_old']['pre_basis'] is None
                    and int(local['p'].shape[1]) <= MAX_COLUMNS and int(local['lam'].shape[0]) <= MAX_COLUMNS,
                    'Complete pre-join8 entry, bounded bank, no pending transported basis')
        start, before = len(self.ctx.op_log), int(self.ctx.flops_used)
        self.packets[stage] = {key:self.encode(local[key],stage+'.'+key,
            keep=not(stage=='end8' and key=='p')) for key in keys}
        if stage.startswith('pre'):
            # Explicit physical current Q31, not the subsequently rounded T31.
            Q31 = (local['k4']/local['vv'])[:,None]*local['Cpre']
            self.packets[stage]['Q31_physical'] = self.encode(Q31,stage+'.Q31_physical')
            del Q31
        stop, after = len(self.ctx.op_log), int(self.ctx.flops_used)
        self.spans.append(dict(stage=stage,start=start,stop=stop,flops=after-before))
        self.flops += after-before; self.operations += stop-start
        require(0 < self.flops <= DIAGNOSTIC_CAP and all(ref() is None for ref in self.refs),
                'Bounded complete diagnostic work; no explicit captured FNP owner retained')

    def finish(self):
        require(tuple(self.packets) == STAGES and self.ctx is None and
                all(ref() is None for ref in self.refs), 'All13 detached packets and retired callback')
        return dict(schema='current_comp_joint_cell_packet_v1',rank=35,layer=8,consumer=9,
            candidate_sha256=CANDIDATE_SHA,packets=self.packets,bytes=self.bytes,
            paid_copy_bytes=self.copied_bytes,diagnostic_flops=self.flops,
            diagnostic_operation_count=self.operations,diagnostic_spans=self.spans,
            all_explicit_arrays_released=True,truth_accessed=False,
            repeated_end_bank_is_fingerprint_only=True)


def derive_restart():
    """Source-only exact two-layer consumer, requiring supplied paid entry/H7.

    _jca_intervene returns (mu,Cpre,K21,k4,k22,Q31_or_None) at L8 only.
    Null returns original objects +None. No intervention at L9. All original
    P/NODE/BD/star/trace/shared-join consumers execute in their existing order.
    This is not runnable admission, a truth reader, or a qualified null claim.
    """
    text, capture = derive_capture(); proofs = []
    original = text.encode(); a,b = function_span(text,'_predict_binary')
    body = original[a:b].decode()
    start = body.index('    previous = fnp.arange(n)')
    stop = body.index('    for l in range(depth):')
    initialization = body[start:stop]
    restore = ''.join('    '+key+' = _jca_entry['+repr(key)+']\n' for key in ENTRY_KEYS)
    restore += '    means, states = ([], [])\n'
    edits = [('def _predict_binary(', 'def _jca_restart_binary('),
        (initialization,restore),('    for l in range(depth):','    for l in range(8, 10):'),
        ('        if l == depth - 1:\n            break', '        if l == 9:\n            break'),
        ('        m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)',
         '        if l == 8:\n'
         '            mu, Cpre, K21, k4, k22, explicit_Q31 = _jca_intervene(mu, Cpre, K21, k4, k22)\n'
         '        else:\n            explicit_Q31 = None\n'
         '        if explicit_Q31 is None:\n'
         '            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n'
         '        else:\n'
         '            m, C, target, radial_new, aux = _jca_explicit_pair(mu, Cpre, K21, k4, k22, metric, explicit_Q31)')]
    text,proof=replace_function(text,'_predict_binary',edits); proofs.append(proof)
    text,proof=replace_function(text,'_predict_learned_compact',[
        ('    state=fnp.zeros((int(mlp.width),int(P["head"].shape[1])),dtype=fnp.float32)',
         '    state = _jca_H7'),
        ('return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook)',
         'return _jca_restart_binary(mlp,hook=hook,coefficient_hook=coefficient_hook)')]); proofs.append(proof)
    # Independent function duplicate: original nonlinear owner remains byte-exact.
    a,b=function_span(source(),'_nonlinear_pair_batch'); pair=source().encode()[a:b].decode()
    explicit,proof=replace_function(pair,'_nonlinear_pair_batch',[
        ("def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):",
         "def _jca_explicit_pair(mu, C, K21, k4, k22_override=None, metric='identity', explicit_Q31=None):"),
        ('    T31 = t4[:, None] * rho if k31 else None',
         '    T31 = explicit_Q31 / (s[:,None]**3 * s[None,:])\n    fnp.fill_diagonal(T31, 0)')]); proofs.append(proof)
    text += '\n\n'+explicit+'\n'
    return text,dict(parent_sha256=CANDIDATE_SHA,capture=capture,restart_inverses=proofs,
        derived_sha256=digest(text.encode()),fixed_layers=[8,9],full_P_NODE_BD=True,
        Q31_physical_argument=True,null_uses_original_T31=True,qualified=False)
