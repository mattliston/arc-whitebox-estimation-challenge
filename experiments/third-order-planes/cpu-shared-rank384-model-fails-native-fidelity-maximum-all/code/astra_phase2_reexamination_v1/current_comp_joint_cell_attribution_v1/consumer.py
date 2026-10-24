"""DRAFT: fixed current-c2 L8 -> post-P9 restart; inert without --run.

No replica B reader or information decision lives here. Each arm owns fresh
physical state and parameter caches. Only N is a null-identity qualification.
"""
import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import sys
import time
import traceback
import types
import weakref
import zipfile

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SHARED = NEW/'current_comp_shared_old384_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
OUTPUT = HERE/'consumer_v1'
SCHEMA = 'current_comp_joint_cell_consumer_v1'
ROSTER = ('tiny', 'N', 'T', 'Q', 'TQ', 'ALL-local')
CAP, RSS, BUDGET = 600., 8_000_000_000, 2**41
PARTIAL_CAP, LOCAL_CAP = 250_000_000_000, 5_000_000_000
KEYS = ('source','watcher','protocol','watch-spec','tiny','tiny-protocol')
DESIGN_SHA = '360add35b70970cba3af6181e7e786ceb28187c89a07813cba8a4143823a6f31'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
PINS = {
    HERE/'DESIGN.md': DESIGN_SHA,
    HERE/'adapter_v2.py': '6f30900ea7a0e3babf2b339c327475d5d64d16fa6ebfdc2ff74c44167e8b3510',
    HERE/'capture_v2.py': '975f0a311c261e3bbe85f409b80872a4fe7ade8762dc08e38c3038508b36f8fc',
    HERE/'watch_capture_v2.py': '111c67bdb425c53902046229ee2373f20ca02ab4ce03e48729c1ec935b99fdd4',
    HERE/'CAPTURE_PROTOCOL_V2.md': 'a84dc711246cbc218cc6712f9cc7e91e8e6a226a145b8f9a75ebc429376f111c',
    HERE/'CAPTURE_WATCH_SPEC_V2.md': '1d0cc069a4018c2a8c7386fb6c9325fb0546b8b52b73ffaf658784c7ec5edbd6',
    SHARED/'probe.py': '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    SHARED/'backend_probe.py': '36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8',
    SHARED/'check4_v2.py': '8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
}
CAPTURE_CHAINS = {
    'capture_v1': (
        '7867fb6b645d042d88fb359d7f831708d2e7feb37e26d845309805aec4796804',
        '0bdb594c411d07ffe9c1fd5c21f394c51d8d558d147aa18752e4067b75021c54',
        '5e5589cbb3fa91d8ef8c3d6dae6587e68c4f174283c0e0e3ec049a44f15cbb78'),
    'capture_watch_v1': (
        '790b0d8ad7e88df9aa258d16de35ec6f5638c3f202f3a0cae51f46812abb9a34',
        '4fd68816a97ae3fb3ee905923fc55eb2fb258eaa531e6359ae927c893551f04e',
        'aa5e395b811694fdbbedee8141cf62238aef6e017fb1c077b4a4cb12d6c703ed'),
}
SUPPLY_FILES = {
    P2/'public_g19_attribution_v1/mps_cov2_rank0035.npz':
        '7e6b97166885de0cdd870a835ae536830fb6da41ca345d8a66e2eb8d0d6380e6',
    P2/'public_g19_attribution_v1/mps_planes_rank0035.npz':
        '64bbf8f8b17115252a186535c45a15295a0869135a547faab04242fb72453398',
}
PACKET_SHA = '1f29ab16f3454a00c180170c7993cdc048c1cb36b15f2c72f8bfb2ff6a259225'
STAGES = ('entry8','pre8','batch8','pair8','head8','postP8','end8',
          'pre9','batch9','pair9','head9','postP9')
SAVED = frozenset(('pair8.m','pair8.C','postP8.m','postP8.C','postP8.host','postP8.dm',
                  'end8.C','pair9.m','postP9.m','postP9.host','postP9.dm'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path, tick=None, *, supply=False):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and not path.is_symlink(), 'Local regular immutable owner')
    require(path not in SUPPLY_FILES or supply, 'A archive only after sealed N')
    require('covariance_replica_truth_v1' not in path.parts and 'truthpost' not in path.parts
            and not (path.suffix in ('.npz','.npy') and 'truth' in str(path).lower()),
            'Replica B/final labels excluded from this worker')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded metadata')
    return json.loads(path.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return dict(nonfinite_float=repr(x))
        if isinstance(x, dict):
            return {str(k):safe(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def bind(items, tick=None, *, supply=False):
    require(isinstance(items, dict), 'Flat direct binding map')
    for p,d in items.items():
        require(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p,tick,supply=supply) == d,
                'Fresh immutable owner: '+str(p))


def load(path, name, expected):
    require(sha(path) == expected, 'Exact source before inert import')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def splice(text, edits):
    original = text; spans = []
    for old,new in edits:
        require(text.count(old) == 1, 'Unique reversible source splice: '+old[:100])
        p = text.index(old); text = text[:p]+new+text[p+len(old):]
        spans.append((p,old,new))
    inverse = text
    for p,old,new in reversed(spans):
        require(inverse[p:p+len(new)] == new, 'Recorded positional source inverse')
        inverse = inverse[:p]+old+inverse[p+len(new):]
    require(inverse == original, 'Complete original source restored by inverse')
    return text, dict(original_sha256=digest(original.encode()),derived_sha256=digest(text.encode()),
                      whole_source_inverse=True,splices=len(spans))


def derive(a, *, local=False, tick=None):
    """Keep actual _predict_binary identity. No original function is renamed."""
    text, capture_proof = a.derive_capture(); proofs = []
    if tick is not None: tick()
    lo,hi = a.function_span(text,'_predict_binary'); body = text.encode()[lo:hi].decode()
    p = body.index('    previous = fnp.arange(n)')
    q = body.index('    for l in range(depth):')
    restore = ''.join('    '+k+' = _jca_entry['+repr(k)+']\n' for k in a.ENTRY_KEYS)
    restore += '    means, states = ([], [])\n'
    text,proof = a.replace_function(text,'_predict_binary',[
        (body[p:q],restore),('    for l in range(depth):','    for l in range(8, 10):'),
        ('        if l == depth - 1:\n            break','        if l == 9:\n            break'),
        ('        m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)',
         '        if l == 8:\n'
         '            mu, Cpre, K21, k4, k22, explicit_Q31 = _jca_intervene(mu, Cpre, K21, k4, k22)\n'
         '        else:\n            explicit_Q31 = None\n'
         '        if explicit_Q31 is None:\n'
         '            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n'
         '        else:\n'
         '            m, C, target, radial_new, aux = _jca_explicit_pair(mu, Cpre, K21, k4, k22, metric, explicit_Q31)')
    ]); proofs.append(proof)
    if tick is not None: tick()
    text,proof = a.replace_function(text,'_predict_learned_compact',[
        ('    state=fnp.zeros((int(mlp.width),int(P["head"].shape[1])),dtype=fnp.float32)',
         '    state = _jca_H7')]); proofs.append(proof)
    if tick is not None: tick()
    # Duplicate the CAPTURED current pair, so explicit Q31 emits the actual
    # post-FP32-cast batch too. Original implicit pair bytes remain unchanged.
    lo,hi = a.function_span(text,'_nonlinear_pair_batch'); pair = text.encode()[lo:hi].decode()
    explicit,proof = a.replace_function(pair,'_nonlinear_pair_batch',[
        ("def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):",
         "def _jca_explicit_pair(mu, C, K21, k4, k22_override=None, metric='identity', explicit_Q31=None):"),
        ('    T31 = t4[:, None] * rho if k31 else None',
         '    T31 = explicit_Q31 / (s[:,None]**3 * s[None,:])\n    fnp.fill_diagonal(T31, 0)')
    ]); proofs.append(proof)
    if tick is not None: tick()
    text += '\n\n'+explicit+'\n'
    if local:
        lo,hi = a.function_span(text,'_predict_binary'); old = text.encode()[lo:hi].decode()
        # Exactly the original hook/recentering span, without a fictitious old
        # CP/metric/omitted-radial input. ALL-local never pushes or commits state.
        hook_start = body.index('        if hook is not None:')
        hook_stop = body.index('        means.append(m.copy())')
        hook = body[hook_start:hook_stop]
        hook = '\n'.join(line[4:] if line.startswith('    ') else line for line in hook.split('\n'))
        replacement = old[:old.index('\n')]+'''\n    _bd_c = _bd_coefficients()
    n, W, l = int(mlp.width), mlp.weights, 8
    mu, Cpre, K21, k4, k22, explicit_Q31 = _jca_all_inputs
    _jca_emit("allpre8", locals())
    m, C, target, radial_new, aux = _jca_explicit_pair(mu, Cpre, K21, k4, k22, metric, explicit_Q31)
    _jca_emit("pair8", locals())
    host = m.copy()
'''+hook+'''    return fnp.asarray(fnp.maximum(fnp.stack([m]), 0.0), dtype=fnp.float32)'''
        text,proof = a.replace_function(text,'_predict_binary',[(old,replacement)])
        proofs.append(proof)
    return text, dict(parent_sha256=CANDIDATE_SHA,capture=capture_proof,functions=proofs,
        derived_sha256=digest(text.encode()),actual_predict_binary_identity_preserved=True,
        full_P_NODE_BD=not local,only_H7_restored=True,null_uses_original_objects_and_T31=True,
        explicit_Q31_batch_captured=True,ALL_local_pair_and_P_only=local)


def packet_arrays(value, prefix=''):
    if isinstance(value, dict) and 'array' in value:
        yield prefix,value['array']
    elif isinstance(value, dict):
        for key,item in value.items():
            yield from packet_arrays(item,prefix+'.'+str(key) if prefix else str(key))
    elif isinstance(value, list):
        for i,item in enumerate(value):
            yield from packet_arrays(item,prefix+'.'+str(i))


def validate_packet(manifest):
    require(manifest['schema']=='current_comp_joint_cell_packet_v1' and
        (manifest['rank'],manifest['layer'],manifest['consumer']) == (35,8,9) and
        manifest['candidate_sha256']==CANDIDATE_SHA and manifest['bytes']==351195456 and
        manifest['packet_sha256']==PACKET_SHA and manifest['saved_after_context_release'] and
        manifest['all_explicit_arrays_released'] and manifest['truth_accessed'] is False,
        'The actual positive target-blind current entry, not a historical oracle reset')
    require(set(manifest['packets'])==set(('head7',)+STAGES), 'All13 original packet stages')
    ranges=[]
    for path,row in packet_arrays(manifest['packets']):
        itemsize={'float32':4,'float64':8,'int32':4,'int64':8,'bool':1}[row['dtype']]
        require(all(type(v) is int and v >= 0 for v in row['shape']) and
            math.prod(row['shape'])*itemsize==row['bytes'] and row['byteorder']==sys.byteorder and
            re.fullmatch('[0-9a-f]{64}',row['sha256']), 'Complete typed array metadata: '+path)
        if 'offset' in row:
            require(type(row['offset']) is int and row['offset']>=0, 'Nonnegative packet offset')
            ranges.append((row['offset'],row['offset']+row['bytes']))
        else:
            require(row.get('fingerprint_only') is True and path.startswith('end8.'),
                    'Only redundant end8 banks may be fingerprint-only')
    cursor=0
    for lo,hi in sorted(ranges):
        require(lo==cursor and hi>=lo, 'Disjoint gap-free saved packet ranges')
        cursor=hi
    require(cursor==manifest['bytes']<512*1024**2,'Exact packet cap')
    entry=manifest['packets']['entry8']
    for path,row in packet_arrays(entry):
        require(row['original_type_module']=='flopscope._ndarray' and
                row['original_type_name']=='FlopscopeArray' and
                (row['original_c_contiguous'] or row['original_f_contiguous']),
                'No invented noncompact/symmetric entry reconstruction: '+path)
    require(entry['p']['array']['shape']==[1024,13376] and
            entry['lam']['array']['shape']==[14912] and
            entry['shared_old']['mapping']['F']['array']['shape']==[384,6144] and
            entry['shared_old']['mapping']['pre_basis']=={'scalar':None}, 'Exact pre-join8 geometry')
    return dict(packet_bytes=cursor,entry_plain_compact=True,H7_only=True,
        F_entry_path='entry8.newborn.sequence.2',
        noncompact_checkpoints_not_loaded=['pre8.vv','pre9.vv'],
        symmetric_checkpoints_not_loaded=['pre8.cp_pre','pre8.star_pre','pre9.cp_pre','pre9.star_pre'])


def physical_metadata(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=int(value.nbytes),
        original_strides=list(value.strides),original_c_contiguous=bool(value.flags.c_contiguous),
        original_f_contiguous=bool(value.flags.f_contiguous),original_type_module=type(value).__module__,
        original_type_name=type(value).__name__,byteorder=sys.byteorder)


def restore(value, raw, np, fnp, refs, path):
    """Called inside owned context. np.frombuffer/reshape are byte views only.

    Only entry8 plus H7 can call this; checkpoints never become replay inputs.
    FNP allocation/copy and complete finite tests are included in this bill.
    """
    require(path.startswith('entry8') or path=='head7.state','No pre8/head8/head9/null leakage')
    if 'array' in value:
        row=value['array']; lo=row['offset']; data=raw[lo:lo+row['bytes']]
        require(digest(data)==row['sha256'],'Exact physical restart bytes: '+path)
        view=np.frombuffer(data,dtype=row['dtype']).reshape(row['shape'])
        order='F' if row['original_f_contiguous'] and not row['original_c_contiguous'] else 'C'
        answer=fnp.array(view,dtype=getattr(fnp,row['dtype']) if row['dtype']!='bool' else fnp.bool_,
                         order=order,copy=True)
        require(bool(fnp.all(fnp.isfinite(answer))) and answer.tobytes()==data,
                'Paid complete finite/copy byte check')
        require(all(physical_metadata(answer)[k]==row[k] for k in physical_metadata(answer)),
                'Exact original entry wrapper, strides, dtype and scalar shape: '+path)
        refs.append(weakref.ref(answer))
        return answer
    if 'scalar' in value:
        return value['scalar']
    if 'mapping' in value:
        return {k:restore(v,raw,np,fnp,refs,path+'.'+k) for k,v in value['mapping'].items()}
    if 'sequence' in value:
        seq=[restore(v,raw,np,fnp,refs,path+'.'+str(i)) for i,v in enumerate(value['sequence'])]
        return tuple(seq) if value['tuple'] else seq
    raise RuntimeError('No entry alias or opaque state: '+path)


def replay_sink_class(a):
    class ReplaySink(a.Sink):
        def __init__(self,fnp,manifest,arm):
            super().__init__(fnp)
            self.reference=manifest['packets']; self.arm=arm; self.saved={}; self.checks={}
            self.roster=('allpre8','batch8','pair8','head8','postP8') if arm=='ALL-local' else STAGES

        def array(self,value,path,keep=True):
            # No borrowed array survives the callback. Persist only the named
            # analysis planes/vectors; all full-carrier checkpoints are hashes.
            row=super().array(value,path,False)
            if self.arm=='N':
                expected=self.reference
                for token in path.split('.'):
                    if 'mapping' in expected: expected=expected['mapping']
                    if 'sequence' in expected: expected=expected['sequence']
                    expected=expected[int(token)] if isinstance(expected,list) else expected[token]
                expected=expected['array']
                require(all(row[k]==expected[k] for k in (*physical_metadata(value),'sha256')),
                        'N byte/layout/class checkpoint identity: '+path)
                self.checks[path]=True
            if path in SAVED:
                # Public second copy/export is deliberate and metered too.
                copy=self.fnp.copy(value); ref=weakref.ref(copy); self.refs.append(ref)
                require(bool(self.fnp.all(self.fnp.isfinite(copy))), 'Saved readout finite')
                data=copy.tobytes(); del copy
                require(ref() is None and digest(data)==row['sha256'],'Detached saved output')
                self.copied_bytes+=len(data)
                require(self.copied_bytes<=a.COPY_CAP,'All replay/analysis copies fit original copy envelope')
                self.saved[path]=(row,data)
            return row

        def encode(self,value,path,keep=True):
            if self.arm=='N' and (value is None or isinstance(value,(str,int,float,bool))):
                expected=self.reference
                for token in path.split('.'):
                    if 'mapping' in expected: expected=expected['mapping']
                    if 'sequence' in expected: expected=expected['sequence']
                    expected=expected[int(token)] if isinstance(expected,list) else expected[token]
                require(expected=={'scalar':value},'N exact structural/scalar state: '+path)
            return super().encode(value,path,False)

        def __call__(self,stage,local):
            if stage=='batch':
                if self.layer not in (8,9): return
                stage='batch'+str(self.layer)
            if stage.startswith('pre') or stage=='allpre8': self.layer=int(stage[-1])
            require(self.ctx is not None and stage==self.roster[len(self.packets)],
                    'Exact restart-only packet roster; no stale head7/pending state')
            keys=(a.ENTRY_KEYS if stage=='entry8' else a.END_KEYS if stage=='end8' else
                  a.PRE_KEYS if stage.startswith('pre') else a.PAIR_KEYS if stage.startswith('pair') else
                  a.HEAD_KEYS if stage.startswith('head') else a.BATCH_KEYS if stage.startswith('batch') else
                  ('mu','Cpre','K21','k4','k22','explicit_Q31') if stage=='allpre8' else a.POST_KEYS)
            require(all(k in local for k in keys),'Actual current consumer local owners')
            start,before=len(self.ctx.op_log),int(self.ctx.flops_used)
            self.packets[stage]={k:self.encode(local[k],stage+'.'+k,False) for k in keys}
            if stage.startswith('pre'):
                require(bool(self.fnp.all(local['vv']>1e-10)),'Unfloored actual pre8/pre9 variance')
                q=(local['k4']/local['vv'])[:,None]*local['Cpre']
                self.fnp.fill_diagonal(q,local['k4'])
                self.packets[stage]['Q31_physical']=self.encode(q,stage+'.Q31_physical',False)
                del q
            if stage=='allpre8':
                require(bool(self.fnp.all(self.fnp.diag(local['Cpre'])>1e-10)), 'Unfloored supplied ALL variance')
            after=int(self.ctx.flops_used)
            self.spans.append(dict(stage=stage,start=start,stop=len(self.ctx.op_log),flops=after-before))
            self.flops+=after-before
            require(all(ref() is None for ref in self.refs), 'Every diagnostic FNP copy retired')

        def finish(self):
            require(self.ctx is None and tuple(self.packets)==self.roster and
                    all(ref() is None for ref in self.refs),'Complete detached restart packet')
            return dict(arm=self.arm,stages=list(self.roster),checkpoints=self.packets,
                byte_layout_checks=len(self.checks),null_byte_exact=self.arm=='N' and all(self.checks.values()),
                diagnostic_flops=self.flops,diagnostic_spans=self.spans,paid_copy_bytes=self.copied_bytes,
                all_explicit_arrays_released=True,head7_pending_never_loaded=True)
    return ReplaySink


def cost_certificate(manifest, a):
    """Conservative source tariff envelope, not measured counterfactual F.

    Every gate-dependent row count is bounded by 1024, with the exact two
    already-sealed joining cohorts. Rectangle costs include their copies;
    96*(mk+kn+mn) dominates the maximum D6 coefficients (64.687/35.964)
    and the extra output allocation. Output/contracted tiling is priced here.
    The nonmatrix allowance is explicitly partitioned below. Per-cell counts
    include dtype rate2, three stacked pair channels, broadcast operands,
    scans, copies and reductions. These are upper counts for the fixed source
    loops, not a fitted fraction of an old bill. No matrix product is hidden
    in them. The n*R portion counts full-bank diagonal, omitted-radial and
    gating/read materializations; all its matrix-vector contractions are
    included at FP64's 2F rate.
    """
    n,r=1024,384
    state=manifest['packets']['entry8']['shared_old']['mapping']
    cohorts=[tuple(v['scalar'] for v in item['sequence']) for item in state['cohorts']['sequence']]
    require(cohorts==[(0,2048),(2048,2048),(4096,2048),(6144,1984),
                     (8128,1856),(9984,1728),(11712,1664),(13376,1536)], 'Exact immutable entry cohorts')
    def mm(m,k,q):
        if not m or not k or not q: return m*q
        if m>1024:
            return sum(mm(min(1024,m-j),k,q) for j in range(0,m,1024))+m*q
        if q>8192:
            return sum(mm(m,k,min(8192,q-j)) for j in range(0,q,8192))+m*q
        if k>8192:
            return sum(mm(m,min(8192,k-j),q) for j in range(0,k,8192))+((k-1)//8192)*m*q
        return 2*m*k*q+96*(m*k+k*q+m*q)
    layers=[]
    for layer,old_before,join_width,inherited,birth_width in (
            (8,6144,1984,13376,768),(9,8128,1856,14912,1024)):
        old_after=old_before+join_width
        columns=inherited+2*birth_width
        yi=inherited-old_after; young=yi+2*birth_width
        shapes=[('join_Jt_Omega',(join_width,n,r)),('join_J_product',(n,join_width,r)),
            ('old_Qt_Omega',(r,n,r)),('old_core_product',(r,r,r)),('old_Q_product',(n,r,r)),
            ('Fj',(r,n,join_width)),('joined_core',(r,join_width,r)),('overlap',(r,n,r)),
            ('rotate_F',(r,r,old_before)),('rotate_core_left',(r,r,r)),('rotate_core_right',(r,r,r)),
            ('affine_basis',(n,n,r)),('old_reconstruction',(n,r,old_after)),
            ('young_inherited',(n,n,yi)),('young_newborn_base',(n,n,birth_width)),
            ('old_read_inner',(n,old_after,r)),('old_read_outer',(n,r,n)),('young_read',(n,young,n))]
        # Two active-cov products plus all three cold-metric half products,
        # and four star products: each independently bounded by a full n cube.
        shapes += [('covariance_'+str(i),(n,n,n)) for i in range(5)]
        shapes += [('star_'+str(i),(n,n,n)) for i in range(4)]
        # All six current GNN messages, three learned affine encoders, NODE,
        # and alpha/rho contraction are already below this explicit bound.
        shapes += [('GNN_message_'+str(i),(n,n,16)) for i in range(6)]
        shapes += [('GNN_dense_'+str(i),(n,256,256)) for i in range(4)]
        shapes += [('NODE',(n,64,4))]
        products=[dict(owner=name,shape=list(shape),upper_flops=mm(*shape)) for name,shape in shapes]
        qr=8*n*r*r
        nonmatrix_counts={
            # Seven Gaussian and three Edgeworth iterations, three channels;
            # <=512 scalar array operations/cell even if all were FP64.
            'pair6_edge2_and_centralization':2048*n*n,
            'cov_star_gather_scatter_cast_epilogues':1024*n*n,
            # Four directions, projection/orthogonalization, actual32 repair.
            'BD_four_directions_and_stored_plane':1024*n*n,
            'mask_trace_birth_recenter_star_assembly':1024*n*n,
            # Thirty slots + six blocks + four dense epilogues + NODE64x4.
            'features_GNN_NODE_nonmatrix':1024*n*n,
            # Raw Gaussian vectors, selector sorting/indexing, live masks,
            # copy/export checks and fixed supply provenance reductions.
            'vectors_selectors_scans_and_guards':2048*n*n,
            # FP64 cast/cube/weighted reduction/omitted radial <=32*n*R;
            # FP32 gathers, young assembly, gates and finite scans <=32*n*R.
            'full_bank_pointwise_and_FP64_contractions':64*n*columns}
        pointwise=sum(nonmatrix_counts.values())
        total=sum(x['upper_flops'] for x in products)+qr+pointwise
        layers.append(dict(layer=layer,bank_columns_upper=columns,old_columns=old_after,
            previous_live_upper=n,response_rows_upper=n,live_rows_upper=n,births_upper=birth_width,
            products=products,qr_upper_flops=qr,nonmatrix_counts=nonmatrix_counts,
            all_other_physics_upper_flops=pointwise,upper_flops=total))
    # Independent from the parent bill: fresh original packed and square
    # bindings/literals, all entry/weight conversions, diagnostics and exports.
    # Actual all possible depth3/4/5 bindings and P/A/cstar prefix from the
    # complete source are 1,944,970,298F. Even charging the 292,225,024-element
    # empty arena once in addition and a million literal/loader scalar F is
    # <2.24BF. 3BF therefore dominates every fresh partial setup, without
    # assuming the same warm cache or subtracting a saved parent's startup.
    setup=3_000_000_000
    copies=3*a.COPY_CAP+3*manifest['bytes']+64*n*n
    upper=sum(x['upper_flops'] for x in layers)+setup+copies
    require(upper<=PARTIAL_CAP,'Complete source envelope must establish <=250BF BEFORE any context/A')
    return dict(schema='current_comp_joint_cell_cost_certificate_v1',layers=layers,
        initialization_upper_flops=setup,entry_input_diagnostic_upper_flops=copies,
        complete_partial_upper_flops=upper,limit_flops=PARTIAL_CAP,
        worst_L9_masks_included=True,actual_bills_still_required=True,
        ALL_local_upper_flops=LOCAL_CAP,not_a_complete_estimator_score=True,
        parent_source_sha256=CANDIDATE_SHA)


def admission(args,tick=None):
    began=time.monotonic(); outer=tick
    def tick():
        require(time.monotonic()-began<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1_000_000_000,
                'Complete stdlib metadata admission60s/1GB')
        if outer is not None: outer()
    names=('consumer.py','watch_consumer.py','CONSUMER_PROTOCOL.md','CONSUMER_WATCH_SPEC.md',
           'tiny.py','TINY_PROTOCOL.md')
    sources={str(HERE/name):getattr(args,key.replace('-','_')+'_sha256') for name,key in zip(names,KEYS)}
    sources.update({str(p):v for p,v in PINS.items()}); bind(sources,tick)
    inputs={}; records={}
    for folder,expected in CAPTURE_CHAINS.items():
        paths=[HERE/folder/name for name in ('result.json','completion.json','finalization.json')]
        actual=tuple(sha(p,tick) for p in paths)
        require(actual==expected,'Actual frozen positive capture three seals')
        r,e,f=map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f)) and
            e['result_sha256']==f['result_sha256']==actual[0] and f['completion_sha256']==actual[1],
            'Completed linked capture seals; no partial eligibility')
        inputs.update({str(p):v for p,v in zip(paths,actual)}); records[folder]=r
    c,w=records['capture_v1'],records['capture_watch_v1']
    require(c['schema']=='current_comp_joint_cell_capture_v1' and
        w['schema']=='current_comp_joint_cell_capture_watch_v1' and
        c['candidate_sha256']==w['candidate_sha256']==CANDIDATE_SHA and
        c['full_model_calls']==c['retired_call_count']==1 and c['truth_accessed'] is False and
        all(c.get(k) is True for k in ('baseline_output_byte_exact','all16_domains_qualified',
            'all47_selectors_qualified','all_original_operations_exact','own_complete_bills',
            'all_capture_arrays_released','all_parameter_and_estimator_owners_retired','hashes_unchanged')),
        'Actual unchanged target-blind current-c2 capture, all16/all47 and whole typed ledger')
    require([w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]==
        list(CAPTURE_CHAINS['capture_v1']) and w['whole_owned_tree_retired'] and w['cleanup_passed'] and
        w['maximum_combined_rss_bytes']<RSS and w['maximum_interval_s']<=1 and
        w['maximum_swap_growth_bytes']==0 and w['elapsed_s']<180, 'Actual capture watch joins/resources/reap')
    for record in (c,w):
        for group in ('source_bindings','input_bindings','artifacts'):
            bind(record[group],tick)
            (sources if group=='source_bindings' else inputs).update(record[group])
    require(all(c['source_bindings'].get(str(p))==v for p,v in PINS.items()
                if p.name not in ('probe.py','backend_probe.py','check4_v2.py')),
            'The completed capture used these exact versioned physical owners')
    path=HERE/'capture_v1/restart_packet.json'; manifest=read(path)
    packet_proof=validate_packet(manifest)
    require(manifest['packet_file']==str(HERE/'capture_v1/restart_packet.bin') and
        inputs[manifest['packet_file']]==PACKET_SHA and c['packet_bytes']==manifest['bytes'] and
        c['trials'][0]['flops']==403400296797+manifest['diagnostic_flops']==403695296665,
        'Actual packet complete-cost and file joins')
    a=load(HERE/'adapter_v2.py','jcell_consumer_adapter',PINS[HERE/'adapter_v2.py'])
    h=load(SHARED/'probe.py','jcell_consumer_paid_owner',PINS[SHARED/'probe.py'])
    backend=load(SHARED/'backend_probe.py','jcell_consumer_backend',PINS[SHARED/'backend_probe.py'])
    owner=h.load(h.OBSERVER,'jcell_consumer_observer_owner',h.OBSERVER_SHA)
    legacy=h.load(h.LEGACY,'jcell_consumer_lifecycle_owner',h.LEGACY_SHA)
    sources.update({str(p):v for p,v in {**backend._BP_PINS,**owner.OWNERS,**legacy.PINS}.items()})
    require(not os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE'),
            'Default unmodified complete-operation weights')
    tariff_path=NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope/data/default_weights.json'
    tariffs=read(tariff_path)
    require(sources[str(tariff_path)]=='f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a'
        and all(tariffs['dtype_rates'][k]==v for k,v in
                dict(bool=1,float32=1,float64=2,int32=1,int64=2).items()), 'Exact dtype tariff basis')
    certificate=cost_certificate(manifest,a)
    text,proof=derive(a,tick=tick); local,local_proof=derive(a,local=True,tick=tick)
    require(proof['capture']==c['capture_source_proof'],'Same exact parent capture source inverse')
    bind(sources,tick); bind(inputs,tick)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No admission numerical imports')
    return dict(source_bindings=sources,input_bindings=inputs,capture_seals=CAPTURE_CHAINS,
        manifest=manifest,packet_proof=packet_proof,cost_certificate=certificate,
        source_proof=proof,local_source_proof=local_proof,source=text,local_source=local,
        control_diagnostics=read(HERE/'capture_v1/call_00_diagnostics.json')),a,h,backend,owner,legacy


def partial_observers(owner,a,local=False):
    (domain,comp,extra),parent_proof=owner.observer_sources()
    count,selectors=(1,0) if local else (2,5)
    domain,dp=splice(domain,[('len(self.variances) == 16 and len(self.selectors) == 47',
        f'len(self.variances) == {count} and len(self.selectors) == {selectors}')])
    extra,ep=splice(extra,[('dict(layer=len(self.small.variances),',
                            'dict(layer=8+len(self.small.variances),')])
    env=dict(c=types.SimpleNamespace(require=require),require=require)
    exec(compile(domain,'<exact-partial-domain-owner>','exec'),env)
    env.update(Domain=env['DomainObserver'],bound_depths=owner.base().bound_depths,
        EXTRA_LAYOUTS=owner.EXTRA_LAYOUTS,EXTRA_BINDINGS=owner.EXTRA_BINDINGS,
        eligible=owner.eligible,product_flops=owner.product_flops)
    exec(compile(comp+'\n'+extra,'<exact-COMP-partial-observer>','exec'),env)
    source=(owner.TL/'native.py').read_text()
    decode=owner.function_source(source,'decode'); diagnostic=owner.function_source(source,'diagnostics')
    begin=diagnostic.index("    c.require(all(selectors[f'0:{slot}']")
    end=diagnostic.index('    return dict(',begin)
    keys="[]" if local else "[f'{l}:{slot}' for l in (8,9) for slot in range(2 if l == 9 else 3)]"
    diagnostic,fp=splice(diagnostic,[
        ("enumerate(packet['variances'])", "enumerate(packet['variances'],8)"),
        ("[f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)]",keys),
        ('len(variances) == 16',f'len(variances) == {count}'),
        ('variance_count=16,selector_count=47',f'variance_count={count},selector_count={selectors}'),
        (diagnostic[begin:end],''),
        ("x['minimum'] >= 1e-10","x['minimum'] > 1e-10")])
    env['hashlib']=hashlib
    exec(compile(decode+'\n'+diagnostic,'<original-domain-gates-partial-labels>','exec'),env)
    return env['Observer'],env['diagnostics'],dict(parent=parent_proof,domain=dp,extra=ep,
        diagnostic=fp,actual_predict_binary_identity=True,layers=[8] if local else [8,9],
        variance_cells=count,selector_cells=selectors,no_all16_or_all47_claim=True)


def npz_header(archive,member):
    """Bounded header read. No object arrays or guessed NPZ member schema."""
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())), 'Unique NPZ member names')
        name=member+'.npy'; info=z.getinfo(name)
        require(not info.is_dir() and info.file_size<256*1024**2,'Bounded declared NPY member')
        with z.open(name) as f:
            require(f.read(6)==b'\x93NUMPY','Actual NPY member')
            version=tuple(f.read(2)); require(version in ((1,0),(2,0),(3,0)),'NPY format version')
            size=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
            require(0<size<=65536,'Bounded literal NPY header')
            header=ast.literal_eval(f.read(size).decode('utf8' if version==(3,0) else 'latin1'))
            require(set(header)=={'descr','fortran_order','shape'} and isinstance(header['shape'],tuple)
                    and type(header['fortran_order']) is bool and isinstance(header['descr'],str)
                    and header['descr'] in ('<f4','<f8','<i8','<i4'), 'Plain fixed typed header only')
            require(8+(2 if version==(1,0) else 4)+size+
                    math.prod(header['shape'])*int(header['descr'][-1])==info.file_size,
                    'Exact uncompressed typed payload size')
            return dict(header,shape=list(header['shape']))


def decode_supply(np,result,tick):
    # The sole opening point for A is AFTER an actual positive, fully released
    # N marker and bill. There is deliberately no reference to a B path here.
    marker=OUTPUT/'call_01_complete.json'
    require(result['retired_call_count']==2 and result['null_byte_exact'] and
        read(marker)['passed'] and read(marker)['arm']=='N' and
        result['artifacts'][str(marker)]==sha(marker,tick), 'N fully sealed before any A opening')
    headers={}
    paths=list(SUPPLY_FILES)
    for p,d in SUPPLY_FILES.items():
        require(sha(p,tick,supply=True)==d,'Exact fixed existing A archive')
    covariance,planes=paths
    for name in ('mc_pre_m','mc_pre_v','mc_pre_k3','mc_pre_k4','mc_post','mc_post_var','mc_se'):
        h=npz_header(covariance,name)
        require(h['shape']==[16,1024] and h['descr']=='<f8','Original FP64 marginal/mean frame')
        headers[name]=h
    for name in ('mc_pre_cov','mc_post_cov'):
        h=npz_header(covariance,name)
        require(h['shape']==[16,1024,1024] and h['descr']=='<f4','Original FP32 covariance frame')
        headers[name]=h
    for name in ('P','rank'):
        h=npz_header(covariance,name)
        require(h['shape']==[] and h['descr'] in ('<i8','<i4'),'Original scalar P/rank')
        headers[name]=h
    for name in ('K21_8','K22_8','K31_8'):
        h=npz_header(planes,name)
        require(h['shape']==[1024,1024] and h['descr']=='<f4','Original ordered preactivation cumulant plane')
        headers[name]=h
    h=npz_header(planes,'layers')
    require(len(h['shape'])==1 and h['descr'] in ('<i8','<i4'),'Original plane layer roster')
    headers['layers']=h
    with np.load(covariance,allow_pickle=False) as c, np.load(planes,allow_pickle=False) as p:
        require(int(c['P'])==8000000 and int(c['rank'])==35 and
            tuple(int(x) for x in p['layers'])==tuple(range(8,15)), 'Fixed original rank/P/layer frame')
        # Basic selected views only; FNP conversions, finite/provenance algebra
        # and diagonal assembly are charged in every applicable arm context.
        answer=dict(mu=c['mc_pre_m'][8],C=c['mc_pre_cov'][8],k3=c['mc_pre_k3'][8],
            k4=c['mc_pre_k4'][8],v=c['mc_pre_v'][8],K21=p['K21_8'],K22=p['K22_8'],K31=p['K31_8'],
            post7=c['mc_post'][7],post8=c['mc_post'][8],pre9=c['mc_pre_m'][9])
    result['supply_bindings']={str(p):d for p,d in SUPPLY_FILES.items()}
    result['supply_headers']=headers
    result.update(supply_A_decoded_after_null_retirement=True,
        supply_A_null_marker_sha256=sha(marker),current_cell_centering_error_bound_available=False,
        original_A_execution_seal_available=False,authenticated_joint_noise_replica_available=False)
    return answer


def prepare_supply(raw,fnp,weights,refs):
    supply={k:fnp.array(v,dtype=fnp.float64,copy=True,order='C') for k,v in raw.items()}
    refs.extend(weakref.ref(v) for v in supply.values())
    require(all(bool(fnp.all(fnp.isfinite(v))) for v in supply.values()), 'Every supplied physical moment finite')
    require(bool(fnp.all(fnp.diag(supply['C'])>1e-10)) and bool(fnp.all(supply['v']>1e-10)),
            'Strict selected A prevariance without a denominator rescue')
    norms={}; old=(.0005967063208329167,.014840580590071591,.012015439279439958)
    for key,diagonal,limit,reference in zip(('K21','K22','K31'),('k3','k4','k4'),(.002,.05,.05),old):
        denominator=fnp.sum(supply[diagonal]*supply[diagonal])
        require(float(denominator)>0 and bool(fnp.isfinite(denominator)), 'Positive physical diagonal denominator')
        error=fnp.diag(supply[key])-supply[diagonal]
        value=float(fnp.sqrt(fnp.sum(error*error)/denominator))
        require(math.isfinite(value) and value<=limit and abs(value-reference)<=1e-10,
                'Unchanged V2 gross guard and recorded first-cell discrepancy, not a V1 rescue')
        norms[key]=value
        fnp.fill_diagonal(supply[key],supply[diagonal])
    affine=[]
    for layer,prior,current in ((8,'post7','mu'),(9,'post8','pre9')):
        actual=weights[layer].astype(fnp.float64).T@supply[prior]
        scale=float(fnp.sqrt(fnp.mean(supply[current]*supply[current])))
        maximum=float(fnp.max(fnp.abs(actual-supply[current])))
        # Fixed gross fixture/stream consistency test, not a precision bound
        # or a statistical confidence statement about the plane supply.
        require(math.isfinite(scale) and math.isfinite(maximum) and
                maximum<=1e-5*max(1.,scale),'Original canonical incoming/next affine mean frame')
        affine.append(dict(layer=layer,maxabs=maximum,scale=scale,limit=1e-5*max(1.,scale)))
    return supply,dict(gross_diagonal_relative=norms,affine=affine,
        independent_marginal_diagonals_installed=True,strength=1.,
        ordered_Q31_not_symmetrized=True,centering_error_bound_available=False)


def parameter_refs(module,refs):
    def visit(value):
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            refs.append(weakref.ref(value))
        elif isinstance(value,dict):
            for v in value.values(): visit(v)
        elif isinstance(value,(tuple,list)):
            for v in value: visit(v)
    for name in ('_GNN_PARAMS','_BD_COEFFICIENTS','_BDNH_PARAMETERS','_SCRATCH'):
        visit(getattr(module,name,None))


def save_arrays(np,sink,call,result):
    manifest={}; blocks=[]; offset=0
    for name,(row,raw) in sink.saved.items():
        manifest[name]=dict(row,offset=offset,bytes=len(raw),file_storage_order='C')
        blocks.append(raw); offset+=len(raw)
    require(offset<40*1024**2, 'Only fixed local response readouts, not four full state copies')
    path=OUTPUT/f'call_{call:02d}_readouts.bin'
    with path.open('xb') as stream:
        for raw in blocks: stream.write(raw)
    result['artifacts'][str(path)]=sha(path)
    return dict(path=str(path),sha256=sha(path),bytes=offset,arrays=manifest)


def lifecycle(flops,ctx,bill_path,legacy,call,arm):
    # Exact original public reset function, called only after the own saved
    # bill and BudgetContext.__exit__. No counter rebasing inside prediction.
    record=legacy.reset_saved_history(flops,ctx,bill_path)
    record.update(call=call,rank=35,arm=arm,complete=True)
    return record


def run_arm(call,arm,result,context,a,h,owner,legacy,metrics,np,flops,fnp,MLP,seed_fn,materialise,tiny,raw,supply_raw,tick):
    """One fresh numerical subject; returns detached outputs/metadata only."""
    local=arm=='ALL-local'; is_tiny=arm=='tiny'
    limit=tiny.FLOP_CAP if is_tiny else LOCAL_CAP if local else PARTIAL_CAP
    require(not is_tiny or limit==10_000_000,'Exact independently fixed tiny10MF cap')
    refs=[]; subject=observer=sink=served=materialized=mlp=supply=None
    input_record=None; input_life=None; in_flops=0
    fixture_path=next(p for p in context['input_bindings'] if p.endswith('/0035/weights.npy'))
    row=read(Path(fixture_path).with_name('row_manifest.json'))
    if not is_tiny:
        weights=np.load(fixture_path,mmap_mode='r',allow_pickle=False)
        require(weights.shape==(16,1024,1024) and weights.dtype==np.float32,
                'Same complete original canonical physical weights')
        started=time.monotonic()
        with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:
            layers=[fnp.array(w,dtype=fnp.float32) for w in weights]
            require(all(bool(fnp.all(fnp.isfinite(w))) for w in layers),'Paid full input finite scans')
        input_record=dict(call=call,rank=35,arm=arm,**metrics(input_ctx,started),layers=16,
            official_fnp_type=all(isinstance(w,fnp.ndarray) for w in layers),
            exact_bytes=all(w.tobytes()==weights[i].tobytes() for i,w in enumerate(layers)),
            independent_weight_buffers=all(not np.may_share_memory(np.asarray(w),weights) for w in layers),
            weight_file_sha256=context['input_bindings'][fixture_path],charged_to_participant=False,
            included_in_complete_research_bill=True,context_closed=True)
        require(0<input_record['flops']<=64*1024*1024 and all(input_record[k] for k in
            ('official_fnp_type','exact_bytes','independent_weight_buffers')), 'Original exact input convention')
        require(input_record['wall_s']<120 and input_record['context_wall_s']<120 and
            input_record['peak_rss_bytes']<RSS and all(math.isfinite(input_record[k]) and input_record[k]>=0
            for k in legacy.TIMINGS),'Actual complete input context resources')
        in_flops=input_record['flops']
        ip=OUTPUT/f'call_{call:02d}_input_bill.json'; write(ip,input_record)
        ir=weakref.ref(input_ctx); input_life=lifecycle(flops,input_ctx,ip,legacy,call,arm)
        del input_ctx; gc.collect(); input_life['context_released']=ir() is None
        require(input_life['context_released'] and input_life['closed_context_bill_unchanged_by_reset'],
                'Independent input context actually released')
        write(OUTPUT/f'call_{call:02d}_input_reset.json',input_life)
        mlp=MLP(width=1024,depth=16,weights=layers,seed=int(seed_fn(row['mlp_seed'])),name='jcell-fixed35-'+arm)
        refs.extend(weakref.ref(w) for w in layers)
        del layers,weights
    started=time.monotonic(); answer_metadata={}
    with flops.BudgetContext(flop_budget=limit-in_flops,wall_time_limit_s=120,quiet=True) as ctx:
        subject=types.ModuleType('jcell_private_'+arm.replace('-','_'))
        subject.__file__='<jcell-fixed-current-c2-'+arm+'>'
        subject._jca_emit=lambda *unused:None
        exec(compile(context['local_source'] if local else context['source'],subject.__file__,'exec'),vars(subject))
        if is_tiny:
            answer_metadata=tiny.qualify(fnp,subject)
            require(isinstance(answer_metadata,dict) and answer_metadata.get('passed') is True,
                    'New independent tiny algebra/null qualification')
            json.dumps(answer_metadata,allow_nan=False)
            diagnostic=dict(call=call,rank=35,arm=arm,tiny=answer_metadata,contains_no_borrowed_arrays=True)
        else:
            entry=context['manifest']['packets']['entry8']
            subject._jca_H7=restore(context['manifest']['packets']['head7']['state'],raw,np,fnp,refs,'head7.state')
            subject._jca_entry={} if local else {k:restore(v,raw,np,fnp,refs,'entry8.'+k) for k,v in entry.items()}
            Sink=replay_sink_class(a); sink=Sink(fnp,context['manifest'],arm); sink.ctx=ctx
            subject._jca_emit=sink
            supply_meta=None
            if arm!='N':
                supply,supply_meta=prepare_supply(supply_raw,fnp,mlp.weights,refs)
            def intervene(mu,C,K21,k4,k22):
                # N returns the same original objects. No captured pre-plane
                # is substituted, even if it would conceal a restart mismatch.
                if arm=='N': return mu,C,K21,k4,k22,None
                return (mu,C,supply['K21'] if arm in ('T','TQ') else K21,
                    supply['k4'] if arm in ('Q','TQ') else k4,
                    supply['K22'] if arm in ('Q','TQ') else k22,
                    supply['K31'] if arm in ('Q','TQ') else None)
            subject._jca_intervene=intervene
            if local:
                subject._jca_all_inputs=tuple(supply[k] for k in ('mu','C','K21','k4','K22','K31'))
            Observer,diagnostics,op=partial_observers(owner,a,local)
            observer=Observer(subject); observer.begin(); observer.ctx=ctx
            estimator=subject.Estimator()
            try:
                served=owner.base().observed_predict(estimator,mlp,observer)
                require(isinstance(served,np.ndarray) and served.shape==((1 if local else 2),1024)
                        and served.dtype==np.float32, 'Actual two-layer or local projected served output')
                materialized=materialise(served,depth=1 if local else 2,width=1024)
                require(bool(fnp.all(fnp.isfinite(served))) and bool(fnp.all(served>=0)),
                        'Paid finite/nonnegative actual served vectors')
                values=materialized.tobytes()
                require(served.tobytes()==values,'Original materialization preserves exact bytes')
            finally:
                observer.ctx=None; sink.ctx=None
            parameter_refs(subject,refs)
    # Model, packet and parameter owners are still alive only until the exact
    # original closed bill/reset sequence below completes; nothing is reused.
    bill=dict(call=call,rank=35,arm=arm,**metrics(ctx,started),includes_initialization=True,
        complete_research_flops=int(ctx.flops_used)+in_flops,physical_flops=int(ctx.flops_used),
        input_flops=in_flops,not_a_complete_estimator_score=True)
    if not is_tiny:
        packet,startup,dispatches=observer.finish(); domain=diagnostics(np,packet)
        checkpoint=sink.finish()
        require(domain['variance_count']==(1 if local else 2) and domain['selector_count']==(0 if local else 5),
                'Honest local domain roster, no borrowed all16/all47 qualification')
        if arm=='N':
            old=context['control_diagnostics']['domains']
            expected_variance={str(i):old['variance_domain'][str(i)] for i in (8,9)}
            keys=('8:0','8:1','8:2','9:0','9:1')
            expected_selection={k:old['selection_cells'][k] for k in keys}
            require(domain['variance_domain']==expected_variance and domain['selection_cells']==expected_selection
                    and checkpoint['null_byte_exact'], 'N exact current8/9 physical/selector bytes and carried checkpoints')
            original=np.load(HERE/'capture_v1/call_00_rank35.npy',allow_pickle=False)
            require(original.shape==(16,1024) and original.dtype==np.float32 and
                    original[8:10].tobytes()==values,'N also replays original captured served8/9 bytes')
            del original
            old_dispatch=[x for x in context['control_diagnostics']['dispatches'] if x['layer'] in (8,9)]
            strip=lambda xs:[{k:v for k,v in x.items() if k!='operation_count'} for x in xs]
            require(strip(dispatches)==strip(old_dispatch), 'N exact physical dispatch shapes/costs, history counts reported')
        require(all(startup['prefix'][k] is True for k in ('P_empty_at_entry','A_empty_at_entry',
                'cstar_empty_at_entry','P_ready_at_binary_entry','A_ready_at_binary_entry','cstar_ready_at_binary_entry')),
                'Fresh actual frozen P/A/cstar in every arm')
        diagnostic=dict(call=call,rank=35,arm=arm,domains=domain,startup=startup,dispatches=dispatches,
            observer_proof=op,checkpoints=checkpoint,supply=supply_meta,
            readouts=save_arrays(np,sink,call,result),evaluation_B_accessed=False,
            source_proof=context['local_source_proof'] if local else context['source_proof'])
        output=OUTPUT/f'call_{call:02d}_served.npy'
        with output.open('xb') as stream:
            np.save(stream,np.frombuffer(values,dtype=np.float32).reshape((1 if local else 2),1024),allow_pickle=False)
        bill.update(output=str(output),output_sha256=sha(output),output_bytes_sha256=digest(values))
    bill_path=OUTPUT/f'call_{call:02d}_bill.json'; write(bill_path,bill)
    context_ref=weakref.ref(ctx); life=lifecycle(flops,ctx,bill_path,legacy,call,arm)
    del ctx; gc.collect(); life['context_released']=context_ref() is None
    require(life['context_released'] and life['closed_context_bill_unchanged_by_reset'],
            'Actual saved partial context retired, not a reset heartbeat')
    subject_ref=weakref.ref(subject)
    observer_ref=None if observer is None else weakref.ref(observer)
    sink_ref=None if sink is None else weakref.ref(sink)
    if not is_tiny:
        del estimator,intervene,packet
    # Sever only our disposable private module after the bill and reset.
    subject.__dict__.clear()
    del subject,observer,sink,served,materialized,mlp,supply
    gc.collect()
    require(subject_ref() is None and (observer_ref is None or observer_ref() is None)
        and (sink_ref is None or sink_ref() is None) and all(ref() is None for ref in refs),
        'All input, entry, current parameter, estimator, callback and output owners released')
    life.update(all_parameter_and_estimator_owners_retired=True,finished_utc=datetime.now(timezone.utc).isoformat())
    write(OUTPUT/f'call_{call:02d}_lifecycle.json',life)
    write(OUTPUT/f'call_{call:02d}_diagnostics.json',diagnostic)
    require(0<bill['flops']<=bill['complete_research_flops']<=limit and bill['wall_s']<120 and bill['context_wall_s']<120 and
        all(math.isfinite(bill[k]) and bill[k]>=0 for k in legacy.TIMINGS) and
        bill['peak_rss_bytes']<RSS,'Complete own partial research cost/resources, no borrowed full-card bill')
    tick()
    return bill,diagnostic,life


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in KEYS: parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(not OUTPUT.exists() and not HERE.is_symlink(),'Exclusive never-resumed consumer output')
    began=time.monotonic(); OUTPUT.mkdir()
    result=dict(schema=SCHEMA,passed=False,completed=False,roster=list(ROSTER),ranks=[35]*6,
        candidate_sha256=CANDIDATE_SHA,retired_call_count=0,full_model_calls=0,
        partial_model_calls=0,tiny_contexts=0,source_bindings={},input_bindings={},artifacts={},
        trials=[],diagnostic_records=[],tiny_qualified=False,null_byte_exact=False,
        supply_A_decoded_after_null_retirement=False,evaluation_B_accessed=False,
        score_measured=False,parameter_updates=0,automatic_next_stage=False,
        current_cell_centering_error_bound_available=False,provisional_joint_supply_witness_measured=False)
    def tick():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,
                'Entire consumer process600s/8GB; actual external watch owns full tree')
    try:
        context,a,h,backend,owner,legacy=admission(args,tick)
        for name in ('source_bindings','input_bindings','capture_seals','packet_proof','cost_certificate',
                     'source_proof','local_source_proof'):
            result[name]=context[name]
        # The exact original closed-ledger helper, in its own namespace.
        # Never merge returned helper dictionaries into their definition globals.
        source=(SHARED/'check4_v2.py').read_text()
        lo,hi=a.function_span(source,'_v2_ledger_metrics')
        env=dict(time=time,hashlib=hashlib,json=json,require=require)
        exec(compile(source.encode()[lo:hi].decode(),'<exact-current-closed-ledger>','exec'),env)
        metrics=env['_v2_ledger_metrics'](legacy.metrics)
        result['lifecycle_source_proof']=dict(owner_sha256=h.LEGACY_SHA,
            metrics_owner_sha256=PINS[SHARED/'check4_v2.py'],
            original_reset_source_sha256=digest(owner.function_source(
                h.LEGACY.read_text(),'reset_saved_history').encode()),
            original_input_copy_and_closed_reset_functions=True,
            partial_output_shape_explicit=True,all_research_inputs_in_complete_bill=True)
        tiny=load(HERE/'tiny.py','jcell_fixed_tiny_qualifier',args.tiny_sha256)
        require(callable(tiny.qualify),'Explicit independent tiny API qualify(fnp,module)')
        backend.backend_select(); h.environment()
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No pre-admission numerics')
        write(OUTPUT/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        raw=Path(context['manifest']['packet_file']).read_bytes()
        require(len(raw)==351195456 and digest(raw)==PACKET_SHA,'Unmodified detached target-blind packet')
        supply_raw=None
        with threadpool_limits(limits=1):
            result['threadpools']=backend.backend_threadpool_info(threadpool_info,result,tick)()
            for call,arm in enumerate(ROSTER):
                require(call==result['retired_call_count'],'No omitted, repeated or automatically retried arm')
                if arm=='N': require(result['tiny_qualified'],'Tiny before any wide consumer')
                if arm=='T': supply_raw=decode_supply(np,result,tick)
                if arm in ('Q','TQ','ALL-local'):
                    require(supply_raw is not None and result['supply_A_decoded_after_null_retirement'],
                            'Same exact A supply, no intermediate retuning')
                bill,diagnostic,life=run_arm(call,arm,result,context,a,h,owner,legacy,metrics,
                    np,flops,fnp,MLP,derive_estimator_seed,materialise_predictions,tiny,raw,supply_raw,tick)
                # Every arm's actual complete numerical outputs, diagnostics,
                # bill and original closed context release precede progress.
                for path in OUTPUT.glob(f'call_{call:02d}_*'):
                    require(path.is_file() and not path.is_symlink(),'Owned flat per-call artifact')
                    result['artifacts'][str(path)]=sha(path,tick)
                dp=OUTPUT/f'call_{call:02d}_diagnostics.json'
                result['trials'].append(bill)
                result['diagnostic_records'].append(dict(path=str(dp),sha256=sha(dp)))
                if arm=='tiny':
                    result['tiny_qualified']=True; result['tiny_contexts']=1
                elif arm=='N':
                    require(diagnostic['checkpoints']['null_byte_exact'],'Only actual new N can qualify restart')
                    result['null_byte_exact']=True
                result['partial_model_calls']+=not (arm=='tiny')
                cp=OUTPUT/f'call_{call:02d}_complete.json'
                write(cp,dict(passed=True,completed=True,call=call,rank=35,arm=arm,
                    context_retired=True,all_parameter_and_estimator_owners_retired=True,
                    bill_sha256=sha(OUTPUT/f'call_{call:02d}_bill.json'),
                    lifecycle_sha256=sha(OUTPUT/f'call_{call:02d}_lifecycle.json'),
                    diagnostic_sha256=sha(dp),elapsed_s=time.monotonic()-began))
                result['artifacts'][str(cp)]=sha(cp)
                result['retired_call_count']+=1
                print(json.dumps(dict(event='retired',call=call,rank=35,arm=arm,complete_sha256=sha(cp))),flush=True)
                tick()
        del raw,supply_raw
        gc.collect()
        require(result['retired_call_count']==6 and result['partial_model_calls']==5 and
            result['tiny_qualified'] and result['null_byte_exact'],'Exactly the fixed six completed contexts')
        result['complete_consumer_flops']=sum(x['complete_research_flops'] for x in result['trials'])
        require(result['complete_consumer_flops']+403695296665<=1_425_000_000_000,
                'Complete acquisition plus every research context fits original aggregate ceiling')
        for name in ('source_bindings','input_bindings','artifacts'):
            bind(result[name],tick)
        bind(result['supply_bindings'],tick,supply=True)
        # Include A in the final authenticated input union only after the true
        # N-before-A event has occurred. The supervisor hashes, never decodes it.
        result['input_bindings'].update(result['supply_bindings'])
        result.update(passed=True,completed=True,own_complete_bills=True,
            all_parameter_and_estimator_owners_retired=True,all_outputs_sealed_before_evaluation_B=True,
            hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=CAP or result['peak_rss_bytes']>=RSS:
        result.update(passed=False,completed=False)
    # Failure-only publication also remains possible after a worker guard;
    # any late or partial success chain is independently rejected by watch.
    write(OUTPUT/'result.json',result); rh=sha(OUTPUT/'result.json')
    ok=bool(result['passed'] and time.monotonic()-began<CAP)
    write(OUTPUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
    ch=sha(OUTPUT/'completion.json'); elapsed=time.monotonic()-began
    ok=bool(ok and elapsed<CAP)
    write(OUTPUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
        completion_sha256=ch,elapsed_s=elapsed))
    return 0 if ok and time.monotonic()-began<CAP else 1


if __name__=='__main__':
    raise SystemExit(main())
