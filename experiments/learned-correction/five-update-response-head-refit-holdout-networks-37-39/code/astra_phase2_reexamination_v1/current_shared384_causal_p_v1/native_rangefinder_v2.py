"""Inert native-order CPU rangefinder FORWARD adapter; not a trained estimator.

V2 metadata-only repair: actual two schedule definitions and exact Torch
synthetic namespace ownership. Frozen V1 never reached numerical execution.

prepare -> select_backend -> caller imports NumPy/Torch/threadpoolctl -> attest
-> build_numpy_forward. The caller owns process/resources/inputs/output seals.
No numerical import, file output, head import, autograd implementation or run at
module load. This file does not authorize a full forward, backward or fit.

Forward is the exact c2 five-product association with original packed scratch
strides and production NumPy/OpenBLAS. Saved-entry Y bytes must STILL qualify.
Any later custom VJP is a derivative of the intended real-algebra map, not of
FP32 rounding. A per-matmul VJP saves only operands, not Winograd descendants:
dA=G@B.T, dB=A.T@G. All J/lambda/Q/core/Omega paths must remain connected.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
C2 = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
C2_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
AP = Path('/Users/matt/anaconda3/bin/python')
TSITE = AP.parent.parent/'lib/python3.11/site-packages'
OSITE = NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages'
LIMITS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
SCHEMA = 'current_shared384_native_rangefinder_source_v1'
PINS = {
    HERE/'native_rangefinder.py':'1db946b41d6888ea586ce3dbb86a02f46874329cf89ed7e742cc4d33c4b521b1',
    TSITE/'torch/_ops.py':'7e88a7ec282d5d445969d0d00afc7b3ce23ceacc88935a96d52e952f3719e8a4',
    TSITE/'torch/_classes.py':'2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf',
    C2:C2_SHA,
    HERE/'shared_operator.py':'09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2',
    OSITE/'numpy/__init__.py':'92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a',
    OSITE/'numpy/__config__.py':'97bedd19442e4714a9147f49cd93357032d21d791d87ca1367f44766db0771e6',
    OSITE/'numpy/_core/_multiarray_umath.cpython-311-darwin.so':'b9a53273d646f3ffbf8307725caa306bd81c7ade0815ae999af98e0cf3e8ebec',
    OSITE/'numpy/.dylibs/libscipy_openblas64_.dylib':'bd3ad3014a7977f69921ceb2f5d7b0a8fb4b96dfb0653f87bd3bfba5af806d78',
    OSITE/'threadpoolctl.py':'12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b',
    TSITE/'torch/__init__.py':'9801e75447c7f585545f989f8a21940b60e0c4cc888effb2f08e739664b4b904',
    TSITE/'torch/version.py':'a8cdda731a4a5a72c4e569f2e16560b3e559e067e620857f48dce602db7314a1',
    TSITE/'torch/_C.cpython-311-darwin.so':'63da29ef5da262164de48f55d3c6417f7508eec4a22670dcb54463b6b100e288',
    TSITE/'torch/lib/libtorch_cpu.dylib':'8bf93b74995987c7edea9ad6d7958ff309eb62ea9de3f4bb22922e638cd22713',
    TSITE/'torch/lib/libtorch_python.dylib':'f5ffcfadd510406cb8b3b4781cef4d4176d8ca1ad34987bd256507c8937405af',
    TSITE/'torch/lib/libc10.dylib':'5c29250c6e2981a10b21688276330c20df3a0f789e7ae46b7bcae4aa0c5c6410',
    TSITE/'torch/lib/libomp.dylib':'7ab6e8bf7c7886311047e611f1a388ba8effd1c9645a36687dde90b06806c525',
    TSITE/'torch/lib/libtorch.dylib':'6d36d4b9a84b40cc91b45d5549aecd656113eacc21f6e7b331398cda73af8190',
    TSITE/'torch/lib/libtorch_global_deps.dylib':'ce0e694d89f80acdbdf470b67e44ca0aa859213791921530bb68a77217f941c8',
    TSITE/'torch/lib/libshm.dylib':'06c74f951867ef5d158e982eb5732f5ed93ea128d5c8726c797695fc74821371',
}
FUNCTION_COUNTS = {'_cp3_bind_arena_views':1,'_split_views_6':1,
    '_packed_product_6':1,'_square_schedule_reference':1,'_square_schedule':2,
    '_square_product':2,'_packed_frame':1,'_packed_product_cached':1,
    '_packed_product_3':2,'_packed_product_4':2,'_packed_product_5':2,
    '_mm':4,'_shared_old384_mm':1}
LITERAL_COUNTS = {'_WINOGRAD':1,'_PACKED_ELEMENTS':3,'_PACKED_TILE':2,
    '_PACKED_PUSH':3,'_PACKED_READ':3,'_PACKED_SQUARE':1,'_PACKED_SQUARE4':1,
    '_PACKED_SQUARE5':1,'_PACKED_SQUARE6':1,'_SQUARE_COSTS':2,
    '_SQUARE_EQUAL_DEPTHS':2,'_SQUARE_CROSS_DEPTHS':2}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path, tick=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(512*1024):
            h.update(block)
            if tick is not None:tick()
    return h.hexdigest()


def _bind(mapping, tick=None):
    for path,expected in mapping.items():
        require(Path(path).is_file() and sha(path,tick)==expected,'Pinned source: '+str(path))


def _cold():
    require(not any(k.split('.')[0] in ('numpy','torch','flopscope','scipy','threadpoolctl')
                    for k in sys.modules),'Explicit admission/selection before numerical imports')


def prepare(reviewed_sha256, tick=None):
    """Stdlib only: authenticate and extract actual LAST c2 definitions once.

    Positive entry8 bundle/control admission belongs to the future unit worker;
    this source builder does not claim it inspected a numerical saved bundle.
    """
    _cold()
    sources={str(p.resolve()):v for p,v in PINS.items()}
    sources[str(Path(__file__).resolve())]=reviewed_sha256
    _bind(sources,tick)
    raw=C2.read_bytes();require(digest(raw)==C2_SHA,'Whole c2/head bytes unchanged')
    lines=raw.splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    tree=ast.parse(raw.decode())
    if tick is not None:tick()
    def span(node):
        lo=offsets[node.lineno-1]+node.col_offset
        hi=offsets[node.end_lineno-1]+node.end_col_offset
        return lo,hi,raw[lo:hi].decode()
    chunks=[];records={};values={}
    for name,count in LITERAL_COUNTS.items():
        nodes=[n for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        require(len(nodes)==count,'Actual literal multiplicity: '+name)
        node=nodes[-1];lo,hi,text=span(node)
        values[name]=ast.literal_eval(node.value);chunks.append(text)
        records[name]=dict(count=count,selected_line=node.lineno,byte_span=[lo,hi],sha256=digest(text.encode()))
    require(values['_PACKED_ELEMENTS']==292225024 and values['_PACKED_TILE']==8192
            and values['_WINOGRAD'] is True
            and tuple(x[0] for x in values['_SQUARE_COSTS'])==(3,4,5)
            and max(max(row) for row in values['_SQUARE_EQUAL_DEPTHS'])==5
            and max(values['_SQUARE_CROSS_DEPTHS'])==5,'Actual final no-D6 square schedules')
    for name,count in FUNCTION_COUNTS.items():
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(nodes)==count,'Actual function multiplicity: '+name)
        node=nodes[-1];lo,hi,text=span(node);chunks.append(text)
        records[name]=dict(count=count,selected_line=node.lineno,byte_span=[lo,hi],sha256=digest(text.encode()))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push']
    require(len(nodes)==1,'Unique current shared push owner')
    lo,hi,body=span(nodes[0])
    first=body.index('        weight = coefficient * coefficient\n')
    last=body.index("        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')",first)
    original=body[first:last]
    edits=[('        Omega = w32[previous, :rank]\n',''),("state['core']",'core')]
    changed=original;replacements=[]
    for old,new in edits:
        require(changed.count(old)==1,'Unique rangefinder argument-binding splice')
        pos=changed.index(old);changed=changed[:pos]+new+changed[pos+len(old):]
        replacements.append((pos,old,new))
    inverse=changed
    for pos,old,new in reversed(replacements):
        require(inverse[pos:pos+len(new)]==new,'Positional source inverse')
        inverse=inverse[:pos]+old+inverse[pos+len(new):]
    require(inverse==original and raw[:lo+first]+inverse.encode()+raw[lo+last:]==raw,
            'Complete c2 inverse; every other physical/head byte unchanged')
    dedented=''.join(line[4:] for line in changed.splitlines(keepends=True))
    require(''.join('    '+line for line in dedented.splitlines(keepends=True))==changed,
            'Exact rangefinder indentation inverse')
    forward=('def _rangefinder(J, coefficient, Omega, Qp, core):\n'
             '    old_columns = Qp is not None\n'+dedented+'    return Y\n')
    chunks.append(forward);source='\n\n'.join(chunks)+'\n'
    ast.parse(source)
    if tick is not None:tick()
    return dict(schema=SCHEMA,source_bindings=sources,input_bindings={},source=source,
        source_proof=dict(candidate_sha256=C2_SHA,actual_last_definitions=records,
            source_sha256=digest(source.encode()),rangefinder_original_sha256=digest(original.encode()),
            rangefinder_forward_sha256=digest(forward.encode()),whole_source_inverse=True,
            original_five_product_association=True,only_external_Omega_and_core_argument_binding=True,
            no_head_or_full_estimator_import=True,no_gradient_implemented=True),
        planning=dict(arena_elements=292225024,arena_bytes=1168900096,
            entry8_five_product_classical_flops=3372220416,
            entry8_five_product_native_source_upper=4162560000,
            initialization_and_bindings_upper=3000000000,
            entry8_ingress_finite_copy_bookkeeping_upper=250000000,
            entry8_cold_forward_source_upper=7412560000,
            source_price_only_not_an_actual_deployment_bill=True,
            full_model_memory_not_yet_qualified=True))


def select_backend(context, tick=None):
    """Explicit AP + production NumPy selection; deliberately NOT old attestor."""
    _cold();_bind(context['source_bindings'],tick)
    require(Path(sys.executable).resolve()==AP.resolve() and sys.version_info[:3]==(3,11,5)
            and not os.environ.get('PYTHONPATH') and not os.environ.get('FLOPSCOPE_SERVER_URL'),
            'Fixed fresh local AP3.11.5 interpreter')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in LIMITS),
            'Caller supplies nice10 and all six single-thread limits before imports')
    require(str(OSITE) not in sys.path,'Exactly one explicit production site prepend')
    sys.path.insert(0,str(OSITE))
    for name,root in (('numpy',OSITE),('threadpoolctl',OSITE),('torch',TSITE)):
        spec=importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None and
                Path(spec.origin).resolve().is_relative_to(root.resolve()),'Declared module origin: '+name)
    return dict(selection='AP CPU with explicit pinned production NumPy before Torch',
                numpy_site=str(OSITE),torch_site=str(TSITE),flopscope_imported=False)


def attest(context, np, torch, threadpool_info, tick=None):
    """New joint-backend attestation; actual loaded modules, no target access.

    Parent must call again after work and rehash this additive inventory. This
    makes no claim that a NumPy research operation has a native paid bill.
    """
    require(np.__version__=='2.4.6' and torch.__version__=='2.7.0'
            and Path(np.__file__).resolve()==(OSITE/'numpy/__init__.py').resolve()
            and Path(torch.__file__).resolve()==(TSITE/'torch/__init__.py').resolve()
            and torch.get_num_threads()==torch.get_num_interop_threads()==1,
            'Exact NumPy/Torch origins, versions and CPU thread configuration')
    require(not any(k.split('.')[0] in ('flopscope','scipy') for k in sys.modules),
            'No native client, SciPy or undeclared second numerical authority')
    _bind(context['source_bindings'],tick)
    modules={};sources=dict(context['source_bindings'])
    roots={'numpy':OSITE/'numpy','torch':TSITE/'torch','threadpoolctl':OSITE}
    synthetic={'torch.ops':('ops','torch._ops','_Ops','_ops.py'),
               'torch.classes':('classes','torch._classes','_Classes','_classes.py')}
    synthetic_seen={}
    for name,module in tuple(sys.modules.items()):
        root=roots.get(name.split('.')[0])
        if root is None:continue
        if name in synthetic:
            attribute,owner_name,class_name,sentinel=synthetic[name]
            owner=sys.modules.get(owner_name)
            expected_owner=(TSITE/'torch'/sentinel).resolve()
            require(isinstance(owner,types.ModuleType) and
                    module is getattr(torch,attribute) and module.__name__==name and
                    type(module) is vars(owner).get(class_name) and
                    type(module).__module__==owner_name and
                    vars(type(module)).get('__file__')==sentinel and
                    getattr(module,'__file__',None)==sentinel and
                    Path(owner.__file__).resolve()==expected_owner,
                    'Exact known Torch synthetic namespace, not an arbitrary relative file')
            path=expected_owner
            synthetic_seen[name]=dict(defining_module=owner_name,class_name=class_name,
                                      sentinel=sentinel,real_owner=str(path))
        else:
            path=getattr(module,'__file__',None)
            if not path:continue
            path=Path(path).resolve()
        require(path.is_relative_to(root.resolve()),'Actual imported owner: '+name)
        d=sha(path,tick)
        require(str(path) not in sources or sources[str(path)]==d,'No imported source conflict')
        sources[str(path)]=d;modules[name]=dict(path=str(path),sha256=d)
    require(set(synthetic_seen)==set(synthetic),'Both exact Torch synthetic namespace owners recorded')
    pools=threadpool_info();blas=[p for p in pools if p.get('user_api')=='blas']
    expected=(OSITE/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve()
    require(len(blas)==1 and Path(blas[0]['filepath']).resolve()==expected
            and blas[0]['internal_api']=='openblas' and blas[0]['num_threads']==1,
            'Only the exact production OpenBLAS leaf backend, one thread')
    for pool in pools:
        path=Path(pool['filepath']).resolve()
        require(str(path) in sources and pool['num_threads']==1,'Pinned loaded numerical library/thread count')
    return dict(schema='current_shared384_numpy_torch_cpu_attestation_v1',
        source_bindings=sources,modules=modules,threadpools=pools,
        synthetic_namespace_owners=synthetic_seen,
        actual_origins_qualified=True,production_numpy_leaf_backend_qualified=True,
        cpu_only_scope=True,flopscope_imported=False,native_or_deployment_bill_claimed=False)


def _forbidden_d6(*args, **kwargs):
    raise RuntimeError('Rangefinder output width384 cannot enter native PUSH-D6 branch')


class _Forward:
    """Caller-owned, non-reentrant scratch only; never retains trajectory inputs."""
    def __init__(self, context, np):
        self.np=np;self.busy=False;self.closed=False
        require(digest(context['source'].encode())==context['source_proof']['source_sha256'],
                'Immutable prepared native statements')
        names=('reshape','unstack','add','subtract','stack','copyto','matmul','concatenate','empty','zeros','float32')
        facade=types.SimpleNamespace(**{name:getattr(np,name) for name in names})
        self.env=dict(fnp=facade,_PACKED_FRAME_CACHE={},_P1_D6_832_BINDINGS=None,
                      _p1_d6_832_product=_forbidden_d6)
        exec(compile(context['source'],'<exact-c2-native-rangefinder-forward>','exec'),self.env)
        arena=np.empty((self.env['_PACKED_ELEMENTS'],),dtype=np.float32)
        bind=self.env['_cp3_bind_arena_views']
        self.env['_SCRATCH']=[arena,bind(arena,self.env['_PACKED_PUSH']),
                              bind(arena,self.env['_PACKED_READ']),None,None,None,None]

    def __call__(self, J, coefficient, Omega, Qp=None, core=None):
        np=self.np
        require(not self.closed and not self.busy,'One owned serial rangefinder workspace')
        require(isinstance(J,np.ndarray) and J.ndim==2,'Explicit native J array')
        a,b=J.shape
        require(384<=a<=1024 and 0<b<=2048 and b%2==0 and
                coefficient.shape==(b,) and Omega.shape==(a,384) and
                ((Qp is None and core is None) or
                 (Qp is not None and core is not None and Qp.shape==(a,384) and core.shape==(384,384))),
                'Original rank384/cohort rangefinder geometry; no rank policy parameter')
        values=(J,coefficient,Omega) if Qp is None else (J,coefficient,Omega,Qp,core)
        require(all(isinstance(v,np.ndarray) and v.dtype==np.float32 and np.isfinite(v).all()
                    for v in values),'Finite original FP32 inputs; no implicit dtype/scalar conversion')
        self.busy=True
        try:
            Y=self.env['_rangefinder'](J,coefficient,Omega,Qp,core)
            require(Y.shape==(a,384) and Y.dtype==np.float32 and np.isfinite(Y).all(),
                    'Finite original FP32 rangefinder output')
            return np.array(Y,dtype=np.float32,copy=True,order='K')
        finally:self.busy=False

    def close(self):
        require(not self.busy,'No workspace retirement during a call')
        self.env['_PACKED_FRAME_CACHE'].clear();self.env.clear();self.closed=True


def build_numpy_forward(context, np, attestation):
    """Allocates only when called inside the independently owned unit/CPU scope.

    The returned object accepts J,lambda,Omega,Qp,core and returns owned FP32 Y.
    The caller retains responsibility for its input/output arrays and close().
    This is NOT an autograd bridge; do not detach a trainable path into it yet.
    """
    require(attestation.get('production_numpy_leaf_backend_qualified') is True
            and attestation.get('actual_origins_qualified') is True
            and all(attestation['source_bindings'].get(k)==v for k,v in context['source_bindings'].items()),
            'Actual fresh joint backend attestation required before allocation')
    require(np is sys.modules.get('numpy'),'Exactly the attested loaded NumPy API')
    return _Forward(context,np)
