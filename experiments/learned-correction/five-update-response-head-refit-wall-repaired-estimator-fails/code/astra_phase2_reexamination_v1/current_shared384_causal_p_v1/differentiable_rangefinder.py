"""Inert per-product autograd bridge for the qualified native rangefinder.

No numerical import, worker, model, targets, fit or execution at module load.
Production: CPU FP32, fixed rank384, exact original NumPy/OpenBLAS products;
each custom node saves ORIGINAL Torch A/B only, never a Winograd descendant.
Backward is dA=G@B.T, dB=A.T@G. This is the intended real-algebra VJP, not the
derivative of FP32 rounding. No frozen state, head or gradient is substituted.

The mathematical factory is separately CPU FP64/rank3 and direct matmul. It is
not a production fallback. First-order tangent/checkpoint and full-forward
fidelity remain prerequisites; merely building this module grants neither.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import weakref

HERE=Path(__file__).resolve().parent
NATIVE=HERE/'native_rangefinder_v2.py'
NATIVE_SHA='e84ad190067382cf2755548df4b8f0b59281b99083c87287f6ccdccf502ea042'
SCHEMA='current_shared384_differentiable_rangefinder_source_v1'

# Added method only. The original constructor, __call__, close, native source
# extractor, backend selection and attestation remain byte-for-byte present.
PRODUCT_METHOD='''    def product(self, left, right):
        np=self.np
        require(not self.closed and not self.busy, 'One owned serial native product')
        require(isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and
                left.ndim==right.ndim==2 and left.dtype==right.dtype==np.float32 and
                0<left.shape[0]<=2048 and 0<left.shape[1]<=2048 and
                right.shape==(left.shape[1],384), 'Exact production rangefinder product domain')
        arena=self.env['_SCRATCH'][0]
        require(not np.may_share_memory(left,arena) and not np.may_share_memory(right,arena),
                'Trajectory operands never borrow reusable scratch storage')
        require(np.isfinite(left).all() and np.isfinite(right).all(), 'Finite native product inputs')
        self.busy=True
        try:
            value=self.env['_shared_old384_mm'](left,right)
            require(value.dtype==np.float32 and value.shape==(left.shape[0],384) and
                    np.isfinite(value).all(), 'Finite exact native product output')
            return np.array(value,dtype=np.float32,copy=True,order='K')
        finally:self.busy=False

'''


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def _owner(tick=None):
    raw=NATIVE.read_bytes();require(digest(raw)==NATIVE_SHA,'Frozen native workspace source')
    if tick is not None:tick()
    spec=importlib.util.spec_from_file_location('bridge_pinned_native_rangefinder',NATIVE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,raw


def prepare(reviewed_sha256,tick=None):
    """Stdlib-only source preparation; no fixture or array admission is claimed."""
    require(digest(Path(__file__).read_bytes())==reviewed_sha256,'Explicit reviewed bridge source')
    native,raw=_owner(tick)
    context=native.prepare(NATIVE_SHA,tick)
    context['source_bindings']=dict(context['source_bindings'])
    context['source_bindings'][str(Path(__file__).resolve())]=reviewed_sha256
    original=raw.decode();marker='    def close(self):\n'
    require(original.count(marker)==1,'Unique original workspace close anchor')
    position=original.index(marker)
    extended=original[:position]+PRODUCT_METHOD+original[position:]
    require(extended[position:position+len(PRODUCT_METHOD)]==PRODUCT_METHOD and
            extended[:position]+extended[position+len(PRODUCT_METHOD):]==original,
            'Complete native-v2 source inverse; additive product method only')
    # Reuse the already-authenticated exact five-product expression, including
    # its lambda-square, old-absent branch and native addition association.
    text=context['source'];lines=text.encode().splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_rangefinder']
    require(len(nodes)==1,'Unique authenticated five-product formula')
    node=nodes[0];lo=offsets[node.lineno-1]+node.col_offset
    hi=offsets[node.end_lineno-1]+node.end_col_offset
    formula=text.encode()[lo:hi].decode()+'\n'
    require(digest(formula.encode())==context['source_proof']['rangefinder_forward_sha256'],
            'Original formula bytes; no whole-Y derivative replacement')
    ast.parse(extended);ast.parse(formula)
    if tick is not None:tick()
    return dict(schema=SCHEMA,native_context=context,source_bindings=dict(context['source_bindings']),
        workspace_source=extended,workspace_source_sha256=digest(extended.encode()),
        formula_source=formula,formula_sha256=digest(formula.encode()),
        source_proof=dict(native_source_sha256=NATIVE_SHA,
            workspace_source_sha256=digest(extended.encode()),whole_workspace_inverse=True,
            only_additive_guarded_product_method=True,formula_sha256=digest(formula.encode()),
            original_five_product_association=True,original_operands_saved_for_backward=True,
            no_Winograd_graph_or_state_detachment=True,real_algebra_VJP_not_rounding_derivative=True))


def select_backend(prepared,tick=None):
    native,unused=_owner(tick)
    return native.select_backend(prepared['native_context'],tick)


def attest(prepared,np,torch,threadpool_info,tick=None):
    native,unused=_owner(tick)
    return native.attest(prepared['native_context'],np,torch,threadpool_info,tick)


def _attested(prepared,np,attestation):
    require(np is sys.modules.get('numpy') and
            attestation.get('production_numpy_leaf_backend_qualified') is True and
            attestation.get('actual_origins_qualified') is True and
            all(attestation['source_bindings'].get(k)==v for k,v in prepared['source_bindings'].items()),
            'Actual prepared joint CPU/NumPy backend and complete source inventory')


def build_workspace(prepared,np,attestation):
    """Caller-owned allocation. Keep open through checkpoint recomputation.

    Caller must retire graphs and tensors, close workspace, delete it, and check
    all actual owners. This function does not create a budget or process scope.
    """
    _attested(prepared,np,attestation)
    require(digest(prepared['workspace_source'].encode())==prepared['workspace_source_sha256'],
            'Exact prepared workspace extension')
    env=dict(__file__=str(NATIVE),__name__='bridge_exact_native_workspace_extension')
    exec(compile(prepared['workspace_source'],'<native-v2-additive-product-method>','exec'),env)
    workspace=env['build_numpy_forward'](prepared['native_context'],np,attestation)
    workspace._bridge_workspace_sha256=prepared['workspace_source_sha256']
    return workspace


def _matrix(torch,value,dtype,name):
    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and
            value.dtype==dtype and value.ndim==2 and
            not value.is_conj() and not value.is_neg(),name+': explicit ordinary CPU matrix')


def _mm_class(torch,dtype,forward_product):
    class Product(torch.autograd.Function):
        @staticmethod
        def forward(ctx,A,B):
            _matrix(torch,A,dtype,'A');_matrix(torch,B,dtype,'B')
            require(A.shape[1]==B.shape[0],'Matrix contraction dimensions')
            # Do not save NumPy views, packed transforms, workspace or outputs.
            # Original tensors preserve ALL upstream J/Q/core/lambda paths.
            ctx.save_for_backward(A,B)
            return forward_product(A,B)

        @staticmethod
        def backward(ctx,G):
            A,B=ctx.saved_tensors
            _matrix(torch,G,dtype,'output cotangent')
            require(G.shape==(A.shape[0],B.shape[1]) and bool(torch.isfinite(G).all()),
                    'Finite correctly shaped cotangent')
            dA=G@B.T if ctx.needs_input_grad[0] else None
            dB=A.T@G if ctx.needs_input_grad[1] else None
            require(all(v is None or bool(torch.isfinite(v).all()) for v in (dA,dB)),
                    'Finite real-algebra product adjoints')
            return dA,dB
    return Product


class _Rangefinder:
    def __init__(self,torch,prepared,mm,dtype,rank,mathematical):
        require(digest(prepared['formula_source'].encode())==prepared['formula_sha256'],
                'Exact authenticated five-product formula')
        env={'_shared_old384_mm':mm}
        exec(compile(prepared['formula_source'],'<unchanged-rangefinder-Torch-association>','exec'),env)
        self._formula=env['_rangefinder'];self.mm=mm;self.torch=torch
        self.dtype=dtype;self.rank=rank;self.mathematical=mathematical
        self.source_proof=prepared['source_proof']

    def __call__(self,J,coefficient,Omega,Qp=None,core=None):
        torch=self.torch
        _matrix(torch,J,self.dtype,'J');_matrix(torch,Omega,self.dtype,'Omega')
        a,b=map(int,J.shape)
        require(isinstance(coefficient,torch.Tensor) and coefficient.device.type=='cpu'
                and coefficient.dtype==self.dtype and tuple(coefficient.shape)==(b,)
                and tuple(Omega.shape)==(a,self.rank),'Explicit signed lambda/Omega domain')
        if self.mathematical:
            require(3<=a<=12 and 0<b<=24,'Separate fixed tiny FP64 mathematical domain')
        else:
            require(384<=a<=1024 and 0<b<=2048 and b%2==0,'Original production rank384 cohort domain')
        if Qp is None:
            require(core is None,'No-old branch has neither basis nor core')
            values=(J,coefficient,Omega)
        else:
            _matrix(torch,Qp,self.dtype,'Qp');_matrix(torch,core,self.dtype,'core')
            require(tuple(Qp.shape)==(a,self.rank) and tuple(core.shape)==(self.rank,self.rank),
                    'Independent old basis/core coordinates; never infer core from F')
            values=(J,coefficient,Omega,Qp,core)
        require(all(bool(torch.isfinite(v).all()) for v in values),'Finite trajectory inputs')
        Y=self._formula(J,coefficient,Omega,Qp,core)
        require(tuple(Y.shape)==(a,self.rank) and Y.dtype==self.dtype and bool(torch.isfinite(Y).all()),
                'Finite rangefinder Y')
        return Y


def build(torch,np,prepared,attestation,workspace):
    """Production bridge only: CPU FP32/rank384 and a caller-owned workspace."""
    _attested(prepared,np,attestation)
    require(workspace._bridge_workspace_sha256==prepared['workspace_source_sha256']
            and not workspace.closed and not workspace.busy and workspace.np is np,
            'Actual open source-qualified extended workspace')
    reference=weakref.ref(workspace)
    def forward_product(A,B):
        owner=reference()
        require(owner is not None and not owner.closed,'Caller workspace survives every forward/recompute')
        # CPU detach is confined to this explicitly differentiated boundary.
        # NumPy receives the SAME strides/values, and never writes A or B.
        result=owner.product(A.detach().numpy(),B.detach().numpy())
        # result is a new owned array, never reusable scratch. Torch owns its
        # storage from here; caller may safely reuse scratch for the next mm.
        return torch.from_numpy(result)
    Product=_mm_class(torch,torch.float32,forward_product)
    return _Rangefinder(torch,prepared,Product.apply,torch.float32,384,False)


def build_mathematical(torch,prepared):
    """Separate CPU FP64/rank3 branch for the fixed independent tiny proof.

    No NumPy/backend/workspace and no implicit precision or device fallback.
    The same saved-operands VJP and authenticated whole-Y formula are used.
    """
    Product=_mm_class(torch,torch.float64,lambda A,B:A@B)
    return _Rangefinder(torch,prepared,Product.apply,torch.float64,3,True)
