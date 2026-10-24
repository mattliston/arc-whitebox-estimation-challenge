"""Shape-only execution through actual current flopscope tariffs; no numerical kernels.
All _call_numpy backend calls are replaced with metadata-only zero-stride arrays.
Only declared shape operations are supported; every unknown backend call fails.
This is a private, inactive probe, never imported by a candidate or live lane.
"""
import collections,hashlib,json,math,os,resource,sys,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget

HERE=Path(__file__).resolve().parent
O=HERE.parents[2]
SOURCE=O/'engineer/a62_modules/s49/s56k1tw_xs/candidate.py'
SOURCE_SHA='a991142325cfa0941a527220183281bd6d898a82bbe0a1aeccce72eab8613280'
CALLS=collections.Counter();PAYLOAD=[0];MAX_RSS=[0]

def virtual(shape,dtype=np.float32):
    shape=tuple(shape) if not isinstance(shape,int) else (shape,)
    scalar=np.zeros((),dtype=dtype)
    PAYLOAD[0]+=scalar.nbytes
    return np.lib.stride_tricks.as_strided(scalar,shape=shape,strides=(0,)*len(shape),writeable=True)

def backend(fn,*args,**kwargs):
    name=getattr(fn,'__name__',repr(fn));CALLS[name]+=1
    if name in ('empty','zeros','ones','full'):
        return virtual(args[0],kwargs.get('dtype',np.float64))
    if name in ('empty_like','zeros_like','ones_like'):
        return virtual(kwargs.get('shape',args[0].shape),kwargs.get('dtype') or args[0].dtype)
    if name=='copyto':return None
    if name in ('reshape','transpose','swapaxes','moveaxis','squeeze','expand_dims','broadcast_to','unstack','split','array_split'):
        return fn(*args,**kwargs)  # zero-stride views only, no numerical operations
    if name in ('array','asarray','asanyarray'):
        if hasattr(args[0],'shape'):
            return virtual(args[0].shape,kwargs.get('dtype') or args[0].dtype)
        small=np.asarray(args[0],dtype=kwargs.get('dtype'))
        assert small.nbytes<=1000000
        return small
    if name in ('copy','astype'):
        return virtual(args[0].shape,kwargs.get('dtype') or (args[1] if len(args)>1 else args[0].dtype))
    if name in ('concatenate','stack'):
        arrays=args[0];axis=kwargs.get('axis',0);sh=list(arrays[0].shape)
        if name=='concatenate':sh[axis]=sum(a.shape[axis] for a in arrays)
        else:sh.insert(axis if axis>=0 else len(sh)+1+axis,len(arrays))
        return virtual(sh,kwargs.get('dtype') or np.result_type(*[a.dtype for a in arrays]))
    if name=='matmul':
        a,b=args[:2];shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2])+(a.shape[-2],b.shape[-1])
        out=kwargs.get('out');return out if out is not None else virtual(shape,np.result_type(a.dtype,b.dtype))
    if isinstance(fn,np.ufunc):
        out=kwargs.get('out')
        if out is not None:return out[0] if isinstance(out,tuple) else out
        shape=np.broadcast_shapes(*[np.shape(a) for a in args])
        return virtual(shape,np.result_type(*args))
    if name in ('getitem','__getitem__'):
        return fn(*args,**kwargs)
    raise RuntimeError('Unsupported metadata backend: '+name)


def load():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
    original=_budget._call_numpy
    patched=[]
    # All public wrappers are imported before patching. The same actual charging
    # code runs; only the numerical backend result is replaced by shape metadata.
    for name,module in list(sys.modules.items()):
        if name.startswith('flopscope') and getattr(module,'_call_numpy',None) is original:
            module._call_numpy=backend;patched.append(name)
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    module=types.ModuleType('shape_candidate');module.__file__=str(SOURCE);sys.modules[module.__name__]=module
    with flops.BudgetContext(flop_budget=2**60,quiet=True):
        source=SOURCE.read_text()
        unused={'_GNN_LITERAL','_FINAL_LITERAL','_BDNH_A_LITERAL','_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_HC_U'}
        lines=[]
        for line in source.splitlines():
            name=line.split('=',1)[0].strip()
            lines.append(name+' = None  # private shape probe: unused learned literal' if name in unused else line)
        exec(compile('\n'.join(lines),str(SOURCE),'exec'),module.__dict__)
        assert all(module.__dict__[key] is None for key in unused), 'A learned literal was used during metadata import'
    return module,patched


def main():
    module,patched=load()
    # Bind only the original shared arena views; no full prebind list or model call.
    with flops.BudgetContext(flop_budget=2**60,quiet=True):
        arena=fnp.empty((module._PACKED_ELEMENTS,),dtype=fnp.float32)
        module._SCRATCH=[arena,module._cp3_bind_arena_views(arena,module._PACKED_PUSH),module._cp3_bind_arena_views(arena,module._PACKED_READ),None,None,None,None]
        module._bind_push6(arena)
    m,k,n=map(int,sys.argv[1:4]);results=[]
    with flops.BudgetContext(flop_budget=2**60,quiet=True):
        a=fnp.empty((m,k),dtype=fnp.float32);b=fnp.empty((k,n),dtype=fnp.float32)
    for phase in ('cold','repeat'):
        with flops.BudgetContext(flop_budget=2**60,quiet=True) as ctx:
            z=module._mm(a,b)
        results.append(dict(phase=phase,bill=ctx.flops_used,shape=list(z.shape)))
    print(json.dumps(dict(source_sha256=SOURCE_SHA,results=results,numerical_payload_bytes=PAYLOAD[0],max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,backend_calls=CALLS,patched_modules=len(patched)),indent=2))
if __name__=='__main__':main()
