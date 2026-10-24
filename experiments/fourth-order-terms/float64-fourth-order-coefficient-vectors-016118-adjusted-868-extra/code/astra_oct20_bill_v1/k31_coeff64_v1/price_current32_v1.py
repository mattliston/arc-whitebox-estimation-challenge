"""Official-tariff K31 local block prices on exact777a observed32 rosters.

Only extracted local-block ASTs execute. The backend returns zero-stride shape
metadata; no estimator import/setup/predict, model data, or truth is accessed.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse,ast,collections,copy,hashlib,importlib.util,json,resource,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
SOURCE=R/'astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
CENSUS=R/'astra_oct20_vm_frontier_v1/science32_msg8_loaderfix_parallel5_v1'
OFFICIAL=R/'astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3'
SHA='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rss_guard():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1_000_000_000:os._exit(97)
        time.sleep(.1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
    threading.Thread(target=rss_guard,daemon=True).start();assert sha(SOURCE)==SHA
    sys.path.insert(0,str(OFFICIAL));import numpy as np;import flopscope as fs;import flopscope.numpy as fnp
    from flopscope import _budget
    assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
    helper=R/'astra_nbmerge_s56_v1/shape_tariff_probe_v2.py';spec=importlib.util.spec_from_file_location('k31_shape_backend',helper);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    original=p.backend
    def backend(fn,*args,**kwargs):
        name=getattr(fn,'__name__','')
        if name in ('sum','prod'):
            p.CALLS[name]+=1;x=args[0];axis=kwargs.get('axis',args[1] if len(args)>1 else None);axes=tuple(range(x.ndim)) if axis is None else (axis,) if isinstance(axis,int) else tuple(axis);axes=tuple(k%x.ndim for k in axes)
            shape=tuple(1 if k in axes else z for k,z in enumerate(x.shape)) if kwargs.get('keepdims',False) else tuple(z for k,z in enumerate(x.shape) if k not in axes)
            return p.virtual(shape,kwargs.get('dtype') or x.dtype)
        if name=='einsum':
            p.CALLS[name]+=1;assert args[0]=='ik,ik->k' and args[1].shape==args[2].shape
            return p.virtual((args[1].shape[1],),np.result_type(args[1].dtype,args[2].dtype))
        if name in ('copy','astype') and hasattr(fn,'__self__'):
            p.CALLS[name]+=1;x=fn.__self__;dt=(args[0] if args else kwargs.get('dtype',x.dtype)) if name=='astype' else x.dtype
            return p.virtual(x.shape,dt).view(type(x))
        if isinstance(fn,np.ufunc) and name in ('greater','less','greater_equal','less_equal','equal','not_equal'):
            p.CALLS[name]+=1;return p.virtual(np.broadcast_shapes(*[np.shape(x) for x in args]),np.bool_)
        return original(fn,*args,**kwargs)
    old=_budget._call_numpy
    for name,module in list(sys.modules.items()):
        if name.startswith('flopscope') and getattr(module,'_call_numpy',None) is old:module._call_numpy=backend
    source=SOURCE.read_text();tree=ast.parse(source);env=dict(fnp=fnp)
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except(ValueError,TypeError,SyntaxError):continue
            for target in node.targets:
                if isinstance(target,ast.Name) and (target.id.startswith('_NP_') or target.id=='_S19_LAM'):env[target.id]=value
        if isinstance(node,ast.FunctionDef) and node.name in ('_np_ages','_np_key','_np_tables','_np_const'):
            if node.name=='_np_ages' and node.name in env:continue
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<table AST>','exec'),env)
    parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_o20_npx_copy_exact_left_v1')
    tail=HERE/'tail_msg8_k31coeff64_v1.py';tt=ast.parse(tail.read_text())
    for node in tt.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_o21_k31_common_matrix64','_o21_k31_common_polynomial32'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<candidate helpers>','exec'),env)
    candidate=next(n for n in tt.body if isinstance(n,ast.FunctionDef) and n.name=='_o21_k31c_left_v1')
    def block(node,name):
        young=next(n for n in node.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='young')
        wrapper=ast.parse('def '+name+'(pp2,pp,rows,young,y0,y1,C,S,lam,hold,sizes,blocks,n):\n    pass\n').body[0]
        wrapper.body=copy.deepcopy(young.body)+[ast.Return(ast.Subscript(ast.Name('local',ast.Load()),ast.Constant(2),ast.Load()))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'<exact young block>','exec'),env)
        return env[name]
    oldblock,newblock=block(parent,'oldblock'),block(candidate,'newblock')
    env['_O21_K31_COEFF64_CONST']=None;env['_O21_K31C_STATS']=dict(calls=0,row_pairs=0,pairs=0);env['_O20_NPX_COPY_EXACT_STATS']=dict(delta_blocks=0)
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        env['_NP_CACHE']={};env['_NP_TABLES']=env['_np_tables'](1024);C=env['_np_const'](1024)
    records=[];pins=[];shape_cache={}
    for path in sorted(CENSUS.glob('candidate_part*/rows.jsonl')):
        assert json.loads((path.parent/'manifest.json').read_text())['sha256']==SHA;pins.append(dict(path=str(path),sha256=sha(path)))
        previous_shapes=[]
        for line in path.read_text().splitlines():
            row=json.loads(line);all_shapes=row['jun_stats']['shapes'];assert all_shapes[:len(previous_shapes)]==previous_shapes and row['jun_stats']['calls']==len(all_shapes)
            births=[int(s[3]) for s in all_shapes[len(previous_shapes):]];assert len(births)==15;previous_shapes=all_shapes
            # These fifteen newborn pushes are layers1..15, hence q is K_0..K_14.
            # select(min(g,1-g),f) and select(where(g>=f,min(g,1-g),-1),f)
            # have identical qualifying cardinality, and share32 rounding.
            # Thus current response count at l equals birth count K_l.
            layers=[]
            for l in range(1,15):
                widths=tuple(births[max(0,l-3):l]);kept=births[l];ages=env['_NP_K3L_BY_LAYER'].get(l,env['_NP_K3L']);strengths=tuple(ages[l-b-1] for b in range(max(0,l-3),l));key=(kept,widths,strengths)
                if key not in shape_cache:
                    sizes=[w for w in widths for _ in range(2)];W=sum(sizes);young=[(b,w,s) for b,(w,s) in enumerate(zip(widths,strengths))]
                    with fs.BudgetContext(flop_budget=2**60,quiet=True):
                        pp=fnp.empty((1024,W),dtype=fnp.float32);pp2=fnp.empty(pp.shape,dtype=fnp.float32);rows=fnp.empty((kept,W),dtype=fnp.float32);lam=fnp.empty((W,),dtype=fnp.float32);parts=fnp.split(rows,list(np.cumsum(sizes))[:-1],axis=1);S={'k4e':{b:fnp.empty((4,2*w),dtype=fnp.float32) for b,w,_ in young}}
                    args=(pp2,pp,rows,young,0,W,C,S,lam,lambda x:x,sizes,parts,1024)
                    measurements=[]
                    for fn in (oldblock,newblock,newblock):
                        p.CALLS.clear()
                        with fs.BudgetContext(flop_budget=2**60,quiet=True)as budget:value=fn(*args)
                        assert value.shape==(kept,W)
                        measurements.append(dict(flops=int(budget.flops_used),charged_calls=len(budget._op_log),backend_calls=sum(p.CALLS.values())))
                    shape_cache[key]=dict(rows=kept,pair_widths=list(widths),strengths=list(strengths),old=measurements[0],new_warm=measurements[2],saving_flops=measurements[0]['flops']-measurements[2]['flops'])
                layers.append(dict(layer=l,**shape_cache[key]))
            saved=sum(x['saving_flops'] for x in layers);calls=sum(x['new_warm']['charged_calls']-x['old']['charged_calls'] for x in layers)
            records.append(dict(rank=row['rank'],parent_flops=row['flops'],birth_counts=births,layers=layers,saving_flops=saved,bill_delta_pct=-100*saved/row['flops'],charged_call_delta=calls))
    assert len(records)==32
    report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='METADATA_ONLY_OFFICIAL_LOCAL_BLOCK_PRICE',reference_submission=337422,source_sha256=SHA,tail_sha256=sha(tail),script_sha256=sha(__file__),helper_sha256=sha(helper),official=fs.__version__,census_pins=pins,records=sorted(records,key=lambda r:r['rank']),unique_shapes=len(shape_cache),mean_bill_delta_pct=sum(x['bill_delta_pct'] for x in records)/32,min_bill_delta_pct=min(x['bill_delta_pct'] for x in records),max_bill_delta_pct=max(x['bill_delta_pct'] for x in records),mean_saving_flops=sum(x['saving_flops'] for x in records)/32,charged_call_delta_range=[min(x['charged_call_delta'] for x in records),max(x['charged_call_delta'] for x in records)],extra_first_call_constants_flops=154,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,caveat='Fixed current rosters and all nonlocal work unchanged. Actual candidate rounding can change subsequent rosters/kernel routing; native whole-network bill and timing remain authoritative. No runtime or accuracy claim.')
    with a.out.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k]for k in ('status','unique_shapes','mean_bill_delta_pct','min_bill_delta_pct','max_bill_delta_pct','mean_saving_flops','charged_call_delta_range','peak_rss_bytes')},indent=2))
if __name__=='__main__':main()
