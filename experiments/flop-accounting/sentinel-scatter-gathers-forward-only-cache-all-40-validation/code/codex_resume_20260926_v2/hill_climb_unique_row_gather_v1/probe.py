"""Root-only paid source-bound small fixture; no network, targets, estimator import."""
from pathlib import Path
import ast,hashlib,json,os,sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent

def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 assert len(sys.argv)==3
 build=json.loads((D/'build.json').read_text());assert pin(D/'build.json')['sha256']==sys.argv[1]
 for name in ['candidate','parent','primitive','fragment']:assert pin(build[name]['path'])==build[name]
 result=Path(sys.argv[2]);assert not result.exists()
 assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'])
 defs=[n for n in ast.parse(Path(build['candidate']['path']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_unique_rows_via_scatter'];assert len(defs)==1
 env={'fnp':fnp};exec(compile(ast.Module(body=defs,type_ignores=[]),'<actual selected helper>','exec'),env)
 receipts=[]
 with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
  for rows0 in [[5,1,4],[6,0,4,1,5,3,2],[],[2]]:
   for width in [1,16,31]:
    for strided in [False,True]:
     m=7;k=len(rows0);storage=np.arange(m*width*(2 if strided else 1),dtype=np.float32).reshape(m,-1)
     storage.flat[0]=-0.0;storage.flat[2]=np.nextafter(np.float32(0),np.float32(1))
     source=fnp.asarray(storage);source=source[:,::2] if strided else source
     original=np.asarray(source).copy();rows=fnp.asarray(rows0,dtype=fnp.int64)
     before=int(ctx.summary_dict()['flops_used']);old=source[rows];old_cost=int(ctx.summary_dict()['flops_used'])-before
     before=int(ctx.summary_dict()['flops_used']);new=env['_unique_rows_via_scatter'](source,rows);new_cost=int(ctx.summary_dict()['flops_used'])-before
     oa,na=np.asarray(old),np.asarray(new)
     assert oa.shape==na.shape and oa.dtype==na.dtype and oa.tobytes()==na.tobytes()
     assert np.asarray(source).tobytes()==original.tobytes() and na.flags.c_contiguous
     assert not np.shares_memory(na,np.asarray(source))
     scatter=(4*k-m)*width>2*m+6*k
     if scatter:assert new_cost<old_cost
     else:assert new_cost==old_cost
     receipts.append(dict(rows=rows0,width=width,strided_source=strided,scatter=scatter,old_paid_flops=old_cost,new_paid_flops=new_cost,all_bytes_equal=True,input_unmodified=True,output_owned_independently=True))
 result.write_text(json.dumps(dict(passed=True,build=pin(D/'build.json'),candidate=build['candidate'],primitive=pin(__file__),cases=receipts,model_executed=False,targets_read=False),indent=2)+'\n')
if __name__=='__main__':main()
