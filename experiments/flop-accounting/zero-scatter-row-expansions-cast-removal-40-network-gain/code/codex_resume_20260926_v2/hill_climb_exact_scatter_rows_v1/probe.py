"""Root-only small paid source-bound scatter fixture. No network or targets."""
from pathlib import Path
import ast, hashlib, json, os, sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent

def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 assert len(sys.argv)==3
 build=json.loads((D/'build.json').read_text());assert pin(D/'build.json')['sha256']==sys.argv[1]
 for key in ['candidate','parent','primitive','fragment']:
  assert pin(build[key]['path'])==build[key]
 result=Path(sys.argv[2]);assert not result.exists()
 assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'])
 tree=ast.parse(Path(build['candidate']['path']).read_text());f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_exact_expand_rows'];assert len(f)==1
 env={'fnp':fnp};exec(compile(ast.Module(body=f,type_ignores=[]),'<actual candidate helper>','exec'),env)
 rows_cases=[(5,[3,0,4]),(5,[]),(5,[4,0,3,1,2]),(1,[0])]
 receipts=[]
 with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
  for m,rows0 in rows_cases:
   for width in [1,4,7]:
    k=len(rows0);x=np.arange(k*width,dtype=np.float32).reshape(k,width)
    if x.size:x.flat[0]=-0.0
    if x.size>1:x.flat[1]=np.nextafter(np.float32(0),np.float32(1))
    rows=fnp.asarray(rows0,dtype=fnp.int64);values=fnp.asarray(x)
    inverse=fnp.full(m,k,dtype=fnp.int64);fnp.put(inverse,rows,fnp.arange(k,dtype=fnp.int64))
    old=fnp.concatenate([values,fnp.zeros((1,width),dtype=fnp.float32)],axis=0)[inverse]
    before=int(ctx.summary_dict()['flops_used']);new=env['_exact_expand_rows'](values,rows,m);cost=int(ctx.summary_dict()['flops_used'])-before
    oa,na=np.asarray(old),np.asarray(new)
    assert oa.dtype==na.dtype and oa.shape==na.shape and oa.tobytes()==na.tobytes()
    assert na.flags.c_contiguous and not np.shares_memory(na,x)
    receipts.append(dict(rows=m,kept=k,width=width,all_bytes_equal=True,paid_helper_flops=cost))
  summary=ctx.summary_dict()
 result.write_text(json.dumps(dict(passed=True,build=pin(D/'build.json'),candidate=build['candidate'],primitive=pin(__file__),cases=receipts,paid_summary=summary,model_executed=False,targets_read=False),indent=2,default=str)+'\n')
if __name__=='__main__':main()
