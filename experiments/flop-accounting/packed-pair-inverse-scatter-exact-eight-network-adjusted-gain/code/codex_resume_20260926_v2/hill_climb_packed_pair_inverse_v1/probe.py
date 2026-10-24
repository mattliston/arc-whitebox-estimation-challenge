"""Root-only paid source-bound primitive, no networks/targets/estimator import."""
from pathlib import Path
import ast,hashlib,json,os,sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
class Hold(list):
    def __call__(self,x):self.append(x);return x
def main():
    assert len(sys.argv)==3
    build=json.loads((D/'build.json').read_text());assert pin(D/'build.json')['sha256']==sys.argv[1]
    for k in ('candidate','parent','fragment','primitive'):assert pin(build[k]['path'])==build[k]
    result=Path(sys.argv[2]);assert not result.exists()
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    names={'_psp_indices','_ppi_make_inverse','_ppi_take','_ppi_copy_p16','_ppi_get','_ppi_predict'}
    defs=[n for n in ast.parse(Path(build['candidate']['path']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(defs)==len(names)
    env=dict(fnp=fnp,_PSP_INDEX={},_E2Hold=Hold,_PPI_RUN=[None]);exec(compile(ast.Module(body=defs,type_ignores=[]),'<actual packed helpers>','exec'),env)
    receipts=[]
    with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
        for n,block in ((16,2),(16,4)):
            idx=env['_psp_indices'](n,block);count=idx['count'];q=count*block*block
            start=int(ctx.summary_dict()['flops_used']);inverse,it=env['_ppi_make_inverse'](idx,n,block)
            cold=int(ctx.summary_dict()['flops_used'])-start;assert cold==n*n+3*q
            iu=np.asarray(inverse);upper=np.asarray(idx['upper']).reshape(-1)
            assert len(set(map(int,upper)))==q
            assert np.array_equal(iu.reshape(-1)[upper],np.arange(q))
            assert np.count_nonzero(iu==q)==n*n-q
            for strided in (False,True):
                for repetition in range(3):
                    host=(np.arange(n*n*(2 if strided else 1),dtype=np.float32).reshape(n,-1)+repetition*10000)
                    host.flat[0]=-0.;host.flat[2]=np.nextafter(np.float32(0),np.float32(1));host.flat[-1]=-13.
                    source=fnp.asarray(host);source=source[:,::2] if strided else source
                    saved=np.asarray(source).copy()
                    for transpose,index in ((False,inverse),(True,it)):
                        old=fnp.take(source,idx['transpose'] if transpose else idx['upper'])
                        start=int(ctx.summary_dict()['flops_used']);new=env['_ppi_take'](source,index,count,block)
                        paid=int(ctx.summary_dict()['flops_used'])-start;assert paid==n*n
                        oa,na=np.asarray(old),np.asarray(new)
                        assert oa.shape==na.shape and oa.dtype==na.dtype and oa.tobytes()==na.tobytes()
                        assert na.flags.c_contiguous and not np.shares_memory(na,np.asarray(source))
                        assert np.asarray(source).tobytes()==saved.tobytes()
                        receipts.append(dict(n=n,block=block,strided=strided,repetition=repetition,transpose=transpose,cold_F=cold,scatter_F=paid,bytes_exact=True,unique_retained=True,discarded_only_duplicates=True,source_unchanged=True))
                    planes=fnp.stack((source,source,source));out=fnp.empty((3,count,block,block),dtype=fnp.float32);hold=Hold()
                    start=int(ctx.summary_dict()['flops_used']);env['_ppi_copy_p16'](planes,inverse,out,count,block,hold)
                    paid=int(ctx.summary_dict()['flops_used'])-start;assert paid==3*n*n+3*q
                    expected=fnp.take(fnp.reshape(planes,(3,n*n)),idx['upper'],axis=1)
                    assert np.asarray(out).tobytes()==np.asarray(expected).tobytes()
                    hold.clear()
    # Scalar-only scope checks use the actual wrapper/getter and token inverse factory.
    actual_factory=env['_ppi_make_inverse'];created=[]
    def factory(idx,n,block):created.append((idx,n,block));return object(),object()
    env['_ppi_make_inverse']=factory;st=object();idx=object();seen=[]
    def parent(self,mlp,budget):
        a=env['_ppi_get'](st,idx,1024,128);b=env['_ppi_get'](st,idx,1024,128);assert a==b
        seen.append(env['_PPI_RUN'][0]);return 'ok'
    env['_PPI_PARENT_PREDICT']=parent
    assert env['_ppi_predict'](None,None,None)=='ok' and len(created)==1 and env['_PPI_RUN'][0] is None and seen[-1]=={}
    def failing(self,mlp,budget):
        env['_ppi_get'](st,idx,1024,128);seen.append(env['_PPI_RUN'][0]);env['_ppi_get'](object(),idx,1024,128)
    env['_PPI_PARENT_PREDICT']=failing
    try:env['_ppi_predict'](None,None,None);raise AssertionError('owner mismatch accepted')
    except RuntimeError as e:assert 'owner mismatch' in str(e)
    assert env['_PPI_RUN'][0] is None and seen[-1]=={} and len(created)==2
    env['_ppi_make_inverse']=actual_factory
    result.write_text(json.dumps(dict(passed=True,build=pin(D/'build.json'),candidate=build['candidate'],primitive=pin(__file__),cases=receipts,P16_bytes_exact=True,P16_copy_tariff_exact=True,scope_success_exception_retired=True,model_executed=False,targets_read=False),indent=2)+'\n')
if __name__=='__main__':main()
