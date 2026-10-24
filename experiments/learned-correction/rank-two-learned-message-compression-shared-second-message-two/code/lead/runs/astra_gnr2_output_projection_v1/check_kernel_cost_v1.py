from pathlib import Path
import ast,json,hashlib,os
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;old=D.parent/'astra_gnn_nonnegative_rank2_v1/fragment_gnr2_UNVALIDATED_v1.py'
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
node=next(x for x in ast.parse(old.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='_gnr2_contract')
e={'fnp':fnp};exec(compile(ast.Module(body=[node],type_ignores=[]),str(old),'exec'),e)
exec(compile((D/'kernel_HELD_v1.py').read_text(),str(D/'kernel_HELD_v1.py'),'exec'),e)
rng=np.random.default_rng(20261014);rows=[]
for n in(64,128,192):
 W=rng.normal(0,np.sqrt(2/n),(n,n)).astype(np.float32);src=rng.normal(size=(4,n,16)).astype(np.float32)
 Ts=np.stack([W**2,W**2,W**4,W**6]).transpose(0,2,1).copy();src[1:]=src[1:]**2
 with fs.BudgetContext(flop_budget=2**34,quiet=True):T=fnp.asarray(Ts);X=fnp.asarray(src)
 before=(np.asarray(T).tobytes(),np.asarray(X).tobytes());bills=[];outputs=[]
 for f in(lambda:e['_gnr2_contract'](T,X),lambda:e['output_projection_contract'](T,X),lambda:fnp.matmul(T,X)):
  with fs.BudgetContext(flop_budget=2**34,quiet=True)as b:y=f()
  bills.append(int(b.summary_dict()['flops_used']));outputs.append(np.array(y))
 assert before==(np.asarray(T).tobytes(),np.asarray(X).tobytes())and all(np.isfinite(y).all()for y in outputs)
 ratios=[]
 for k in range(4):
  A=Ts[k].astype(np.float64);B=src[k].astype(np.float64);mu=B.mean(0);C=B-mu;_,V=np.linalg.eigh(C.T@C);P=A@(C@V[:,-2:]);Q,_=np.linalg.qr(P);Y=A@C
  previous=P@V[:,-2:].T;optimal=Q@(Q.T@Y)
  olderr=np.linalg.norm(Y-previous);newerr=np.linalg.norm(Y-optimal)
  assert newerr<=olderr*(1+1e-12)
  ratios.append(float(np.linalg.norm(outputs[2][k]-outputs[1][k])/np.linalg.norm(outputs[2][k]-outputs[0][k])))
 rows.append(dict(n=n,gnr2_bill=bills[0],output_projection_bill=bills[1],exact_generic_four_maps_bill=bills[2],increment=bills[1]-bills[0],real_projection_error_not_increased=True,synthetic_f32_error_ratios=ratios,finite=True,inputs_immutable=True))
co=np.rint(np.linalg.solve([[x['n']**2,x['n'],1]for x in rows],[x['increment']for x in rows])).astype(np.int64)
assert all(x['increment']==int(co[0])*x['n']**2+int(co[1])*x['n']+int(co[2])for x in rows)
result=dict(status='KERNEL_ONLY_SYNTHETIC_COST_NO_CANDIDATE_OR_MODELS',kernel_sha256=hashlib.sha256((D/'kernel_HELD_v1.py').read_bytes()).hexdigest(),rows=rows,increment_polynomial=co.tolist(),GNR2_reference_saving_15layers_1024=1617680640,projected_remaining_saving=1617680640-15*(int(co[0])*1024**2+int(co[1])*1024+int(co[2])),no_network_accuracy_claim=True)
with(D/'KERNEL_COST_RESULT_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
