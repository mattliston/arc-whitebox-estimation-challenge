from pathlib import Path
import json,zipfile,time,resource,os
import numpy as np
import flopscope as fs
import flopscope.numpy as f
D=Path(__file__).resolve().parent;R=D.parent;C=R/'astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
assert os.getpriority(os.PRIO_PROCESS,0)==10
assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
def shape(path,key):
 with zipfile.ZipFile(path)as z,z.open(key+'.npy')as fp:
  version=np.lib.format.read_magic(fp)
  return (np.lib.format.read_array_header_1_0(fp)if version==(1,0)else np.lib.format.read_array_header_2_0(fp))[0]
start=time.monotonic();rows=[]
for l in range(2,15):
 p=C/f'L{l:02d}.npz';m,k=shape(p,'left_young');q=shape(p,'Q')[1]
 with np.load(p)as z:
  rr=z['rows'];cc=z['cols'][int(z['c15']):];pc=z['cols'][:int(z['c15'])]
 t=len(cc);pfx=len(pc);_,ir,ic=np.intersect1d(rr,pc,return_indices=True)
 with fs.BudgetContext(flop_budget=2**50,quiet=True):
  A=f.asarray(np.zeros((m,pfx),dtype='f'));Q=f.asarray(np.zeros((m,q),dtype='f'));S=f.asarray(np.zeros((q,k),dtype='f'));B=f.asarray(np.zeros((t,k),dtype='f'));T=f.asarray(np.zeros((m,t),dtype='f'));G=f.asarray(np.ones((t,2),dtype='f'));sig=f.asarray(np.ones(t,dtype='f'));var=f.asarray(np.ones(m,dtype='f'));I=f.asarray(ir);J=f.asarray(ic);K=f.asarray(np.zeros((1024,1024),dtype='f'));RR=f.asarray(rr);CC=f.asarray(cc);F=f.asarray(np.zeros((1024,30),dtype='f'));VV=f.asarray(np.ones(1024,dtype='d'));PC=f.asarray(pc)
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as b:
  variance=VV.astype(f.float32)
  residual=A-Q@(Q.T@A)
  f.put(residual,I*pfx+J,f.float32(0))
  scale=f.sqrt(variance[PC])
  normalized=residual/(variance[RR,None]*scale[None,:])
  energy=f.einsum('ij,ij->i',normalized,normalized,optimize=False)
  ratio=f.sum(1/variance[CC])/f.sum(1/variance[PC])
  energy=energy*ratio
  old=F[RR,22];new=f.sqrt(f.maximum(old*old+energy,f.float32(0)))
  f.put(F,RR*30+22,new)
 rows.append(dict(layer=l,m=m,k=k,tail=t,q=q,diagonal=len(ir),flops=int(b.flops_used),ops=len(b._op_log)))
 del A,Q,S,B,T,G,sig,var,I,J,K,RR,CC,F,VV,PC,variance,residual,scale,normalized,energy,ratio,old,new
bill=sum(r['flops']for r in rows);out=dict(status='PREFIX_ENERGY_ACTUAL_FNP_TARIFF',total_flops=bill,bill_percent=100*bill/242136236203,total_ops=sum(r['ops']for r in rows),scope='All current affected hidden layers L2-14, saved TRAIN16 shapes; original paid left/finalQ/scores/weighted-tail/young-tail/K21 assumed reused. Includes global diagonal restoration and feature22 update; no owner buffer/setup overhead.',rows=rows,models=0,captures=0,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'PREFIX_ENERGY_TARIFF_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}))
