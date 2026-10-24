from pathlib import Path
import ast,json,hashlib,numpy as np,flopscope as fs,flopscope.numpy as fnp
D=Path(__file__).resolve().parent
source=(D/'check_tiny_tariff_lean_v2.py').read_text();ns0={'__file__':str(D/'check_tiny_tariff_lean_v2.py')};exec(source[:source.index('rng=np.random.default_rng')],ns0)
make=ns0['make'];namespace=ns0['namespace'];cases=[];rng=np.random.default_rng(20261015)
def arena(xp,n,packed):
 if packed:
  nn=4*n*n;q=3*n*n;tail=(2,3,n,n)
  owner=xp.asarray(np.full(72*nn,np.nan,np.float32))
  def view(start,count,shape):return owner[start:start+count].reshape(shape)
  X2=view(0,20*q,(5,2)+tail)
  a=dict(X1=view(0,6*q,(3,)+tail),X2=X2,X3=view(0,16*q,(4,2)+tail),T=view(45*nn,10*q,(5,)+tail),U=view(60*nn,8*q,(4,)+tail),Z=view(12*nn,10*q,(5,)+tail))
 else:
  nn=n*n;owner=xp.asarray(np.full(72*nn,np.nan,np.float32));Z=xp.asarray(np.full((5,3,n,n),np.nan,np.float32))
  def view(start,count,shape):return owner[start:start+count].reshape(shape)
  X2=view(0,30*nn,(5,2,3,n,n))[:,:,1:2]
  a=dict(X1=view(0,9*nn,(3,3,n,n))[:,1:2],X2=X2,X3=view(0,24*nn,(4,2,3,n,n))[:,:,1:2],T=view(45*nn,15*nn,(5,3,n,n))[:,1:2],U=view(60*nn,12*nn,(4,3,n,n))[:,1:2],Z=Z[:,1:2]);tail=(1,n,n)
 a['div']=xp.asarray(np.array([2,4,6,24],np.float32).reshape((4,)+(1,)*len(tail)));a['X2r1']=X2[:,1:2]
 return a,owner
for xp in (np,fnp):
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  for packed in (False,True):
   ns=namespace(xp);n=8;a,own=arena(xp,n,packed);b,ownb=arena(xp,n,packed)
   for repeat in range(2):
    args,_,qr,qc=make(xp,rng,np.float32,2 if packed else 1,n,packed)
    a['RR1']=b['RR1']=args[4]['P6'][:1][None]
    before=[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
    out=ns['_astra_hf_grouped'](*args[:8],a,lambda x:x,qr,qc)
    ref=ns['_frozen_hf_grouped'](*args[:8],b,lambda x:x,qr,qc)
    assert np.asarray(out).tobytes()==np.asarray(ref).tobytes(),(xp.__name__,packed,repeat)
    assert np.isfinite(np.asarray(out)).all()
    assert before==[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
    cases.append(dict(backend=xp.__name__,packed=packed,repeat=repeat,exact_frozen_output_bytes=True,nan_initialized_real_arena_alias=True))
r=dict(status='PASS',cases=cases,candidate_sha256=hashlib.sha256((D/'candidate_hermite_group_lean_QR_HELD_v2.py').read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256((D/'fragment_hermite_group_lean_HELD_v2.py').read_bytes()).hexdigest(),models=0)
p=D/'REAL_ARENA_ALIAS_CHECK_v2.json';p.open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
