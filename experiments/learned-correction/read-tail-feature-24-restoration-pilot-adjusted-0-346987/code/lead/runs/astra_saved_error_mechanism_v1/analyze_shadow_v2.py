from pathlib import Path
import json,time,resource,os
import numpy as np
D=Path(__file__).resolve().parent;H=D/'run_train16_v1';C=H/'native_16_capture_shadow'
assert json.loads((H/'COMPLETE_v1.json').read_text())['all16_byte_identical']
assert json.loads((H/'RETIREMENT_v1.json').read_text())['all_PID_PPID_PGID_absent']
cfg=json.loads((D/'CAPTURE_MANIFEST_v2.json').read_text());W=np.load(cfg['network']['weights_path'],mmap_mode='r')
assert all(os.environ[k]=='1'for k in cfg['thread_variables'])
load=lambda l,s:{k:v for k,v in np.load(C/f'L{l:02d}_{s}.npz').items()}
start=time.monotonic();results=[]
for l in(4,5,6):
 r=load(l,'read');p=load(l,'prepair');z=load(l,'postpair');x=load(l,'features');g=load(l,'gnn');n=1024
 rows=r['rows'];cols=r['cols'];c15=int(r['c15']);tailcols=cols[c15:]
 exact=(r['left'].astype(float)*r['lam'].astype(float))@r['right'][c15:].astype(float).T
 residual=exact-r['answer'][:,c15:].astype(float)
 # Production reinstates the exact diagonal after all read forms.
 for i,row in enumerate(rows):residual[i,tailcols==row]=0
 E=np.zeros((n,n));E[np.ix_(rows,tailcols)]=residual
 v=np.diag(p['Cpre']);s=np.sqrt(v);a=p['mu']/s;pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi);gate=z['gate'];mg=p['mu']*gate+s*pdf
 rho=p['Cpre']/np.outer(s,s);np.fill_diagonal(rho,0)
 A=np.array([mg/s,gate,pdf,-a*pdf,(a*a-1)*pdf]);B=np.array([(a*a+1)*gate+a*pdf,2*mg/s,2*gate,2*pdf,-2*a*pdf])
 def differential(E):
  t=E/np.outer(v,s)
  def pair(F,G,scale):
   T1=F[2,:,None]*G[1,None,:]+rho*F[3,:,None]*G[2,None,:]
   T2=F[1,:,None]*G[2,None,:]+rho*F[2,:,None]*G[3,None,:]
   return .5*scale*(t*T1+t.T*T2)
  dc=pair(A,A,np.outer(s,s));dk21=pair(B,A,np.outer(v,s))-2*z['m'][:,None]*dc
  dpk21=dk21+2*z['m'][:,None]*dc;dpk22=pair(B,B,np.outer(v,v))
  dk22=dpk22-2*z['m'][:,None]*dpk21.T-2*z['m'][None,:]*dpk21+4*np.outer(z['m'],z['m'])*dc-4*z['C']*dc-2*dc*dc
  np.fill_diagonal(dc,0);np.fill_diagonal(dk21,0);np.fill_diagonal(dk22,0)
  cov=z['C'];den=np.trace(cov)**2+2*np.sum(cov*cov);dennew=np.trace(cov)**2+2*np.sum((cov+dc)**2)
  dr=(z['radial']*den+3*np.sum(dk22))/dennew-z['radial']
  w=W[l+1].astype(float);dv=np.sum(w*(dc@w),axis=0)
  return dc,dk21,dr,dv
 dc,dt,dr,dv=differential(E)
 # Counterfactual cheap residual moments. These bases are fixed analytic ones,
 # not fitted against truth: preserve exact tail row sum, or pdf/gate moment.
 probes=[]
 for name,basis in [('constant',np.ones(len(tailcols))),('gate',gate[tailcols]),('pdf',pdf[tailcols])]:
  b=basis/np.linalg.norm(basis);approx=np.outer(residual@b,b);EE=np.zeros((n,n));EE[np.ix_(rows,tailcols)]=approx;np.fill_diagonal(EE,0)
  dd,tt,rr,vv=differential(EE)
  probes.append(dict(name=name,read_energy_recovered=1-np.sum((residual-approx)**2)/np.sum(residual**2),covariance_energy_recovered=1-np.sum((dc-dd)**2)/np.sum(dc**2),next_variance_energy_recovered=1-np.sum((dv-vv)**2)/np.sum(dv**2),target_energy_recovered=1-np.sum((dt-tt)**2)/np.sum(dt**2)))
 # Direct head inputs affected by the changed K21 plane / covariance.
 K=p['K21']/np.outer(v,s);KK=(p['K21']+E)/np.outer(v,s)
 feature_deltas={22:np.linalg.norm(KK,axis=1)-np.linalg.norm(K,axis=1),23:np.linalg.norm(KK,axis=0)-np.linalg.norm(K,axis=0),24:np.sum(E/np.outer(v,s),axis=1)/32,25:np.sum(E/np.outer(v,s),axis=0)/32}
 stdpost=np.sqrt(np.maximum(np.diag(z['C']),1e-12));pr=np.clip(x['C']/np.outer(stdpost,stdpost),-1,1);pr2=np.clip((x['C']+dc)/np.outer(stdpost,stdpost),-1,1)
 feature_deltas[27]=np.sqrt(np.mean(pr2**2,axis=1))-np.sqrt(np.mean(pr**2,axis=1));feature_deltas[28]=np.mean(pr2-pr,axis=1)
 feature_stats={str(k):dict(rms=float(np.sqrt(np.mean(d**2))),relative_to_feature_sd=float(np.std(d)/max(np.std(x['features30'][:,k]),1e-20)))for k,d in feature_deltas.items()}
 next_mean=None
 if l<6:
  pnext=load(l+1,'prepair');vn=np.diag(pnext['Cpre']);sn=np.sqrt(vn);an=pnext['mu']/sn;dn=np.exp(-an*an/2)/(np.sqrt(2*np.pi)*2*sn);dm=dn*dv
  next_mean=float(np.sqrt(np.mean(dm**2)))
 values=dict(E=E,dc=dc,dtarget=dt,next_dv=dv)
 for k,val in feature_deltas.items():values[f'dfeature{k}']=val
 dest=D/f'L{l:02d}_PAIR_DIFFERENTIAL_v1.npz';assert not dest.exists();np.savez(dest,**values)
 results.append(dict(layer=l,read_shape=list(residual.shape),residual_rms=float(np.sqrt(np.mean(residual**2))),covariance_delta_rms=float(np.sqrt(np.mean(dc**2))),target_delta_rms=float(np.sqrt(np.mean(dt**2))),radial_delta=float(dr),radial_current=float(z['radial']),next_variance_delta_rms=float(np.sqrt(np.mean(dv**2))),next_mean_variance_only_rms=next_mean,cheap_row_moment_probes=probes,feature_deltas=feature_stats,prepair_diagonal_change=0,pair_mean_direct_change=0,post_target_diagonal_change=0))
 del r,p,z,x,g,E,dc,dt,values
result=dict(layers=results,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exact same-input linear K21 pair-map differential in float64 mathematical arithmetic, no native shadow rollout. Production rounding and downstream learned/cumulant feedback excluded. Three fixed analytic moment probes; no trained coefficients/truth selected.')
(D/'PAIR_DIFFERENTIAL_v1.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
