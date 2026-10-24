from pathlib import Path
import json,os,time,resource
import numpy as np
assert os.getpriority(os.PRIO_PROCESS,0)==10
DEST=Path(__file__).resolve().parent;D=DEST.parent/'astra_saved_error_mechanism_v1';R=D/'run_replay_train16_v1';C=R/'native_16_capture_shadow'
assert json.loads((R/'RETIREMENT_v1.json').read_text())['all_PID_PPID_PGID_absent']
assert json.loads((R/'COMPLETE_v1.json').read_text())['all16_byte_identical']
cfg=json.loads((D/'CAPTURE_MANIFEST_v3.json').read_text())
assert all(os.environ[k]=='1' for k in cfg['thread_variables'])
load=lambda l,s:{k:v for k,v in np.load(C/f'L{l:02d}_{s}.npz').items()}
truth=np.load(cfg['network']['truth_path'])['mc_post'].astype(float)
parent=np.load(D/'run_train16_v1/native_16_parent_pred.npz')['16'].astype(float)
silu=lambda x:x/(1+np.exp(-x))
start=time.monotonic();results=[];directions={}
for l in (4,5,6):
 g=load(l,'gnn');x=load(l,'features');f=load(l,'fhm');bias=load(l,'hcbias')['bias'];H=g['H'];dx_exact=np.load(D/f'L{l:02d}_PAIR_DIFFERENTIAL_v1.npz')['dfeature22']
 rp=load(l,'read');p=load(l,'prepair');E=np.load(D/f'L{l:02d}_PAIR_DIFFERENTIAL_v1.npz')['E'];v=np.diag(p['Cpre']);sig=np.sqrt(v);rows=rp['rows'];tailcols=rp['cols'][int(rp['c15']):]
 obs=D.parent/'astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'/f'L{l:02d}.npz'
 with np.load(obs)as z:Q=z['Q'].astype(float);AA=z['left_young'].astype(float);RR=z['right_young'][int(z['c15']):].astype(float);lam=z['lambda_young'].astype(float)
 assert np.array_equal(AA,rp['left']) and np.array_equal(RR,rp['right'][int(rp['c15']):])
 resid=(AA-Q@(Q.T@AA))/v[rows,None];B=RR*lam[None,:]/sig[tailcols,None]
 bounds=np.r_[0,np.flatnonzero(np.sign(lam[1:])!=np.sign(lam[:-1]))+1,len(lam)]
 assert len(bounds)%2==1
 energy=np.zeros(len(rows));diagonal=np.zeros(len(rows));_,ir,ic=np.intersect1d(rows,tailcols,return_indices=True)
 for j in range(0,len(bounds)-1,2):
  st,md,ed=bounds[j:j+3];assert md-st==ed-md
  u=resid[:,st:md];w=resid[:,md:ed];b=B[:,st:md];c=B[:,md:ed]
  energy+=(u*u)@np.sum(b*b,axis=0)+(w*w)@np.sum(c*c,axis=0)+2*(u*w)@np.sum(b*c,axis=0)
  diagonal[ir]+=np.sum((u[ir]*b[ic]+w[ir]*c[ic])**2,axis=1)
 energy-=diagonal
 exactenergy=np.sum((E[rows][:,tailcols]/(v[rows,None]*sig[tailcols]))**2,axis=1)
 prior=x['features30'][:,22].astype(float);change=np.zeros(1024);change[rows]=energy;dx=np.sqrt(np.maximum(prior*prior+change,0))-prior
 cosine=float(dx@dx_exact/max(np.linalg.norm(dx)*np.linalg.norm(dx_exact),1e-30));relerr=float(np.linalg.norm(dx-dx_exact)/np.linalg.norm(dx_exact))
 ene_cos=float(energy@exactenergy/(np.linalg.norm(energy)*np.linalg.norm(exactenergy)));ene_rel=float(np.linalg.norm(energy-exactenergy)/np.linalg.norm(exactenergy))
 exself=np.zeros(1024);exself[rows]=exactenergy;exdx=np.sqrt(prior*prior+exself)-prior
 exself_cos=float(exdx@dx_exact/(np.linalg.norm(exdx)*np.linalg.norm(dx_exact)));exself_rel=float(np.linalg.norm(exdx-dx_exact)/np.linalg.norm(dx_exact))
 direct=resid@B.T;direct[ir,ic]=0;target=E[rows][:,tailcols]/(v[rows,None]*sig[tailcols]);contract_error=float(np.linalg.norm(direct-target)/np.linalg.norm(target))
 assert contract_error<1e-3,contract_error
 def replay(h):
  h=silu(h@g['enc0_w'].T+bias)
  h=silu(h@g['enc2_w'].T+g['enc2_b'])
  state=h@g['enc4_w'].T+g['enc4_b']
  delta=(g['sigma'].astype(np.float32)*g['rms'])*(state@g['head']+g['head_b'])*g['gain']
  return state,delta
 state,delta=replay(H)
 assert np.array_equal(state,g['state64']),float(np.max(np.abs(state-g['state64'])))
 assert np.max(np.abs(delta-g['delta']))<1e-10,float(np.max(np.abs(delta-g['delta'])))
 delta=g['delta'].copy()
 expected=np.clip((x['features30'][:,22]-g['centers'][22])/g['scales'][22],-8,8)
 assert np.array_equal(expected,H[:,22])
 trial=H.copy();trial[:,22]=np.clip(((x['features30'][:,22].astype(float)+dx).astype(np.float32)-g['centers'][22])/g['scales'][22],-8,8)
 newstate,newdelta=replay(trial)
 # The ordinary early-layer FHM rule clips the corrected host mean at zero.
 eff=np.maximum(delta,-f['host']);assert np.array_equal(eff,f['effective_delta'])
 deff=np.maximum(newdelta,-f['host'])-eff
 pred=f['host']+eff
 assert np.array_equal(pred.astype(np.float32),parent[l]),float(np.max(np.abs(pred-parent[l])))
 deff=(pred+deff).astype(np.float32).astype(float)-parent[l];pred=parent[l]
 error=pred-truth[l];mse=np.mean(error**2);newmse=np.mean((error+deff)**2)
 dmse=2*np.mean(error*deff)+np.mean(deff**2)
 results.append(dict(layer=l,pair_self_energy_cosine=ene_cos,pair_self_energy_relative_error=ene_rel,exact_E2_feature_cosine=exself_cos,exact_E2_feature_relative_error=exself_rel,feature_displacement_cosine=cosine,relative_feature_displacement_error=relerr,residual_contract_relative_error=contract_error,baseline_state_byte_identical=True,baseline_delta_replay_max_error_lt_1e10=True,baseline_local_mean_byte_identical=True,feature22_rms=float(np.sqrt(np.mean(dx*dx))),normalized_input_change_rms=float(np.sqrt(np.mean((trial[:,22]-H[:,22])**2))),local_mean_change_rms=float(np.sqrt(np.mean(deff**2))),local_state_change_rms=float(np.sqrt(np.mean((newstate-state)**2))),local_mean_error_rms=float(np.sqrt(mse)),local_mse=mse,local_counterfactual_mse=newmse,local_mse_gain_pct=float(100*(1-newmse/mse)),local_error_direction_cos=float(np.mean(error*deff)/np.sqrt(np.mean(error**2)*np.mean(deff**2))),local_mse_derivative_at_zero=float(2*np.mean(error*deff)/mse),clip_count=int(np.sum(newdelta<-f['host']))))
 directions[f'L{l:02d}_dmean']=deff;directions[f'L{l:02d}_dstate']=newstate-state
out=dict(scope='One fixed signed-pair SELF E2-only feature22 restoration including global diagonal exclusion, L4-6 diagnostic same-input frozen GNN. No fitting, no alternative rollouts, no native final-MSE claim.',layers=results,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(DEST/'PAIR_SELF_FEATURE22_RESPONSE_v1.json').open('x').write(json.dumps(out,indent=2)+'\n')
dest=DEST/'PAIR_SELF_FEATURE22_RESPONSE_v1.npz';assert not dest.exists();np.savez(dest,**directions)
print(json.dumps(out))
