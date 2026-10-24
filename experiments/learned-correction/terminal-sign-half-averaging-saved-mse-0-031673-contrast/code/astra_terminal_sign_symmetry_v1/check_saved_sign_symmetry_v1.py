from pathlib import Path
import ast,base64,hashlib,json,math,os,resource,time,zlib
import numpy as np
from scipy.special import ndtr
D=Path(__file__).resolve().parent;R=D.parent;O=D.parents[2];START=time.monotonic()
assert os.getpriority(os.PRIO_PROCESS,0)==10
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
assert all(os.environ.get(k)=='1'for k in THREADS)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S=R/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';assert sha(S)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
def literals(p):
 t=ast.parse(p.read_text());names={'_GNN_LITERAL','_HCF_U','_HCF_COLUMNS'}
 return {x.id:ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)for x in n.targets if isinstance(x,ast.Name)and x.id in names}
L=literals(S);assert L==literals(R/'astra_native_terminal_hidden_refit_v1/candidate_QR_hidden_capture_HELD_v1.py');assert L['_HCF_COLUMNS']==('f15_diag',)
P={k:np.frombuffer(zlib.decompress(base64.b64decode(v[1])),dtype='f4').reshape(v[0]).copy()for k,v in L['_GNN_LITERAL'].items()};U=np.asarray(L['_HCF_U'],dtype='f4')
HRET=json.loads((R/'astra_native_terminal_hidden_refit_v1/ROOT_CAPTURE_RETIREMENT_v1.json').read_text());LRET=json.loads((R/'astra_native_terminal_linear_refit_v1/ROOT_CAPTURE_RETIREMENT_v1.json').read_text());META=json.loads((R/'astra_native_terminal_linear_refit_v1/CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text())
assert HRET['approved']and LRET['approved']and HRET['all_owned_PID_PPID_PGID_absent']and LRET['all_owned_PID_PPID_PGID_absent']
refs={x['seed']:x for x in LRET['evaluations']};meta={x['rank']:x for x in META['network_metadata']};ZPATH=R/'astra_enc0_hcf_recovery_v1/RECONSTRUCTED_HCF_Z_v1.npz';Z=np.load(ZPATH,allow_pickle=False)
def silu(x):return x/(np.float32(1)+np.exp(-x))
def head(H,z,sigma,rms,gain):
 bias=P['enc0_b']+np.einsum('kn,kc->nc',z[None,:],U,optimize=False)
 h=silu(H@P['enc0_w'].T+bias);h=silu(h@P['enc2_w'].T+P['enc2_b']);state=h@P['enc4_w'].T+P['enc4_b'];return ((sigma.astype('f4')*rms)*(state@P['head'][15]+P['head_b'][15]))*gain
MASK=np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1],dtype='f4')
def mirror_features(x):
 y=x.astype('d').copy();a=x[:,3].astype('d');m=x[:,8].astype('d');v=x[:,11].astype('d');s3=x[:,10].astype('d');s4=x[:,12].astype('d');c3=x[:,20].astype('d');c4=x[:,21].astype('d');mn=m-a
 vn=1-v+2*m*(a-m)
 r3=c3+3*m*v+m**3;rn3=r3-(a**3+3*a+s3);cn3=rn3-3*mn*vn-mn**3
 r4=c4+4*m*c3+6*m*m*v+m**4+3*v*v;rn4=(a**4+6*a*a+4*a*s3+3+s4)-r4;cn4=rn4-4*mn*cn3-6*mn*mn*vn-mn**4-3*vn*vn
 y[:,3]=-a;y[:,5]=1-x[:,5];y[:,7]=x[:,7]-a;y[:,8]=mn;y[:,10]=-s3;y[:,11]=vn;y[:,16]=-x[:,16];y[:,18]=-x[:,18];y[:,20]=cn3;y[:,21]=cn4;y[:,29]=x[:,29]-a.mean()
 return y.astype('f4')
# Check fixed reflection formulas independently on random scalar Edgeworth moments.
def scalar(a,s3,s4):
 phi=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=ndtr(a);mg=phi+a*g
 m=mg-s3*a*phi/6+s4*(a*a-1)*phi/24
 r2=(a*a+1)*g+a*phi+s3*phi/3-s4*a*phi/12
 r3=a*(a*a+3)*g+(a*a+2)*phi+s3*g+s4*phi/4
 r4=(a**4+6*a*a+3)*g+a*(a*a+5)*phi+4*s3*mg+s4*g
 v=r2-m*m;k3=r3-3*m*r2+2*m**3;k4=r4-4*m*r3+6*m*m*r2-3*m**4-3*v*v
 return m,v,k3,k4,mg,g,phi
rng=np.random.default_rng(20261015);a=rng.uniform(-5,5,1024);s3=rng.normal(size=1024)*.02;s4=rng.normal(size=1024)*.02;p=scalar(a,s3,s4);q=scalar(-a,-s3,s4);assert np.max(abs(p[0]-q[0]-a))<3e-14
x=np.zeros((1024,30),dtype='d');x[:,3]=a;x[:,5]=p[5];x[:,6]=p[6];x[:,7]=p[4];x[:,8]=p[0];x[:,9]=p[0]-p[4];x[:,10]=s3;x[:,11]=p[1];x[:,12]=s4;x[:,20]=p[2];x[:,21]=p[3];x[:,29]=p[0].mean();xm=mirror_features(x)
for j,v in ((8,q[0]),(11,q[1]),(20,q[2]),(21,q[3])):assert np.max(abs(xm[:,j]-v))<1e-6
rows=[];allodd=[];alltarget=[];parity=[]
# One frozen, untuned half-average; no fit and no candidate selection.
for e in sorted(HRET['evaluations'],key=lambda x:x['seed']):
 rank=e['seed'];ref=refs[rank];assert sha(e['capture_path'])==e['capture_sha256']and sha(ref['capture_path'])==ref['capture_sha256']
 with np.load(e['capture_path'],allow_pickle=False)as f:H=f['encoder_input'].copy()
 with np.load(ref['capture_path'],allow_pickle=False)as f:x=f['features30'].copy();host=f['host'];sigma=f['sigma'];delta=f['delta'];rms=f['rms'];gain=f['gain'];raw=f['raw_final_float64'];truth=f['target_final_float64'];base=f['predictions_float32'][15]
 assert not np.any(abs(H[:,30:78])>=8)and not np.any(H[:,5]==0)
 zn=Z[str(rank)];replay=head(H,zn,sigma,rms,gain);err=float(np.max(abs(replay-delta)));assert err<3e-8;parity.append(err)
 xn=mirror_features(x);Hn=H.copy();Hn[:,:30]=np.clip((xn-P['centers'][15])/P['scales'][15],-8,8)*MASK
 # Paid un-clipped messages are recovered to F32 roundoff; W/W3 are odd.
 ratio=Hn[:,5:6]/H[:,5:6];Hn[:,30:78]=np.clip(H[:,30:78]*ratio*np.tile(np.repeat(np.asarray([-1,1,-1],dtype='f4'),16),(1024,1)),-8,8)
 dm=head(Hn,zn,sigma,rms,gain);sym=np.maximum(raw+.5*(dm.astype('d')-delta),0).astype('f4').astype('d');bmse=float(np.mean((base.astype('d')-truth)**2));smse=float(np.mean((sym-truth)**2));odd=(delta.astype('d')-dm.astype('d'))*.5
 mm=meta[rank];assert sha(mm['truth_path'])==mm['truth_sha256'];w=np.load(mm['weights_path'],mmap_mode='r',allow_pickle=False)[15].astype('d')
 with np.load(mm['truth_path'],allow_pickle=False)as f:tprev=f['mc_post'][14].astype('d');assert np.array_equal(f['mc_post'][15].astype('d'),truth)
 mu_est=x[:,3].astype('d')*sigma;mu_true=w.T@tprev;mu_error=mu_est-mu_true;target=-.5*mu_error
 mirror_truth=truth-mu_true;mirror_base=host-mu_est+dm.astype('d');mirror_sym=host-mu_est+.5*(delta.astype('d')+dm.astype('d'))
 rows.append(dict(rank=rank,base_mse=bmse,symmetric_mse=smse,log_MSE_ratio=math.log(smse/bmse),original_delta_replay_max=err,mirrored_feature_max=float(np.max(abs(xn))),odd_RMS=float(np.sqrt(np.mean(odd*odd))),half_mu_error_RMS=float(np.sqrt(np.mean(target*target))),odd_target_cosine=float(np.dot(odd,target)/max(np.linalg.norm(odd)*np.linalg.norm(target),1e-30)),mirror_original_mse=float(np.mean((np.maximum(mirror_base,0)-mirror_truth)**2)),mirror_symmetric_mse=float(np.mean((np.maximum(mirror_sym,0)-mirror_truth)**2))))
 allodd.append(odd);alltarget.append(target)
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000 and time.monotonic()-START<150
odd=np.concatenate(allodd);target=np.concatenate(alltarget);logs=np.array([r['log_MSE_ratio']for r in rows]);out=dict(status='FIXED_SAVED_TEST_COMPLETE',candidate_parent_sha256=sha(S),capture_scope='Historical QR TRAIN32; frozen GNN and HCF constants identical to current1c913; no current-model accuracy claim.',rule='Exactly one unfit half-average of original and sign-reflected terminal learned deltas.',geometric_MSE_change_percent=float(100*np.expm1(logs.mean())),mean_log_MSE_percent=float(100*logs.mean()),log_SE_percent=float(100*logs.std(ddof=1)/np.sqrt(32)),network_wins=int(np.count_nonzero(logs<0)),network_count=32,max_control_delta_replay=float(max(parity)),odd_target_cosine=float(np.dot(odd,target)/(np.linalg.norm(odd)*np.linalg.norm(target))),odd_target_slope=float(np.dot(odd,target)/np.dot(target,target)),odd_RMS=float(np.sqrt(np.mean(odd*odd))),half_mu_error_RMS=float(np.sqrt(np.mean(target*target))),rows=rows,models=0,captures=0,fit=False,elapsed_seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'SAVED_SIGN_SYMMETRY_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}))
