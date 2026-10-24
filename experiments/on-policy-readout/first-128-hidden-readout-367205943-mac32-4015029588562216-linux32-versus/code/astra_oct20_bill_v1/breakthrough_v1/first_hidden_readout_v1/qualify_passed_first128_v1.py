"""Pass-only frozen direct edge8+complement+first128 source export and complete source qualification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parents[2]/'astra_oct20_vm_frontier_v1';A=HERE.parents[2]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f
sha=f.sha;utc=f.utc;read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as out:json.dump(x,out,indent=2,allow_nan=False);out.write('\n')
def rec(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),effective_sha256=sha(p))
PREREG='20c127e280f6bc40ee95013406f0c5eb46684e1bbc7e0c9d10d0e74444ba5e69'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
assert sha(HERE/'PREREG_FIRST128_v1.json')==PREREG
fitp=RUNS/'first_hidden128_local90_fit_v1/FITTED_v1.json';fit=read(fitp)
assert fit['prereg_sha256']==PREREG and fit['script_sha256']=='8e252f559879be4d6c2bceb033bf9d84408ada1c418942bcf4d4d939d009c526'
assert fit['no_protected_truth_read'] and fit['no_candidate_exported'] and len(fit['provenance'])==90
assert fit['source_sha256']==SCI and fit['release_source_sha256']==REL
finish=read(HERE/'PIPELINE_FINAL_v1.json');assert finish['successful'] and finish['fitted_sha256']==sha(fitp)
gp=RUNS/'first_hidden128_local90_fit_v1_GUARD_v1.json';guard=read(gp)
assert guard['successful'] and guard['child_exit']==0 and guard['whole_group_retired'] and not guard['cleanup_errors'] and guard['max_owned_swap_bytes']==0
assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
out=HERE/'native_v1';out.mkdir(exist_ok=False)
if not fit['passed']:
 assert fit['status']=='REJECT_BEFORE_NATIVE'
 write(out/'QUALIFIED_v1.json',dict(utc=utc(),successful=True,qualified=[],rejected=True,fit_sha256=sha(fitp),no_native_launch=True))
 print('FROZEN_INCREMENTAL_CV_REJECT_NO_EXPORT');sys.exit(0)
assert fit['status']=='INCREMENTAL_PROXY_PASS_NATIVE_REQUIRED'
assert fit['first128_reconstruction_custody']['all90_reconstruction_passed'] and fit['first128_reconstruction_custody']['all90_before_truth_array_open']
assert finish['fit_guard_sha256']==sha(gp)
assert fit['loo']['n']==90 and fit['loo']['mean_incremental_relative_pct']<=-.15 and fit['loo']['incrementally_lower']>=54
assert sum(x['mean_incremental_relative_pct']<0 for x in fit['descriptive_blocks'])>=4
assert fit['physical']['incremental_correction_to_original_head_rms']<=.25 and fit['physical']['max_layer_ratio']<=1.
KO=HERE.parents[2]/'fable_oct19_kc_v1'
parents={'science':KO/'candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py',
 'release':KO/'candidate_rel_astra_o21_edge8_hidden_complement90_v1.py'}
pins={'science':'b54f81fff93462607a7129eb7d42ba619fb0b001f3a09b01e71c713a8740214c',
 'release':'4685e1454ebd2d9607fcd88b64cc1d8435050761212efd1902b04d3f83151a46'}
for role,path in parents.items():assert sha(path)==pins[role]
original=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(original)==SCI
literal,_=f.literals_and_functions(original.read_text());P=f.decode(literal['_GNN_LITERAL'])
bliteral,_=f.literals_and_functions(parents['science'].read_text());B=f.decode(bliteral['_GNN_LITERAL'])
assert set(B)==set(P)|{'hidden_head'} and all(np.array_equal(P[k],B[k]) for k in P if k not in ('head','head_b'))
baseline=np.asarray(fit['baseline_coef_by_layer'],dtype='f8');assert baseline.shape==(15,321)
np.testing.assert_array_equal(baseline[:,193:],0)
np.testing.assert_array_equal((P['head'][1:].astype('f8')+baseline[:,:64]).astype('f4'),B['head'][1:])
np.testing.assert_array_equal((P['head_b'][1:].astype('f8')+baseline[:,64]).astype('f4'),B['head_b'][1:])
np.testing.assert_array_equal(baseline[:,65:193].astype('f4'),B['hidden_head'][1:])
c=np.asarray(fit['coef_by_layer'],dtype='f8');assert c.shape==(15,321) and np.isfinite(c).all()
Q={k:v.copy() for k,v in B.items()}
Q['head'][1:16]=(P['head'][1:16].astype('f8')+c[:,:64]).astype('f4')
Q['head_b'][1:16]=(P['head_b'][1:16].astype('f8')+c[:,64]).astype('f4')
Q['hidden_head'][1:16]=c[:,65:193].astype('f4')
Q['first_hidden_head']=np.zeros((16,128),dtype='f4');Q['first_hidden_head'][1:16]=c[:,193:].astype('f4')
np.testing.assert_array_equal(Q['head'][1:].astype('f8')-P['head'][1:].astype('f8'),c[:,:64])
np.testing.assert_array_equal(Q['head_b'][1:].astype('f8')-P['head_b'][1:].astype('f8'),c[:,64])
np.testing.assert_array_equal(Q['hidden_head'][1:].astype('f8'),c[:,65:193])
np.testing.assert_array_equal(Q['first_hidden_head'][1:].astype('f8'),c[:,193:])
assert all(np.array_equal(B[k],Q[k]) for k in B if k not in ('head','head_b','hidden_head'))
rt=HERE/'FIRST_READOUT_PRICE_TAIL_v1.py';assert sha(rt)=='5f044a20afe7cfb7c0c187aa0e2d89e7006a807431df5cf961f3e5efccb34a56'
tail='\n# Frozen TRAIN90 per-layer first-hidden128 readout on exact edge8+complement; no inference state change.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\n_GNN_LITERAL = '+repr(f.encode(Q))+'\n'+rt.read_text()+'\n_FAST_RESCUE_FRAC = {}\n'
name='edge8_complement_first128_local90';folder=out/name;folder.mkdir();outputs={}
for role,parent in parents.items():
 old=parent.read_text();source=old+tail;p=folder/('candidate_'+name+'_'+role+'_v1.py');compile(source,str(p),'exec')
 oldnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 newnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 tailnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(rt.read_text()).body if isinstance(n,ast.FunctionDef)}
 assert set(newnodes)==set(oldnodes)|{'_fh90_first_head'}
 assert {k:v for k,v in newnodes.items() if k not in tailnodes}=={k:v for k,v in oldnodes.items() if k not in tailnodes}
 assert all(newnodes[k]==v for k,v in tailnodes.items())
 with p.open('x') as stream:stream.write(source)
 decoded,_=f.literals_and_functions(source);decoded=f.decode(decoded['_GNN_LITERAL']);assert all(np.array_equal(Q[k],decoded[k]) for k in Q)
 outputs[role]=dict(path=str(p),sha256=sha(p),parent_path=str(parent),parent_sha256=sha(parent))
build=folder/'BUILD_v1.json'
write(build,dict(utc=utc(),outputs=outputs,fit_sha256=sha(fitp),prereg_sha256=PREREG,tail_sha256=sha(rt),
 exact_original_source_prefix=True,baseline_tensors_reproduce_deployed_parent_exactly=True,training_trajectory_source_sha256=SCI,native_transfer_not_yet_verified=True,only_changed_functions=['_gop2_gnn_step'],new_functions=['_fh90_first_head'],
 changed_parameters=['head[1:16]','head_b[1:16]','hidden_head[1:16]','new first_hidden_head(16,128)'],state_message_nonlinear_path_unchanged=True,
 requires_native_validation=True,no_native_launch=True,first_hidden_reconstruction_custody=fit['first128_reconstruction_custody']))
tiny=HERE/'check_first128_tiny_v1.py';imp=HERE/'check_first128_import_v1.py'
recipe=read(HERE/'QUALIFY_RECIPE_v1.json')
assert sha(tiny)==recipe['tiny_sha256'] and sha(imp)==recipe['import_sha256'] and sha(__file__)==recipe['qualifier_sha256']
for tool,python,outname in [(tiny,'/home/arcworker/arc-envs/server/bin/python','TINY_OFFICIAL_v1.json'),(imp,'/home/arcworker/arc-envs/client/bin/python','NUMPY_FREE_IMPORT_v1.json')]:
 with (folder/(outname+'.log')).open('x') as log:subprocess.run([python,'-B',str(tool),'--build',str(build),'--out',str(folder/outname)],stdout=log,stderr=subprocess.STDOUT,check=True)
 r=read(folder/outname);assert r['successful']
assert read(folder/'TINY_OFFICIAL_v1.json')['state_path_byte_identical']
source=Path(outputs['science']['path']);plans={}
for key,base,pin in [
 ('mac',V/'PLAN_BANK8_ACTUAL_SHARED_v1.json','cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'),
 ('linux',A/'cloud_linux_markline_v1/PLAN_MSG8_LOADERFIX_CONTROL_v1.json','9276f1d7d0dfc8b9d5a51c768a98e3eb2e1f2d152f6b038240db0a43b0d075a6')]:
 assert sha(base)==pin;plan=read(base);plan.update(utc=utc(),kind='FROZEN_TRAIN90_FIRST_HIDDEN128_DIRECT_B54_STACK',reference_submission=337466,native_parent_source_sha256=pins['science'],fit_sha256=sha(fitp),prereg_sha256=PREREG)
 plan['candidates']['edge8_hidden_complement90']=dict(rec(parents['science']),role='measured_native_parent')
 plan['candidates'][name]=dict(rec(source),role='frozen_candidate')
 pp=folder/('PLAN_'+key.upper()+'_v1.json');write(pp,plan);plans[key]=dict(path=str(pp),sha256=sha(pp))
if 'linux' in plans:
 sys.path.insert(0,str(A/'cloud_linux_markline_v1'));from common_linux_v7 import load_plan,candidate_path
 pp=Path(plans['linux']['path']);lp=load_plan(ROOT,pp);assert sha(candidate_path(ROOT,lp['candidates'][name],folder))==sha(source)
ready=dict(utc=utc(),successful=True,candidate=name,source_sha256=sha(source),outputs=outputs,plans=plans,
 fit_sha256=sha(fitp),fit_guard_sha256=sha(gp),prereg_sha256=PREREG,build_sha256=sha(build),
 tiny_sha256=sha(folder/'TINY_OFFICIAL_v1.json'),import_sha256=sha(folder/'NUMPY_FREE_IMPORT_v1.json'),
 requested_jobs=12,shared_max_active=36,blas_threads=4,no_native_launch=True,reference_submission=337466,native_parent_source_sha256=pins['science'],
 local_proxy_only=True,requires_actual_incremental_comparison_to_b54_stack=True)
write(folder/'LINUX_READY_v1.json',ready)
write(out/'QUALIFIED_v1.json',dict(utc=utc(),successful=True,qualified=[ready],fit_sha256=sha(fitp),script_sha256=sha(__file__),no_native_launch=True,no_adjusted_gain_claim=True))
print(json.dumps(dict(utc=utc(),qualified=[name],source_sha256=sha(source))),flush=True)

