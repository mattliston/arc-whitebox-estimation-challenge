"""Target-free, frozen20-step first-hidden reconstruction and full90 custody."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,ast,gzip,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE.parents[1];V=HERE.parents[2]/'astra_oct20_vm_frontier_v1'
RUNS=Path('/home/arcworker/arc-runs');CAP=RUNS/'full_state_response_train90_v1';OUT=RUNS/'first_hidden128_reconstruct90_v1'
PREREG='20c127e280f6bc40ee95013406f0c5eb46684e1bbc7e0c9d10d0e74444ba5e69'
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(name,path,pin):
 assert sha(path)==pin,(str(path),sha(path))
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def setup():
 assert sys.platform=='linux' and sha(HERE/'PREREG_FIRST128_v1.json')==PREREG
 p=read(HERE/'PREREG_FIRST128_v1.json');assert len(p['train'])==90 and len(p['reconstruction']['shards'])==6
 ph=module('fixed_physical_custody',D/'physical_input29_v1/fit_physical29_v1.py',p['pinned_dependencies']['physical_custody_module_sha256'])
 inv=module('fixed_inverse20',HERE/'diagnose_hc_inverse_v1.py',p['reconstruction']['script_sha256'])
 f=module('fixed_parameter_decoder',V/'fit_frontier_v1.py','b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363')
 source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(source)==SCI
 text=source.read_text();literal,_=f.literals_and_functions(text);P=f.decode(literal['_GNN_LITERAL']);extra={}
 for n in ast.parse(text).body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in ('_HC_U','_HCF_U'):extra[t.id]=np.asarray(ast.literal_eval(n.value),'f4')
 assert extra['_HC_U'].shape==(5,128) and extra['_HCF_U'].shape==(1,128)
 return p,ph,inv,f,P,extra
def clean_guard(path,threads=1):
 g=read(path)
 assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==threads and g['max_owned_swap_bytes']==0
 return g
def validate_anchor(p):
 dp=RUNS/'first_hidden128_hc_inverse_train4_v1/DIAGNOSTIC_v1.json'
 assert sha(dp)==p['reconstruction']['anchor_report_sha256']
 a=read(dp);assert a['reconstruction_passed'] and a['source_sha256']==SCI and len(a['rows'])==60 and all(r['passed'] for r in a['rows'])
 assert a['no_truth_arrays_read'] and a['no_response_fit'] and a['script_sha256']==p['reconstruction']['script_sha256']
 clean_guard(RUNS/'first_hidden128_hc_inverse_train4_GUARD_v1.json')
 pp=RUNS/'first_hidden128_readout_price_v1/PRICE_v1.json';assert sha(pp)==p['price_reference']['sha256']
 pg=RUNS/'first_hidden128_readout_price_GUARD_v1.json';assert sha(pg)==p['price_reference']['guard_sha256'];clean_guard(pg)
 price=read(pp);assert price['successful'] and price['passes_price_gate']
def prepare():
 p,ph,inv,f,P,extra=setup();validate_anchor(p)
 hfiles,proof=ph.validate_full_h90(p['train'])
 planp=D/'truth_response95_v1/PLAN_CAPTURE_v1.json';assert sha(planp)==ph.PLAN
 plan=read(planp);rows=[];seen=set()
 for gi,group in enumerate(plan['groups'],1):
  folder=CAP/('part%02d'%gi);gp=CAP/('part%02d_GUARD.json'%gi);clean_guard(gp,4)
  cd=folder/'gnn_full_state_v1';meta=read(cd/'METADATA_v1.json')
  assert meta['source_sha256']==SCI and meta['hook_sha256']==ph.HOOK and meta['target_free']
  captured=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],captured[k]) for k in P)
  done=read(cd/'CAPTURE_COMPLETE_v1.json');assert done['complete'] and done['target_free']
  for seed in group:
   raw=folder/('diagnostic_pred_%d.bin'%seed);assert raw.stat().st_size==65536
   for l in range(1,16):
    rr=[r for r in done['files'] if r['rank']==seed and r['layer']==l and r['phase']=='scored'];assert len(rr)==1
    rp=cd/rr[0]['path'];assert sha(rp)==rr[0]['sha256']
    with gzip.open(rp,'rt') as stream:r=json.load(stream)
    hidden=inv.arr(r,'hidden2');gate=inv.arr(r,'gate_feature');assert hidden.shape==(1024,128) and gate.shape==(1024,)
    hr=hfiles[seed,l];hp=Path(hr['path']);assert sha(hp)==hr['sha256']
    H=np.fromfile(hp,dtype='<f4').reshape(1024,78);assert np.isfinite(H).all();np.testing.assert_array_equal(H[:,5],gate)
    assert (seed,l) not in seen;seen.add((seed,l))
    rows.append(dict(seed=seed,layer=l,H_path=str(hp),H_sha256=sha(hp),state_capture_path=str(rp),state_capture_sha256=sha(rp),prediction_path=str(raw),prediction_sha256=sha(raw),guard_path=str(gp),guard_sha256=sha(gp)))
 assert seen=={(s,l) for s in p['train'] for l in range(1,16)}
 OUT.mkdir(exist_ok=False)
 result=dict(utc=utc(),successful=True,source_sha256=SCI,prereg_sha256=PREREG,script_sha256=sha(__file__),train=p['train'],rows=rows,fullH_capture_receipts=proof,all90_H5_byte_exact=True,no_truth_arrays_read=True,no_response_fit=True)
 write(OUT/'CUSTODY_v1.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','train','fullH_capture_receipts')}),flush=True)
def reconstruct(shard):
 p,ph,inv,f,P,extra=setup();assert 1<=shard<=6
 clean_guard(RUNS/'first_hidden128_reconstruct90_custody_GUARD_v1.json')
 cp=OUT/'CUSTODY_v1.json';custody=read(cp);assert custody['successful'] and custody['source_sha256']==SCI and custody['script_sha256']==sha(__file__) and custody['prereg_sha256']==PREREG
 seeds=p['reconstruction']['shards'][shard-1];folder=OUT/('shard%02d'%shard);folder.mkdir(exist_ok=False);records=[]
 for cr in custody['rows']:
  seed,l=cr['seed'],cr['layer']
  if seed not in seeds:continue
  hp=Path(cr['H_path']);rp=Path(cr['state_capture_path']);assert sha(hp)==cr['H_sha256'] and sha(rp)==cr['state_capture_sha256']
  H=np.fromfile(hp,dtype='<f4').reshape(1024,78)
  with gzip.open(rp,'rt') as stream:r=json.load(stream)
  target=inv.arr(r,'hidden2');assert H.shape==(1024,78) and target.shape==(1024,128)
  U=np.concatenate([extra['_HC_U'],extra['_HCF_U']],axis=0).astype('f8') if l==15 else extra['_HC_U'].astype('f8')
  base=(H@P['enc0_w'].T).astype('f8')+P['enc0_b'].astype('f8')
  z,h1,h2,ratio=inv.inverse(base,target.astype('f8'),U,P['enc2_w'].astype('f8'),P['enc2_b'].astype('f8'))
  h1f=h1.astype('f4');pre2=h1f@P['enc2_w'].T+P['enc2_b'];deployed=pre2/(np.float32(1)+np.exp(-pre2))
  metric=inv.metric(h2.astype('f4'),target);replay=inv.metric(deployed,target)
  passed=bool(inv.small(metric,128,16) and inv.small(replay,128,16) and ratio>=1e-8 and np.isfinite(h1f).all() and np.isfinite(z).all() and np.max(np.abs(z))<=8)
  fp=folder/('rank%d_L%02d.first128_f32.bin'%(seed,l));h1f.astype('<f4').tofile(fp)
  zp=folder/('rank%d_L%02d.z_f32.bin'%(seed,l));z.astype('<f4').tofile(zp)
  records.append(dict(seed=seed,layer=l,passed=passed,path=str(fp),sha256=sha(fp),bytes=fp.stat().st_size,z_path=str(zp),z_sha256=sha(zp),z_shape=list(z.shape),hidden2=metric,quantized_hidden1_replay=replay,min_jacobian_normal_eigen_ratio=ratio,z_absmax=float(np.max(np.abs(z))),H_sha256=sha(hp),state_capture_sha256=sha(rp)))
  if l==15:print(json.dumps(dict(utc=utc(),shard=shard,seed=seed,records=len(records),all_passed=all(x['passed'] for x in records))),flush=True)
 assert [(r['seed'],r['layer']) for r in records]==[(s,l) for s in seeds for l in range(1,16)]
 result=dict(utc=utc(),successful=True,reconstruction_passed=all(r['passed'] for r in records),shard=shard,train=seeds,rows=records,source_sha256=SCI,prereg_sha256=PREREG,script_sha256=sha(__file__),custody_sha256=sha(cp),no_truth_arrays_read=True,no_response_fit=True)
 write(folder/'RECON_COMPLETE_v1.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
def validate_all_recon():
 p,ph,inv,f,P,extra=setup();cp=OUT/'CUSTODY_v1.json';custody=read(cp)
 assert custody['successful'] and custody['source_sha256']==SCI and custody['prereg_sha256']==PREREG and custody['script_sha256']==sha(__file__)
 clean_guard(RUNS/'first_hidden128_reconstruct90_custody_GUARD_v1.json')
 files={};proof=[]
 for shard,seeds in enumerate(p['reconstruction']['shards'],1):
  gp=RUNS/('first_hidden128_reconstruct90_shard%02d_GUARD_v1.json'%shard);clean_guard(gp)
  rp=OUT/('shard%02d'%shard)/'RECON_COMPLETE_v1.json';r=read(rp)
  assert r['successful'] and r['reconstruction_passed'] and r['source_sha256']==SCI and r['prereg_sha256']==PREREG and r['script_sha256']==sha(__file__)
  assert r['train']==seeds and r['custody_sha256']==sha(cp) and r['no_truth_arrays_read'] and r['no_response_fit']
  assert len(r['rows'])==225
  for row in r['rows']:
   key=(row['seed'],row['layer']);assert key not in files and row['passed'] and row['seed'] in seeds
   fp=Path(row['path']);assert fp.stat().st_size==row['bytes']==1024*128*4 and sha(fp)==row['sha256']
   assert inv.small(row['hidden2'],128,16) and inv.small(row['quantized_hidden1_replay'],128,16) and row['min_jacobian_normal_eigen_ratio']>=1e-8 and row['z_absmax']<=8
   files[key]=row
  proof.append(dict(shard=shard,report_path=str(rp),report_sha256=sha(rp),guard_sha256=sha(gp)))
 assert set(files)=={(s,l) for s in p['train'] for l in range(1,16)}
 # Recheck every source/H/hidden/prediction hash before labels, without reading any truths.
 hfiles,hproof=ph.validate_full_h90(p['train'])
 for cr in custody['rows']:
  key=(cr['seed'],cr['layer']);rr=files[key]
  assert sha(cr['state_capture_path'])==cr['state_capture_sha256']==rr['state_capture_sha256']
  assert sha(cr['H_path'])==cr['H_sha256']==rr['H_sha256']==hfiles[key]['sha256']
  assert sha(cr['prediction_path'])==cr['prediction_sha256']
 return files,dict(custody_sha256=sha(cp),all90_reconstruction_passed=True,all90_before_truth_array_open=True,shards=proof,fullH_capture_receipts=hproof)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--shard',type=int);args=ap.parse_args()
 if args.prepare:prepare()
 else:assert args.shard is not None;reconstruct(args.shard)
if __name__=='__main__':main()
