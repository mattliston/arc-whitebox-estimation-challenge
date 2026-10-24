"""Frozen historical prefix and one passive analytic fourth-face supplier."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
P2=HERE.parents[1];ASTRA=P2/'astra_phase2_reexamination_v1'
sys.path.insert(0,str(ASTRA));sys.path.insert(0,str(HERE.parent/'fourth_matrix_face_capacity_v1'))
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from physical_projection import project
from supplier import make_supplier

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()

seal=json.loads((HERE/'seal.json').read_text())
for path,digest in seal['files'].items():assert sha(Path(path))==digest,path
assert json.loads((HERE/'units.json').read_text())['passed']
assert json.loads((HERE/'projection_units.json').read_text())['passed']
pg=json.loads((HERE/'projection_units_guard.json').read_text())
assert pg['completed'] and pg['whole_group_retired'] and pg['max_swap_growth_bytes']==0
g=json.loads((HERE/'units_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
out=HERE/'capture.json';assert not out.exists() and not out.with_suffix('.npz').exists()
start=time.monotonic()
folder=ASTRA/'rows_current_transport_shadow_v2/row_0035'
W=np.load(folder/'weights.npy',allow_pickle=False)
oldx=np.load(folder/'x.npy',mmap_mode='r');olds=np.load(folder/'sigma.npy',mmap_mode='r')
P=dict(np.load(ASTRA/'payload_compact_feed_fullpair_v1.npz',allow_pickle=False))
f=dict(np.load(ASTRA/'fourth_trace_oracle_inputs_r35_v1.npz',allow_pickle=False))
state=np.zeros((1024,64),np.float32)
checks=[];saved={};call=0;pair=make_supplier()

class Complete(Exception):pass

def nonlinear(*args,**kwargs):
 global call
 if call==12:
  ans=pair(*args,**kwargs)
  saved['analytic_post31']=ans[4]['post31'].copy()
  saved['analytic_post22']=ans[4]['post22'].copy()
  saved['analytic_k4']=ans[4]['post_k4'].copy()
 else:ans=parent.env['nonlinear'](*args,**kwargs)
 call+=1
 return ans

def hook(layer,m,C,K21,aux,w):
 global state
 x=features(layer,m,C,K21,aux,w)
 checks.append(bool(np.array_equal(x,oldx[layer]) and np.array_equal(aux['sigma'].astype(np.float32),olds[layer])))
 assert checks[-1],layer
 delta,state=gnn_step(layer,x,aux['sigma'],w,state,P)
 return delta

def post(layer,values):
 if layer!=12:return
 assert len(checks)==13 and all(checks)
 assert np.array_equal(values['C'],f['post12_covariance'])
 assert float(values['radial_new'])==float(f['post12_radial'])
 C=(values['C']+values['C'].T)/2
 v=np.diag(C)
 live=np.outer(values['alive'],values['alive'])
 q4=saved['analytic_k4']
 q31=saved['analytic_post31']*live
 q22=saved['analytic_post22']*live
 np.fill_diagonal(q31,q4);np.fill_diagonal(q22,q4)
 q22=(q22+q22.T)/2
 D,info=project(C,q4,q31,q22)
 w=W[13].astype(np.float64);pushed=w.T@D@w
 np.savez(out.with_suffix('.npz'),Dpost12=D,Dpre13=pushed,post12_covariance=C,
          analytic_post31=saved['analytic_post31'],analytic_post22=saved['analytic_post22'],
          analytic_k4=saved['analytic_k4'],alive=values['alive'])
 record=dict(passed=True,prefix_feature_sigma_byte_parity=checks,frozen_post12_covariance_byte_parity=True,
             projection=info,active_count=int(values['alive'].sum()),variance_min=float(v.min()),nonpositive_count=int((v<=0).sum()),
             truth_accessed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             utc_finished=datetime.now(timezone.utc).isoformat(),output_sha256=sha(out.with_suffix('.npz')),
             qualification='Passive single post12 analytic supplier, unchanged old rank35 prefix; not a recurrent candidate.')
 with out.open('x') as o:json.dump(record,o,indent=2);o.write('\n')
 print(json.dumps(record,indent=2),flush=True)
 raise Complete

source=parent.PREDICT_SOURCE
old='        radial = radial_new';assert source.count(old)==1
source=source.replace(old,'        post(l,locals())\n'+old)
env=dict(parent.env,nonlinear=nonlinear,post=post);exec(source,env)
try:env['predict'](W,hook=hook,gate_floor=.01)
except Complete:pass
assert out.exists()
