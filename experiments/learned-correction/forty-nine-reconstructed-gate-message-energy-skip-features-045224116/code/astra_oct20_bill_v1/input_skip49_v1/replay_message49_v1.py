"""Target-free replay of exact source message/energy H columns; no truth loading."""
import ast,base64,hashlib,zlib
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
SOURCE_SHA='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
def replay(P,E,W,previous_states,gates,source,tiny):
 assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_SHA
 assert fs.__version__.startswith('0.12.3')
 defs={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 d=dict(fnp=fnp,_static_base64=base64,_static_zlib=zlib,_GOP2_ON=True,_GOP2_ACTIVE=[True],_GNN_PARAMS=None,
  _E2_GNN_STATE={},_E2_MSG={},_E2_WPOW={},_FAST_RESCUE_FRAC={},_HC_CONST={},_HCF_CONST={},_MESSAGE_EDGE_PARAMETERS=None,
  _GOP2_STATS=dict(calls=0,layers=[],energy_blocks=0,message_misses=0),_W3R_STATS=dict(calls=0,metric_builds=0),
  _VMFR_CM=8,_VMFR_ORIGINAL_W3_ENCODER=E,_VMFR_W3_ENCODER_ARRAY=None,_W3R_METRIC=None)
 exec(tiny.load_functions(source),d)
 exec(compile(ast.Module(body=[defs['_e2_wpow']],type_ignores=[]),'<exact-source-wpow>','exec'),d)
 result=[]
 with fs.BudgetContext(flop_budget=2**43,quiet=True) as ctx:
  p={k:fnp.asarray(v) for k,v in P.items()};d['_GNN_PARAMS']=p;k=d['_e2_gnn_consts'](p,1024)
  for l in range(1,16):
   st=np.asarray(previous_states[l-1],dtype='f4');ng=np.asarray(gates[l-1],dtype='f4')
   assert st.shape==(1024,64) and ng.shape==(1024,) and np.isfinite(st).all() and np.isfinite(ng).all()
   weights=fnp.asarray(np.asarray(W[l],dtype='f4'));state=fnp.asarray(st);gate=fnp.asarray(ng[:,None])
   fnp.copyto(k['Hg'],gate);d['_E2_MSG'].clear()
   M=d['_gop2_messages'](k,weights,state)
   fnp.power(k['SRC3'],2,out=k['SQ'])
   energy=d['_gop2_contract'](d['_e2_wpow'](weights,need6=True)['T135'],k['SQ'])
   fnp.multiply(k['Hg'],M,out=k['HM03']);fnp.maximum(energy,0,out=energy)
   fnp.log1p(energy,out=k['HM36']);fnp.clip(k['Hm'],-8,8,out=k['Hm'])
   h=np.concatenate((ng[:,None],np.asarray(k['Hm']).copy()),axis=1)
   assert h.shape==(1024,49) and np.isfinite(h).all();result.append(h)
  flops=int(ctx.flops_used)
 return result,dict(diagnostic_replay_flops=flops,exact_source_wpow=True,exact_source_gop2_and_w3=True,source_sha256=SOURCE_SHA)
