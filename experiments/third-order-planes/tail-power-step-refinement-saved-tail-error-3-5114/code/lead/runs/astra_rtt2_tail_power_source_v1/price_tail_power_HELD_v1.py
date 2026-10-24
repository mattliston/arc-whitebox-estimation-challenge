from pathlib import Path
import importlib.util,json,hashlib,sys
import flopscope as fs
import flopscope.numpy as fnp
from flopscope import _budget
D=Path(__file__).resolve().parent;R=D.parent
sp=importlib.util.spec_from_file_location('tariff',R/'astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
original=_budget._call_numpy
for name,module in list(sys.modules.items()):
 if name.startswith('flopscope')and getattr(module,'_call_numpy',None)is original:module._call_numpy=p.backend
r=json.loads((D/'OFFLINE_TAIL_POWER_RESULT_v1.json').read_text());prices=[]
for row in r['per_layer']:
 m,k,c,t,old=row['m'],row['k'],row['c15'],row['tail'],row['oldrank']
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  A=fnp.empty((m,k),dtype=fnp.float32);B=fnp.empty((t,k),dtype=fnp.float32);Q=fnp.empty((m,2),dtype=fnp.float32);P=fnp.empty((m,c),dtype=fnp.float32)
  I=fnp.empty((m,old),dtype=fnp.float32);V=fnp.empty((c,old),dtype=fnp.float32)
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as cc:
  v=P.T@Q
  if old:v=v-V@(I.T@Q)
  u=P@v
  if old:u=u-I@(V.T@v)
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as cn:u=A@(B.T@(B@(A.T@Q)))
 prices.append(dict(layer=row['layer'],prefix_refine_flops=int(cc.flops_used),tail_refine_flops=int(cn.flops_used),extra_flops=int(cn.flops_used-cc.flops_used),prefix_charged_ops=len(cc._op_log),tail_charged_ops=len(cn._op_log)))
extra=sum(x['extra_flops']for x in prices)
out=dict(status='ACTUAL_PUBLIC_FNP_TARIFF_METADATA_ONLY',flopscope_version=fs.__version__,per_layer=prices,extra_flops=extra,extra_percent_245G=100*extra/245e9,charged_op_delta=sum(x['tail_charged_ops']-x['prefix_charged_ops']for x in prices),unchanged='Initial/final QR shape, final scores/T/tail construction, exact oldterm, c2 mask and full source state unchanged.',conservative='Does not credit deleting old_right prefix gather; weighted tail already paid and can be reused. Whole-network exact native bill is not claimed.',numerical_payload_bytes=p.PAYLOAD[0],models=0)
p=D/'TAIL_POWER_TARIFF_v1.json';p.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='per_layer'}));print(hashlib.sha256(p.read_bytes()).hexdigest())
