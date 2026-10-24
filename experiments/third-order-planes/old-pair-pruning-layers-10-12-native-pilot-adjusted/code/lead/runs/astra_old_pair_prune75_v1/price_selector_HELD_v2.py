from pathlib import Path
import hashlib,importlib.util,json,types,sys,resource,time
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;p=D/'candidate_QR_OP75_ON_HELD_v2.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
s=importlib.util.spec_from_file_location('_op75_price',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
P=5856;n=672;t=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True):
 u=fnp.zeros((256,P),dtype=fnp.float32);v=fnp.zeros((320,P),dtype=fnp.float32);left=fnp.zeros((n,2*P),dtype=fnp.float32);lam=fnp.ones(2*P,dtype=fnp.float32)
with flops.BudgetContext(flop_budget=2**41,quiet=True) as c:
 k,cols,uv=m._op75_select((u,v,((0,P),),256),lam,left)
assert int(k.shape[0])==4416
price=json.loads((D.parent/'astra_read_cp512_cost_v1/PAIR_PRUNE75_VIRTUAL_TARIFF_v1.json').read_text())
out=dict(status='ACTUAL_FNP_SELECTOR_TARIFF_NO_MODEL',rows=n,P=P,kept=4416,selector_flops=c.flops_used,selector_ops=len(c._op_log),selection_calls_per_network=1,modeled_inner_credit=price['mean_credit'],modeled_net_credit=price['mean_credit']-c.flops_used,modeled_percent_of247G=100*(price['mean_credit']-c.flops_used)/247e9,seconds=time.monotonic()-t,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_whole_bill_still_authoritative=True)
(D/'SELECTOR_TARIFF_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
