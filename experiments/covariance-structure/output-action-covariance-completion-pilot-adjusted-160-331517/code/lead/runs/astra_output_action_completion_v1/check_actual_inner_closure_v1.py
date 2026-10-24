from pathlib import Path
import os,sys,types,json,hashlib,resource,time
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;S=D/'candidate_output_action_HELD_v6.py';assert os.getpriority(os.PRIO_PROCESS,0)==10
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('oac_symmetric_check');M.__file__=str(S)
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 exec(compile(S.read_text(),str(S),'exec'),M.__dict__)
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
 M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
rng=np.random.default_rng(902);rows=[]
for n,k in ((704,832),(704,896),(768,896),(768,960),(832,960),(832,1024),(896,1024),(960,1024)):
 a=(rng.normal(size=(n,k))/k**.5).astype('f')
 q=(rng.normal(size=(k,8))/8**.5).astype('f');c=(np.diag(rng.uniform(.01,1,k))+.1*q@q.T).astype('f');b=(c@a.T).astype('f');ref=a.astype('d')@b.astype('d')
 with fs.BudgetContext(flop_budget=2**60,quiet=True):aa=fnp.asarray(a);bb=fnp.asarray(b)
 for repeat in range(1):
  with fs.BudgetContext(flop_budget=2**60,quiet=True):out=M._oac_symmetric(aa,bb)
  actual=np.asarray(out).copy();relative=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref));sym=float(np.max(abs(actual-actual.T)))
  print(n,repeat,relative,sym,flush=True)
  assert relative<2e-4 and sym/float(np.max(abs(ref)))<2e-5
  assert np.array_equal(np.asarray(aa),a)and np.array_equal(np.asarray(bb),b)
  rows.append(dict(output=n,inner=k,repeat=repeat,relative=relative,symmetry_max=sym))
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000
print(json.dumps(rows));result=dict(candidate_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,captures=0,status='PASS_ORIGINAL_ENGINE_VARIABLE_OUTPUT_NUMERIC',limits=['All eight actual pilot output/input shapes; synthetic PSD covariance congruence with native FP32 first. No new model/capture.'])
(D/'ACTUAL_INNER_SYMMETRIC_CLOSURE_v1.json').open('x').write(json.dumps(result,indent=2)+'\n')

