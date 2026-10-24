"""Actual public-client terminal P circuit proof; never executes a model."""
import array, gc, hashlib, json, random, sys, time, types
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];O=R.parents[1]
sys.path.insert(0,str(O/'release'));import rusage
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection, _dispatch
requests=[0];send=_connection.Connection.send_recv
def counted_send(self,raw):
    requests[0]+=1
    return send(self,raw)
_connection.Connection.send_recv=counted_send
_dispatch.__dict__.update(_ts_raw=rusage.raw,_ts_depth=[0],_ts_acc=[0,0],_ts_span=_dispatch._counted_span)
exec('''
@contextmanager
def _counted_span():
 with _ts_span():
  top=_ts_depth[0]==0
  if top:c0,i0=_ts_raw()
  _ts_depth[0]+=1
  try:yield
  finally:
   _ts_depth[0]-=1
   if top:
    c1,i1=_ts_raw();_ts_acc[0]+=c1-c0;_ts_acc[1]+=i1-i0
''',_dispatch.__dict__)
source=R/'astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py'
fragment=D/'fragment_terminal_scale_v1.py.txt'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
M=types.ModuleType('terminal_scale_candidate');M.__file__=str(source);sys.modules[M.__name__]=M
with flops.BudgetContext(flop_budget=2**50,quiet=True):
    exec(compile(source.read_text(),str(source),'exec'),M.__dict__)
    exec(compile(fragment.read_text(),str(fragment),'exec'),M.__dict__)
    M._tps_bind_coefficients()  # same setup-context creation; no full model setup
def measure(call):
    r0=requests[0];d0=_dispatch._ts_acc[:];c0,i0=rusage.raw();t0=time.perf_counter()
    with flops.BudgetContext(flop_budget=2**50,quiet=True)as budget:result=call()
    wall=time.perf_counter()-t0;c1,i1=rusage.raw();d1=_dispatch._ts_acc[:]
    return result,dict(requests=requests[0]-r0,client_instructions=i1-i0,
        residual_instructions=i1-i0-(d1[1]-d0[1]),wall_s=wall,flops=int(budget.flops_used))
def fetch(x):
    with flops.BudgetContext(flop_budget=2**50,quiet=True):return x._fetch_data()[0]
rows=[];rng=random.Random(20261018)
for name in ('signed_executable_1','signed_executable_1_transpose','signed_executable_0'):
    plan=M._SMX_RECORDS[name]['prepared']['P']
    for shape in ((1,1),(7,11),(32,48),(224,256),(320,288)):
        count=shape[0]*shape[1]
        with flops.BudgetContext(flop_budget=2**50,quiet=True):
            inputs=[]
            for i in range(len(plan[0])):
                # Include signed zero, small/large finite exponents and irregular mantissas.
                pattern=[rng.uniform(-1,1)*2**((i%9-4)*8) for _ in range(97)]+[0.0,-0.0]
                data=array.array('f',(pattern[j%len(pattern)]for j in range(count)))
                inputs.append(fnp.reshape(fnp.frombuffer(data.tobytes(),dtype=fnp.float32),shape))
            inputs=tuple(inputs)
            sizes=[len(nodes)for head,nodes in plan[1]];total=sum(sizes)
            nodes=fnp.empty((total,*shape),dtype=fnp.float32);cuts=[];at=0
            for size in sizes[:-1]:at+=size;cuts.append(at)
            prepared=M._Sm40Prepared((tuple(fnp.split(nodes,cuts,axis=0)),tuple(fnp.unstack(nodes,axis=0))))
        def call(fn):return fn(fnp,plan,inputs,shape,dict(temp_peak=0),prepared=prepared)
        parent0,r0=measure(lambda:call(M._TPS_PARENT_GROUPED));b0=fetch(parent0)
        candidate0,c0=measure(lambda:call(M._astra_sm40_linear_grouped));bc0=fetch(candidate0)
        parent1,r1=measure(lambda:call(M._TPS_PARENT_GROUPED));b1=fetch(parent1)
        candidate1,c1=measure(lambda:call(M._astra_sm40_linear_grouped));bc1=fetch(candidate1)
        assert b0==bc0==b1==bc1
        assert parent0.shape==candidate0.shape and parent0.strides==candidate0.strides
        assert str(parent0.dtype)==str(candidate0.dtype)=='float32'
        # A following call rewrites every prepared intermediate. Returned output must own its data.
        with flops.BudgetContext(flop_budget=2**50,quiet=True):fnp.copyto(nodes,13.0)
        assert fetch(candidate0)==bc0 and fetch(parent0)==b0
        row=dict(record=name,shape=shape,bytes_sha=hashlib.sha256(b0).hexdigest(),output_shape=list(parent0.shape),
            output_strides=list(parent0.strides),output_bytes=len(b0),exact=True,fresh_ownership=True,
            parent_cold=r0,candidate_cold=c0,parent_warm=r1,candidate_warm=c1,
            delta_requests=c1['requests']-r1['requests'],delta_flops=c1['flops']-r1['flops'],
            delta_residual_instructions=c1['residual_instructions']-r1['residual_instructions'])
        rows.append(row);print(json.dumps({k:row[k]for k in ('record','shape','delta_requests','delta_flops','delta_residual_instructions')}),flush=True)
        del inputs,nodes,prepared,parent0,parent1,candidate0,candidate1;gc.collect()
out=dict(status='PASS',model_runs=0,source=str(source),source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
    fragment_sha=hashlib.sha256(fragment.read_bytes()).hexdigest(),rows=rows,
    proof='All three production P records; layer independent arithmetic; original add/sub order and destinations unchanged; finite float32 exponent stress, exact bytes/strides/dtype and fresh output ownership.',
    caveat='Component instruction measurements only; no network timing extrapolation.')
(D/'CLIENT_TINY_v1.json').write_text(json.dumps(out,indent=2)+'\n')
