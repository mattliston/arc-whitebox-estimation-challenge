"""Build only an inactive, reviewable candidate; no import/setup/predict."""
import ast,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
source=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
spec=importlib.util.spec_from_file_location('private_sm40',D/'smirnov_batched_v1.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
records=json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']
constants={}
for r in records:
 if r['shape']!=[3,6,3]:continue
 prepared={}
 for key in ('L','R','P'):
  ops,uses,dest=helper.prepare(r['circuits'][key]);uses={i:uses[i]for i,op,args,lit in ops}
  prepared[key]=(ops,uses,dest)
 constants[r['name']]=dict(shape=r['shape'],prepared=prepared)
prices=json.loads((D/'CURRENT24_FULL_KERNEL_PRICE_v1.json').read_text())
plans={tuple(r['shape']):(r['best']['name'],r['best']['L'],r['best']['group'])for r in prices['rows']if r['best']and r['best']['saving']>0}
helper_source=(D/'smirnov_batched_v1.py').read_text();tree=ast.parse(helper_source)
snippets=[]
for n in tree.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in ('linear','Kernel'):
  snippets.append(ast.get_source_segment(helper_source,n))
body='\n\n'.join(snippets).replace('linear(fnp,','_astra_sm40_linear(fnp,').replace('class Kernel:','class _AstraSm40Kernel:').replace("self.circuits={k:prepare(record['circuits'][k])for k in ('L','R','P')}","self.circuits=record['prepared']")
body=body.replace('math.prod','_rc_math.prod').replace('math.ceil','_rc_math.ceil')
prefix='''
# Private exact local Smirnov rank40 read route. Positive shape tariffs only.
_ASTRA_SM40_ON = True
_ASTRA_SM40_STATS = dict(calls=0, shapes=[], max_workspace_bytes=0)
'''
suffix='''
class _AstraSm40Environment:
    pass
_ASTRA_SM40_ENV = _AstraSm40Environment()
_ASTRA_SM40_ENV.fnp = fnp
_ASTRA_SM40_ENV._IPT_REGION = _IPT_REGION
_ASTRA_SM40_ENV._BH_HOLD = _BH_HOLD
_ASTRA_SM40_ENV._rs_nat_perm = _rs_nat_perm
_ASTRA_SM40_ENV._rs_basis_views = _rs_basis_views
_ASTRA_SM40_KERNELS = {name:_AstraSm40Kernel(_ASTRA_SM40_ENV,record) for name,record in _ASTRA_SM40_RECORDS.items()}
_ASTRA_SM40_PARENT_READ = _RTT_PARENT_READ
def _astra_sm40_read(left,young_right,lam,live_indices,state):
    old=int(state['columns']);m,k,n=int(left.shape[0]),int(young_right.shape[1]),int(young_right.shape[0])
    rank=int(state['pre_basis'].shape[1])if old else 0
    key=(m,k,n,rank);plan=_ASTRA_SM40_PLANS.get(key)
    if (not _ASTRA_SM40_ON or plan is None or not _WINOGRAD or not _AB_ON
            or str(left.dtype)!=str(young_right.dtype)or str(left.dtype)!='float32'
            or str(lam.dtype)!='float32' or _ASTRA_RLF_CONTEXT[0]is not None
            or _UBX_CTX[0]is not None or _UBC_CONTEXT[0]is not None or _UBX_RC[0]is not None):
        return _ASTRA_SM40_PARENT_READ(left,young_right,lam,live_indices,state)
    hold=_E2Hold();old_product=None
    if old:
        inner=hold(_cp_slr_inner_sq(hold(left[:,:old]),hold(lam[:old]),state))
        right=hold(hold(_xsg_rows(5119,state['pre_basis'],live_indices)).T)
        old_product=(inner,right)
    a=hold(left[:,old:]);b=hold(young_right.T);young_lam=hold(lam[old:])
    if m<n:a=hold(a*young_lam)
    else:b=hold(hold(young_right*young_lam).T)
    name,L,P=plan;kernel=_ASTRA_SM40_KERNELS[name]
    result=kernel(a,b,L,P,old_product)
    _ASTRA_SM40_STATS['calls']+=1;_ASTRA_SM40_STATS['shapes'].append(key)
    _ASTRA_SM40_STATS['max_workspace_bytes']=max(_ASTRA_SM40_STATS['max_workspace_bytes'],kernel.stats['conservative_peak_bytes'])
    return result
_RTT_PARENT_READ = _astra_sm40_read
'''
result=source.read_text()+prefix+'\n_ASTRA_SM40_RECORDS = '+repr(constants)+'\n_ASTRA_SM40_PLANS = '+repr(plans)+'\n'+body+'\n'+suffix
target=D/'candidate_smirnov_owned8_HELD_v2.py';compile(result,str(target),'exec')
with target.open('x')as f:f.write(result)
manifest=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate=str(target),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((D/'smirnov_batched_v1.py').read_bytes()).hexdigest(),plans=len(plans),source_price=prices['summary'],scope='Inactive source build only. Exact local circuit; explicit positive-shape dispatch; unknowns use exact incumbent parent. No import/setup/predict/native run or live file changes.')
with(D/'BUILD_v2.json').open('x')as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(manifest,indent=2))
