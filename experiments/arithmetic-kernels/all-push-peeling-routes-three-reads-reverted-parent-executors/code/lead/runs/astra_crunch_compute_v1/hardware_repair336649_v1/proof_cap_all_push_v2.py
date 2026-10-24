"""Source-extracted routing/cap proof, no arrays, SDK, or model execution."""
import ast,copy,hashlib,itertools,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
SOURCE=R/'astra_crunch_runtime_v1/failure336649_v1/candidate_repair_v21a.py'
FRAGMENT=HERE/'fragment_all_push_proof_only_v1.py.txt'
src=SOURCE.read_text();nodes=ast.parse(src).body
names=('_PEEL_CALL_COSTS','_PEEL_PUSH_PLANS','_PEEL_READ_PLANS')
tables={}
for node in nodes:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in names:
                tables[target.id]=ast.literal_eval(node.value)
tables['_PEEL_PUSH_PLANS'].pop((736,768,6400),None)
take=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_peel_take')
push=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_pushsm_mm')
read=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_peel_read')
def make(patched):
    e=copy.deepcopy(tables)
    e.update(_PEEL_CALL_BUDGET=dict(left=750,spent=0,accepted=0,rejected=0),
             _AB_ON=True,_WINOGRAD=True,_SMX_ON=True,_SCRATCH=[object()],
             _ASTRA_RLF_CONTEXT=[None],_UBX_CTX=[None],_UBC_CONTEXT=[None],_UBX_RC=[None],
             _PEEL_STATS=dict(push=0,push_shapes=[],read=0,read_shapes=[]))
    e['_PEEL_PARENT_PUSH']=lambda *args:'parent'
    e['_PEEL_PARENT_READ']=lambda *args:'parent'
    e['_peel_product']=lambda *args:'peel'
    def entered_read():raise EnteredRead
    e['_peel_bind_read_io']=entered_read
    exec(compile(ast.Module(body=[take,push,read],type_ignores=[]),str(SOURCE),'exec'),e)
    if patched:exec(compile(FRAGMENT.read_text(),str(FRAGMENT),'exec'),e)
    return e
class EnteredRead(Exception):pass
class Array:
    def __init__(self,shape,dtype='float32'):self.shape=shape;self.dtype=dtype

plain,patched=make(False),make(True)
tests=0
keys=list(tables['_PEEL_CALL_COSTS'])+[('push',(1,2,3)),('read',(1,2,3,0))]
for left,key in itertools.product(range(751),keys):
    start=dict(left=left,spent=750-left,accepted=4,rejected=3)
    for e in(plain,patched):e['_PEEL_CALL_BUDGET'].update(start)
    a=plain['_peel_take'](*key);b=patched['_peel_take'](*key)
    assert plain['_PEEL_CALL_BUDGET']==patched['_PEEL_CALL_BUDGET']
    assert b==(a and key[0]!='push' and key not in patched['_HW_REPAIR_SLOW_ROUTES_V1'])
    tests+=1
sequence_tests=0
for seq in (keys,keys[::-1],keys*3):
    for e in(plain,patched):e['_PEEL_CALL_BUDGET'].update(left=750,spent=0,accepted=0,rejected=0)
    for key in seq:
        plain['_peel_take'](*key);patched['_peel_take'](*key)
        assert plain['_PEEL_CALL_BUDGET']==patched['_PEEL_CALL_BUDGET']
        sequence_tests+=1
routing=[]
for kind,key in sorted(patched['_HW_REPAIR_SLOW_ROUTES_V1']):
    gates=['enabled','AB_OFF','WINOGRAD_OFF','SMX_OFF','no_arena','dtype64','cap_reject']
    if kind=='read':gates+=['lam64','mismatch','RLF','UBX','UBC','UBX_RC']
    for gate in gates:
        outputs=[];budgets=[]
        for changed in(False,True):
            e=make(changed);m,k,n=key[:3]
            if gate=='AB_OFF':e['_AB_ON']=False
            if gate=='WINOGRAD_OFF':e['_WINOGRAD']=False
            if gate=='SMX_OFF':e['_SMX_ON']=False
            if gate=='no_arena':e['_SCRATCH']=None
            if gate=='cap_reject':e['_PEEL_CALL_BUDGET']['left']=0
            for label,name in [('RLF','_ASTRA_RLF_CONTEXT'),('UBX','_UBX_CTX'),('UBC','_UBC_CONTEXT'),('UBX_RC','_UBX_RC')]:
                if gate==label:e[name][0]=object()
            dtype='float64'if gate=='dtype64'else'float32'
            try:
                if kind=='push':out=e['_pushsm_mm'](Array((m,k),dtype),Array((k,n),dtype))
                else:out=e['_peel_read'](Array((m,k),dtype),Array((n,k),'float64'if gate=='mismatch'else dtype),Array((k,),'float64'if gate=='lam64'else dtype),Array((n,)),dict(columns=0))
            except EnteredRead:out='peel'
            outputs.append(out);budgets.append(dict(e['_PEEL_CALL_BUDGET']))
        assert budgets[0]==budgets[1]
        expect_peel = gate=='enabled' or (gate=='cap_reject' and tables['_PEEL_CALL_COSTS'][(kind,key)]==0)
        assert outputs==(['peel','parent'] if expect_peel else ['parent','parent']),(kind,key,gate,outputs)
        routing.append(dict(kind=kind,key=key,gate=gate,outputs=outputs,cap=budgets[0]))
result=dict(status='PASS',models=0,array_operations=0,source=str(SOURCE),source_sha256=hashlib.sha256(src.encode()).hexdigest(),
    fragment_sha256=hashlib.sha256(FRAGMENT.read_bytes()).hexdigest(),exhaustive_cap_cases=tests,
    sequence_steps=sequence_tests,routing_cases=len(routing),routing=routing,
    proof='Original eligibility gates are unchanged. Original _peel_take executes exactly once; all cap state is identical. Only its Boolean return is suppressed for the fixed hardware exclusions. Existing fallback functions perform all numerical work.')
with(HERE/'PROOF_CAP_ALL_PUSH_v2.json').open('x')as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items()if k!='routing'}))
