"""Tiny synthetic native edge identity/accounting test; never imports full estimator.
Uses S46 _ecut_raw and _e2_edge_consts AST extracts on 4x4 arrays under flopscope.
Numpy appears only in fixture generation and offline comparisons, not fragment.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import flopscope as flops
from edband_build_v2 import prepare, normalize, signatures

HERE = Path(__file__).resolve().parent
fixture = json.loads((HERE / 'fixtures/identity_s46.json').read_text())
parent = Path(fixture['source']['native']).read_text()
assert hashlib.sha256(parent.encode()).hexdigest() == fixture['parent_native_sha256']
tree = ast.parse(parent)
extracts = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('_ecut_raw', '_e2_edge_consts'):
        extracts[node.name] = ast.get_source_segment(parent, node)
assert len(extracts) == 2

class Hold:
    def __call__(self, value):
        return value

fnp = flops.numpy
rng = np.random.default_rng(56)
left_np, right_np = [rng.normal(size=(4,16)).astype(np.float32) for _ in range(2)]
edge_np = rng.normal(size=(3,16)).astype(np.float32)
R_np, S_np = [rng.normal(size=(4,4)).astype(np.float32) for _ in range(2)]
outputs = {}
reports = {}
for mode in ('off', 'identity', 'changed'):
    values = dict(fixture)
    if mode == 'off':
        values.update(edges=[],bias=None,output=None)
    elif mode == 'changed':
        values['bias'] = [list(v) for v in fixture['bias']]
        values['output'] = [list(v) for v in fixture['output']]
        values['bias'][1][3] += .125
        values['output'][2][7] -= .25
    edits,fragment,meta = prepare(parent,values)
    assert len(edits) == 1 and signatures(fragment) == []
    ns=dict(fnp=fnp,_E2Hold=Hold,_ECUT_STANDIN=True,_ECUT_C1=.165,_ECUT_C2=.0098,
            _ECUT_STATS=dict(layers=0,kept=0,full=0),_E2_EDGE_STATE={},_CS_DICTS=())
    ns['_cs_build'] = lambda n=1024,depth=16: None
    for text in extracts.values():
        exec(compile(text,'<S46 tiny edge extraction>','exec'),ns)
    exec(compile(fragment,'<EDBAND fragment>','exec'),ns)
    with flops.BudgetContext(flop_budget=10**7,quiet=True) as ctx:
        shared = {key:fnp.asarray(fixture[key][0],dtype=fnp.float32) for key in ('bias','output')}
        shared['edge']=fnp.asarray(edge_np,dtype=fnp.float32)
        ns['_EDGE_PARAMETERS']=shared
        left,right=[fnp.asarray(v,dtype=fnp.float32) for v in (left_np,right_np)]
        pf=fnp.asarray([.3,.2,.4,.1],dtype=fnp.float32)
        gf=fnp.asarray([.8,.9,.7,.6],dtype=fnp.float32)
        ns['_E2_LAYER']=dict(ecut_rows=fnp.asarray([0,1,3]),ecut_cols=fnp.asarray([0,2,3]))
        k=ns['_e2_edge_consts'](shared,4)
        fnp.copyto(k['R'],fnp.asarray(R_np,dtype=fnp.float32))
        fnp.copyto(k['S'],fnp.asarray(S_np,dtype=fnp.float32))
        cached_R,cached_S=k['R'],k['S']
        start=len(ctx._op_log)
        ns['_cs_build'](4,16)
        setup=len(ctx._op_log)-start
        start=len(ctx._op_log)
        ns['_cs_build'](4,16)
        repeated_setup=len(ctx._op_log)-start
        row_outputs,op_counts,selection_counts=[],[],[]
        for layer in list(range(8,15))*2:
            start=len(ctx._op_log)
            got_k=ns['_e2_edge_consts'](shared,4)
            selected=ns['_edge_band_parameters'](layer,shared)
            selection_counts.append(len(ctx._op_log)-start)
            assert got_k is k and k['R'] is cached_R and k['S'] is cached_S and k['t'] is shared
            assert selected['edge'] is shared['edge']
            start=len(ctx._op_log)
            result=ns['_ecut_raw'](left,right,selected,k,pf,gf,[])
            op_counts.append(len(ctx._op_log)-start)
            row_outputs.append(np.asarray(result).copy())
        assert ns['_edge_band_parameters'](7,shared) is shared
        assert ns['_edge_band_parameters'](15,shared) is shared
    assert setup == (0 if mode=='off' else 6),(mode,setup)
    assert repeated_setup==0 and selection_counts==[0]*14,(mode,repeated_setup,selection_counts)
    outputs[mode]=row_outputs
    reports[mode]=dict(setup_requests=setup,repeated_setup_requests=repeated_setup,
                       per_layer_selection_requests=selection_counts,edge_requests=op_counts,
                       shared_cache_identity=True,payload_bytes=meta['payload_bytes'])
assert all(np.array_equal(a,b) for a,b in zip(outputs['off'],outputs['identity']))
assert reports['off']['edge_requests']==reports['identity']['edge_requests']==reports['changed']['edge_requests']
for i,l in enumerate(list(range(8,15))*2):
    equal=np.array_equal(outputs['off'][i],outputs['changed'][i])
    assert equal == (l<=9),(i,l,equal)
for bad in (dict(fixture,edges=[[8,10],[10,14]]),dict(fixture,bias=[[0]*15]*3),dict(fixture,output=[[float('nan')]*16]*3)):
    try:
        normalize(bad)
    except (AssertionError,ValueError,OverflowError):
        pass
    else:
        raise AssertionError('invalid specification accepted')
print(json.dumps(dict(status='PASS',scope='4x4 synthetic edge only, no full native session',
                      identity_bit_exact=True,new_einsum_signatures=[],reports=reports),indent=2))
