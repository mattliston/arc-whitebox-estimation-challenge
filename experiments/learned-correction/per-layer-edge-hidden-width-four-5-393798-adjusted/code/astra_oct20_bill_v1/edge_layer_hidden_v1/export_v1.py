"""Export two fixed per-layer edge maps without adding hot arithmetic."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent;D=HERE.parent
V=D.parent/'astra_oct20_vm_frontier_v1'
RUNS=Path('/home/arcworker/arc-runs')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,s):
    with Path(p).open('x') as f:f.write(s)
def jwrite(p,x):write(p,json.dumps(x,indent=2,allow_nan=False)+'\n')
fit=read(RUNS/'edge_layer_hidden2_4_fit_v1/FITTED_v1.json')
guard=read(RUNS/'edge_layer_hidden2_4_fit_v1_GUARD.json')
assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
assert guard['max_owned_swap_bytes']==0 and guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
assert fit['widths']==[2,4] and not fit['uses_truth'] and not fit['validation_selection']
parents={
 'science':V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py',
 'release':V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_rel_v1.py'}
assert sha(parents['science'])=='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
assert sha(parents['release'])=='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
defs={n.name:n for n in ast.parse(parents['science'].read_text()).body if isinstance(n,ast.FunctionDef)}
base=defs['_g5_message_edge_base'];body=[];replacements=[]
for node in base.body:
    if isinstance(node,ast.Global) and node.names==['_EDGE_PARAMETERS']:
        replacements.append('remove global');continue
    if isinstance(node,ast.If) and ast.unparse(node.test)=='_EDGE_PARAMETERS is None':
        replacements.append('remove shared parameter creation');continue
    if isinstance(node,ast.Assign) and ast.unparse(node)=='t = _EDGE_PARAMETERS':
        body.extend(ast.parse('t, context = _pl_edge_parameters(layer)').body);replacements.append('layer parameters');continue
    if isinstance(node,ast.Assign) and ast.unparse(node)=='k = _e2_edge_consts(t, n)':
        body.extend(ast.parse('k = _pl_edge_consts(t, n, layer)').body);replacements.append('layer scratch');continue
    body.append(node)
assert len(replacements)==4,replacements
base.body=body;ast.fix_missing_locations(base)
kernel=ast.unparse(base)
helpers="""
def _pl_edge_parameters(layer):
    got = _PL_EDGE_PARAMETERS.get(layer)
    if got is None:
        t = {key: fnp.asarray(value, dtype=fnp.float32) for key, value in _PL_EDGE_LITERALS[layer].items()}
        context = {key: fnp.asarray(value, dtype=fnp.float32) for key, value in _PL_MESSAGE_LITERALS[layer].items()}
        got = (t, context)
        _PL_EDGE_PARAMETERS[layer] = got
    return got

def _pl_edge_consts(t, n, layer):
    key = (n, layer)
    k = _PL_EDGE_CONSTS.get(key)
    if k is not None and k['t'] is t:
        return k
    R = fnp.empty((n, n), dtype=fnp.float32)
    S = fnp.empty((n, n), dtype=fnp.float32)
    e0, e1, e2 = fnp.unstack(t['edge'])
    k = dict(t=t, R=R, S=S, R3=R[:, :, None], S3=S[:, :, None], ST3=S.T[:, :, None], e0=e0, e1=e1, e2=e2)
    _PL_EDGE_CONSTS[key] = k
    return k

def _message_edge_parameters():
    return _pl_edge_parameters(8)[1]
"""
records=[]
for width in fit['widths']:
    rows=[r for r in fit['results'] if r['width']==width]
    assert {r['layer'] for r in rows}==set(range(8,15))
    el={r['layer']:r['edge_literal'] for r in rows}
    ml={r['layer']:r['message_edge_literal'] for r in rows}
    assert all(len(v['output'])==width and len(ml[k]['left'][0])==width for k,v in el.items())
    out=HERE/('native_hidden%d_v1'%width);out.mkdir(exist_ok=False)
    tail='\n# Frozen per-layer nonlinear edge hidden%d, actual777a teacher, train4 only.\n'%width
    tail+='assert _EDGE_PARAMETERS is None and _MESSAGE_EDGE_PARAMETERS is None and not _E2_EDGE_STATE\n'
    tail+='assert _GNN_PARAMS is None and not _E2_GNN_STATE and _VMFR_CM == 8\n'
    tail+='_O20_EDGE_HIDDEN = '+str(width)+'\n'
    tail+='_PL_EDGE_LITERALS = '+repr(el)+'\n_PL_MESSAGE_LITERALS = '+repr(ml)+'\n'
    tail+='_PL_EDGE_PARAMETERS = {}\n_PL_EDGE_CONSTS = {}\n'
    tail+='_EDGE_LITERAL = _PL_EDGE_LITERALS[8]\n_MESSAGE_EDGE_LITERAL = _PL_MESSAGE_LITERALS[8]\n'
    tail+=helpers+'\n'+kernel+'\n_gn_message_edge_source_base = _g5_message_edge_base\n_FAST_RESCUE_FRAC = {}\n'
    tp=out/('tail_edge_layer_hidden%d_v1.py'%width);write(tp,tail)
    record=dict(width=width,tail=str(tp),tail_sha256=sha(tp),outputs={},
        fit_sha256=sha(RUNS/'edge_layer_hidden2_4_fit_v1/FITTED_v1.json'),
        guard_receipt_sha256=sha(RUNS/'edge_layer_hidden2_4_fit_v1_GUARD.json'),
        metrics=fit['aggregate'][str(width)],source_body_replacements=replacements,
        extra_hot_arithmetic=0,additional_scratch_upper_bytes=6*2*1024*1024*4,
        per_layer_private_parameters_and_scratch=True,no_runtime_governors=True,no_native_gain_claim=True)
    for role,parent in parents.items():
        path=out/('candidate_edge_layer_hidden%d_%s_v1.py'%(width,role))
        source=parent.read_text()+tail;compile(source,str(path),'exec')
        assert ast.unparse(ast.parse(source).body[-1])=='_FAST_RESCUE_FRAC = {}'
        write(path,source)
        record['outputs'][role]=dict(path=str(path),sha256=sha(path),parent_path=str(parent),parent_sha256=sha(parent))
    jwrite(out/'BUILD_v1.json',record);records.append(record)
jwrite(HERE/'EXPORTED_v1.json',dict(utc=utc(),script_sha256=sha(__file__),records=records,
    official_tiny_and_full_import_required=True,no_native_launch=True))
print(json.dumps([dict(width=r['width'],outputs=r['outputs'],metrics=r['metrics']) for r in records],indent=2))

