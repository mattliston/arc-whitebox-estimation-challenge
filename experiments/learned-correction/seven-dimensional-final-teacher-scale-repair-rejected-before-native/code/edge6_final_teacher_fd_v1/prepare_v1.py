"""Freeze one truth-free seven-direction native teacher Jacobian. No launch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'HANDOFF_CURRENT.md').exists())
A = D.parent / 'fable_oct19_kc_v1/agent_astra_joint_v1'
SCI = D / 'edge_hidden6_nonlinear_v1/native_hidden6_v1/candidate_msg8_edge_hidden6_nonlinear_science_v1.py'
SCI_SHA = '5b5f5f3e3e7609d788047732f3f81aa1ef3ad7449854f8f3163d84273e6e2a65'
EDGE8 = D / 'edge_hidden_nonlinear_v1/native_hidden8_v1/candidate_msg8_edge_hidden8_nonlinear_science_v1.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p, value):
    with p.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
def nodes(path):
    tree=ast.parse(path.read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    assignments={t.id:n.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    return functions,assignments

def main():
    assert sha(SCI)==SCI_SHA
    assert sha(EDGE8)=='627c79b74be1747e15de3e1d8bf22acb4a3838fa168e3c55c04179c0a3bdc992'
    fs,aa=nodes(SCI);f8,a8=nodes(EDGE8)
    for name in ('_message_edge_source','_ps32'):
        assert ast.dump(fs[name])==ast.dump(f8[name]),name
    original=ast.literal_eval(aa['_GN_EDGE'])
    assert original==ast.literal_eval(a8['_GN_EDGE']) and set(original)==set(range(8,15))
    proposal=D/'edge_final_teacher_scale_v1/PROPOSAL_v1.json'
    spec=json.loads(proposal.read_text())
    assert spec['fixed_fd_design']['h']==.02 and spec['fixed_fit']['ridge_lambda']==.1
    proof=D/'edge_final_teacher_scale_v1/LIVE_BINDING_v1.json'
    assert sha(proof)==spec['source_binding']['identity_receipt']['sha256']
    assert json.loads(proof.read_text())['cache_restore_exact']
    parent=A/'cloud_linux_markline_v1/PLAN_MSG8_LOADERFIX_CONTROL_v1.json'
    plan=json.loads(parent.read_text())
    plan.update(utc=utc(),kind='EDGE6_FINAL_TEACHER_FD_TRAIN4_ONLY',groups=[[35,150,200],[250]],
                first7=[35,150,200,250],held25=[],candidates={})
    # Inference needs weights only. Deliberately remove every truth record.
    plan['rows']={str(rank):{'weights':plan['rows'][str(rank)]['weights']} for rank in [35,150,200,250]}
    records=[]
    for layer in range(8,15):
        for label,delta in [('minus',-.02),('plus',.02)]:
            name=f'edge6_teacher_fd_l{layer}_{label}_v1'
            scales=dict(original);scales[layer]=(1+original[layer])*(1+delta)-1
            assert scales[layer]!=0
            tail=(f'# Fixed truth-free teacher FD layer{layer}, scale{1+delta!r}.\n'
                  + '_GN_EDGE = '+repr(scales)+'\n_FAST_RESCUE_FRAC = {}\n')
            target=HERE/(name+'_tail.py')
            with target.open('x') as f:f.write(tail)
            effective=SCI.read_bytes()+b'\n'+tail.encode()
            tree=ast.parse(effective.decode());assert ast.literal_eval(tree.body[-1].value)=={}
            rec=dict(path=str(SCI.relative_to(ROOT)),sha256=SCI_SHA,
                append_tail=dict(path=str(target.relative_to(ROOT)),sha256=sha(target)),
                effective_sha256=hashlib.sha256(effective).hexdigest())
            plan['candidates'][name]=rec
            records.append(dict(name=name,layer=layer,delta=delta,scales=scales,**rec))
    plan['teacher_fit']=dict(teacher_sha256=spec['teacher_source_sha256'],student_sha256=SCI_SHA,
        no_truth_targets=True,no_held25_training=True,train=[35,150,200,250],validation=[259,277,287],
        proposal_sha256=sha(proposal),new_predictions=56,baseline_reused_only_with_matching_phase_groups_and_closed_guards=True)
    output=HERE/'PLAN_v1.json';write(output,plan)
    write(HERE/'BUILD_v1.json',dict(utc=utc(),plan_sha256=sha(output),parent_plan_sha256=sha(parent),
        source_sha256=SCI_SHA,proposal_sha256=sha(proposal),live_binding_receipt_sha256=sha(proof),
        live_wrapper_and_cache_AST_identical_to_tiny_proof=True,records=records,
        script_sha256=sha(__file__),no_native_launch=True))
    print(json.dumps(dict(plan=str(output),sha256=sha(output),arms=len(records),new_predictions=56)))

if __name__=='__main__':main()
