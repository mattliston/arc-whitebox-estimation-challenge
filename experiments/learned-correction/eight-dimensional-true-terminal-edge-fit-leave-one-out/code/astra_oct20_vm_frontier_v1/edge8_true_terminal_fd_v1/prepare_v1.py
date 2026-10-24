"""Freeze exactly eight shared edge-output directions, not layer scales."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
V = HERE.parent
D = V.parent / 'astra_oct20_bill_v1'
ROOT = next(p for p in HERE.parents if (p / 'HANDOFF_CURRENT.md').exists())
A = V.parent / 'fable_oct19_kc_v1/agent_astra_joint_v1'
SCI = D / 'edge_hidden_nonlinear_v1/native_hidden8_v1/candidate_msg8_edge_hidden8_nonlinear_science_v1.py'
SCI_SHA = '627c79b74be1747e15de3e1d8bf22acb4a3838fa168e3c55c04179c0a3bdc992'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
TRAIN = [35, 150, 200, 250]
VALID = [259, 277, 287]
def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
def main():
    assert sha(SCI) == SCI_SHA
    text = SCI.read_text()
    tree = ast.parse(text)
    aa = {t.id:n.value for n in tree.body if isinstance(n, ast.Assign)
          for t in n.targets if isinstance(t, ast.Name)}
    edge = ast.literal_eval(aa['_EDGE_LITERAL'])
    out = edge['output']
    assert len(out) == 8 and all(math.isfinite(v) and v != 0 for v in out)
    rms = math.sqrt(sum(v*v for v in out)/8)
    scales = [max(abs(v), rms) for v in out]
    spec = dict(utc=utc(), status='PREREGISTERED_BEFORE_FD_AND_TRUTH',
        parent_source_sha256=SCI_SHA, valid_reference_submission=337422,
        train=TRAIN, validation=VALID, no_held25_or_fresh_fit=True,
        hypothesis='Eight shared nonlinear edge output coefficients on the already cheap current edge8; exact native final-error objective, not teacher imitation or per-layer scalar calibration.',
        parameters='Only _EDGE_LITERAL[output][0:8], shared over layers8..14. All hidden/context/GNN/birth/physics parameters fixed.',
        original_output=out, coordinate_scales=scales, h=0.1,
        direction='output_j := original_j +/- h * max(abs(original_j), RMS(original_output)); all other coefficients unchanged.',
        predictions=64, groups=[[35,150,200],[250]], blas_threads=4,
        baseline_tag='msg8_edge_hidden8_nonlinear_blas4_v1',
        reuse='Only exact same source/BLAS/weights/order/scored phase with successfully retired guards. Group2 uses rank250 as its first cold network.',
        objective='True final-layer independent-half cross-product U, equal-network relative-U weighting. Ridge fits midpoint residual; reported proxy uses actual A/B cross-products.',
        ridge_lambda=0.1, damping=0.5,
        derivative='J_j=(y_plus-y_minus)/2 per FD-coordinate unit; Q_j=(y_plus+y_minus-2*y0)/2, all at L15.',
        conditioning='TRAIN-only equal-network relative-U scaled RMS column normalization; one normalized ridge .1; no intercept.',
        trust='One deterministic radial shrink after damping: ||t||2<=1 (inside sampled FD radius), RMS(Jt)<=.25*RMS(midpoint-y0), and sum_j RMS(Q_j)*t_j^2 <= .25*RMS(Jt). All use TRAIN-only relative-U metric. No coefficient-norm threshold.',
        curvature='Report every direction and network; the proposed-step second-order bound is the physical curvature gate. Cross-direction curvature is unmeasured and must be checked by actual fixed-candidate predictions.',
        loo='Four leave-one-network-out solves, rebuilding scales/trust with the other3 networks. Descriptive true-U proxy for the held TRAIN network.',
        promotion='Freeze the one fit before any validation3 truth/prediction access. Require train linear proxy <=-.25%, aggregate LOO<0 and at least3/4 LOO networks improve; otherwise close without export. Qualified one fit receives actual train4+validation3 feedback checks before scientific32. No second fit or h/ridge/damping/feature search.',
        billing='Eight output literals only. Same operation graph, deployed bill expected identical except any downstream numerical roster change; must measure actual32.',
        history='Distinct from closed layer-output TEACHER ridge and seven-layer final-TEACHER scalar FD. Historical trained covariance edge is useful; current compressed edge output coefficients have not been optimized on true final error.')
    write(HERE/'PREREG_v1.json', spec)
    parent = A/'cloud_linux_markline_v1/PLAN_MSG8_LOADERFIX_CONTROL_v1.json'
    oldplan = json.loads(parent.read_text())
    plan = dict(oldplan)
    plan.update(utc=utc(), kind='EDGE8_OUTPUT8_TRUE_TERMINAL_FD_TRAIN4_ONLY',
                groups=spec['groups'], first7=TRAIN, held25=[], candidates={})
    plan['rows'] = {str(r):{'weights':oldplan['rows'][str(r)]['weights']} for r in TRAIN}
    records = []
    for j in range(8):
        for sign in [-1,1]:
            label = 'minus' if sign < 0 else 'plus'
            name = f'edge8_terminal_fd_c{j}_{label}_v1'
            values = list(out)
            values[j] += sign * spec['h'] * scales[j]
            tail = ('# Frozen output8 true-terminal FD; TRAIN4 only.\n'
                + 'assert _EDGE_PARAMETERS is None and not _E2_EDGE_STATE\n'
                + 'assert _EDGE_LITERAL[\'output\'] == '+repr(out)+'\n'
                + '_EDGE_LITERAL[\'output\'] = '+repr(values)+'\n'
                + '_FAST_RESCUE_FRAC = {}\n')
            p = HERE/(name+'_tail.py')
            with p.open('x') as f: f.write(tail)
            effective = SCI.read_bytes()+b'\n'+tail.encode()
            assert ast.unparse(ast.parse(effective.decode()).body[-1]) == '_FAST_RESCUE_FRAC = {}'
            rec = dict(path=str(SCI.relative_to(ROOT)), sha256=SCI_SHA,
                append_tail=dict(path=str(p.relative_to(ROOT)),sha256=sha(p)),
                effective_sha256=hashlib.sha256(effective).hexdigest())
            plan['candidates'][name] = rec
            records.append(dict(name=name,index=j,sign=sign,output=values,**rec))
    plan['fd_prereg'] = dict(path=str((HERE/'PREREG_v1.json').relative_to(ROOT)),
        sha256=sha(HERE/'PREREG_v1.json'), train=TRAIN, no_truth_in_worker_plan=True,
        new_predictions=64, no_validation_in_fd=True)
    write(HERE/'PLAN_v1.json',plan)
    # This fitter-only record contains exactly TRAIN4 truth pointers, never held25.
    truth = {str(r):{k:v for k,v in oldplan['rows'][str(r)].items() if k in ('truth_A','truth_B')} for r in TRAIN}
    assert all(set(x)=={'truth_A','truth_B'} for x in truth.values())
    write(HERE/'FIT_INPUTS_v1.json',dict(utc=utc(),truth=truth,train=TRAIN,
        source_plan_sha256=sha(parent),baseline_tag=spec['baseline_tag'],
        no_truth_contents_read=True))
    write(HERE/'BUILD_v1.json',dict(utc=utc(),plan_sha256=sha(HERE/'PLAN_v1.json'),
        source_sha256=SCI_SHA,parent_plan_sha256=sha(parent),
        prereg_sha256=sha(HERE/'PREREG_v1.json'),fit_inputs_sha256=sha(HERE/'FIT_INPUTS_v1.json'),
        records=records,script_sha256=sha(__file__),no_native_launch=True,
        no_truth_contents_read=True))
    print(json.dumps(dict(utc=utc(),plan_sha256=sha(HERE/'PLAN_v1.json'),
        arms=16,new_predictions=64,prereg_sha256=sha(HERE/'PREREG_v1.json'))))
if __name__ == '__main__': main()

