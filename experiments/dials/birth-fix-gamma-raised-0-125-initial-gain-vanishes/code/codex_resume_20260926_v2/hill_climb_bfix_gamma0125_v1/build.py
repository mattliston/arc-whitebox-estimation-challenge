"""Stdlib-only append build; no estimator import or numerical work."""
from pathlib import Path
import ast
import hashlib
import json

D=Path(__file__).resolve().parent
PARENT=D.parent/'hill_climb_precision_g3_symdiag128_v1/candidate.py'
PARENT_SHA='88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
SUFFIX='\n_BFIX_GAMMA = 0.125\n'


def pin(path):
    p=Path(path).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def check(parent,child):
    assert child==parent+SUFFIX and child[:-len(SUFFIX)]==parent
    pt,ct=ast.parse(parent),ast.parse(child)
    assert len(ct.body)==len(pt.body)+1
    assert all(ast.dump(a)==ast.dump(b) for a,b in zip(pt.body,ct.body))
    assignments=[n for n in pt.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='_BFIX_GAMMA' for t in n.targets)]
    assert len(assignments)==1 and ast.literal_eval(assignments[0].value)==.1
    last=ct.body[-1];assert ast.dump(last)==ast.dump(ast.parse('_BFIX_GAMMA = 0.125').body[0])
    body=[n for n in pt.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'
          and any(isinstance(x,ast.Name) and x.id=='_BFIX_GAMMA' for x in ast.walk(n))]
    assert len(body)==1
    source=ast.get_source_segment(parent,body[0])
    assert source.count('_BFIX_GAMMA * _e2_H(radial_new - radial)')==1
    assert source.rfind('means.append(m)',0,source.index('if _BFIX_GAMMA:'))>=0
    assert source.index('if _BFIX_GAMMA:')<source.index('if _S19_BSCALE is not None:')
    return dict(active_predictor_line=body[0].lineno,old_assignment_line=assignments[0].lineno,
                mean_recorded_before_bfix=True,bscale_applies_after_bfix=True,
                dynamic_global=True,all_parent_function_asts_unchanged=True)


def main():
    assert pin(PARENT)['sha256']==PARENT_SHA
    parent=PARENT.read_text();child=parent+SUFFIX;proof=check(parent,child)
    with (D/'candidate.py').open('x') as f:f.write(child)
    record=dict(parent=pin(PARENT),parent_build=pin(PARENT.parent/'build.json'),
        candidate=pin(D/'candidate.py'),builder=pin(__file__),primitive=pin(D/'probe.py'),
        protocol=pin(D/'PROTOCOL.md'),policy=pin(D.parent/'hill_climb_policy_20260928_v2.json'),
        history=pin(D.parent/'teammate_s33_bfix_coordinate_lead_v1.md'),
        append_only=True,suffix=SUFFIX,suffix_bytes=len(SUFFIX.encode()),
        source_inverse_exact=True,all_unlisted_source_bytes_unchanged=True,
        old_value=.1,new_value=.125,source_proof=proof,
        setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),
        source_only=True,numerical_execution=False,execution_authorized=False)
    with (D/'build.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record['candidate']))


if __name__=='__main__':main()
