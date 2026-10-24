"""Source-only exclusive build of the fixed terminal marginal completion."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'hill_climb_star_projection32_v1/candidate.py'
SHA='f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
BEGIN='    # BEGIN fixed terminal k3-square raw-moment completion\n'
END='    # END fixed terminal k3-square raw-moment completion\n'


def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    check=HERE/'candidate.py';receipt=HERE/'build.json'
    assert not check.exists() and not receipt.exists()
    parent=PARENT.read_bytes();text=parent.decode();fragment=(HERE/'fragment.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==SHA and len(parent)==930166
    tree=ast.parse(text)
    original=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='nonlinear_scalar'][-1]
    original=ast.get_source_segment(text,original)
    extra=ast.parse(fragment)
    functions=[node for node in extra.body if isinstance(node,ast.FunctionDef)]
    assert len(functions)==1 and functions[0].name=='nonlinear_scalar'
    override=ast.get_source_segment(fragment.decode(),functions[0])
    assert override.count(BEGIN)==override.count(END)==1
    start=override.index(BEGIN);stop=override.index(END)+len(END)
    assert override[:start]+override[stop:]==original
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom,ast.AugAssign)) for node in ast.walk(extra))
    combined=parent+b'\n\n'+fragment
    parsed=ast.parse(combined)
    assert ast.dump(ast.Module(body=parsed.body[:len(tree.body)],type_ignores=[]))==ast.dump(tree)
    assert combined[:-len(fragment)-2]==parent
    protocol=json.loads((HERE/'protocol.json').read_text())
    expected=dict(path=str(check),sha256=hashlib.sha256(combined).hexdigest(),bytes=len(combined))
    assert protocol['parent']==pin(PARENT) and protocol['fragment']==pin(HERE/'fragment.py')
    assert protocol['candidate']==expected
    with check.open('xb') as output:output.write(combined)
    result=dict(passed=True,parent=pin(PARENT),fragment=pin(HERE/'fragment.py'),candidate=pin(check),
        builder=pin(Path(__file__).resolve()),protocol=pin(HERE/'protocol.json'),
        exact_parent_prefix=True,exact_inverse=True,original_scalar_arithmetic_preserved=True,
        only_appended_override='nonlinear_scalar',numerical_execution=False)
    with receipt.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
