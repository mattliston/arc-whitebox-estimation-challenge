"""One pinned HK integration: preserve old capacity policy, then append."""
from pathlib import Path
import ast
import hashlib
import json

D=Path(__file__).resolve().parent
H2=D.parent
PARENT=H2/'hill_climb_exact_bundle_v1/candidate.py'
PARENT_SHA='3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
ORDER=['constants.py','kernel_blocked.py','setup.py','dispatcher.py']
CAPACITY_EDITS=[
    ('if tail + numel <= int(arena.shape[0]):',
     'if tail + numel <= min(391602176, int(arena.shape[0])):'),
    ('tail + 7 * (u_stop - u_start) // 3 <= int(arena.shape[0])',
     'tail + 7 * (u_stop - u_start) // 3 <= min(391602176, int(arena.shape[0]))'),
]


def pin(p):
    data=p.read_bytes()
    return dict(path=str(p.resolve()),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    frozen=json.loads((D/'assembly_inputs.json').read_text())
    assert frozen['order']==ORDER
    assert pin(PARENT)['sha256']==PARENT_SHA
    fragments=[]
    for name in ORDER:
        p=D/name
        assert pin(p)==frozen['files'][name],name
        text=p.read_text()
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert all(a.name not in ('numpy','scipy','os','sys') for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                assert node.module not in ('numpy','scipy','os','sys')
        fragments.append(p.read_bytes())
    parent=PARENT.read_bytes()
    prefix=parent.decode()
    for before,after in CAPACITY_EDITS:
        assert prefix.count(before)==1
        prefix=prefix.replace(before,after)
    inverse=prefix
    for before,after in reversed(CAPACITY_EDITS):
        assert inverse.count(after)==1
        inverse=inverse.replace(after,before)
    assert inverse.encode()==parent
    candidate=prefix.encode()+b'\n\n'+b'\n\n'.join(fragments)
    ast.parse(candidate)
    assert candidate.startswith(prefix.encode())
    dest=D/'candidate.py'
    with dest.open('xb') as f:f.write(candidate)
    result=dict(source_only=True,numerical_execution=False,parent=pin(PARENT),candidate=pin(dest),
                inputs=frozen,assembly_inputs=pin(D/'assembly_inputs.json'),builder=pin(Path(__file__)),
                capacity_edits=[dict(before=b,after=a) for b,a in CAPACITY_EDITS],
                old_capacity_policy_preserved=391602176,patched_prefix_inverse_exact=True,
                appended_fragments_only_after_two_metadata_edits=True,qualified=False,submitted=False)
    with (D/'build.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['candidate']))


if __name__=='__main__':main()
