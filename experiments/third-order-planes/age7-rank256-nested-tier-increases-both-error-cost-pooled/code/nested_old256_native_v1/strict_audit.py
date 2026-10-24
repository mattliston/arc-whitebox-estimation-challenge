"""Tighten nonzero-reference unit normalization without changing any fixture."""
from common import *
seal_check()
original=(HERE/'unit.py').read_text()
old="return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1.0))"
new="return float(np.linalg.norm(a-b)/(float(np.linalg.norm(b)) if float(np.linalg.norm(b)) > 0 else 1.0))"
edits=[(old,new),("HERE/'unit.json'","HERE/'strict_unit.json'"),
       ("Frobenius difference / max(1, reference Frobenius norm)",
        "true relative Frobenius error for nonzero references; absolute Frobenius error for exact zero")]
source=original
for a,b in edits:
    assert source.count(a)==1
    source=source.replace(a,b)
inverse=source
for a,b in reversed(edits):
    assert inverse.count(b)==1
    inverse=inverse.replace(b,a)
assert inverse==original
write(HERE/'strict_audit_build.json',dict(source=pin(HERE/'unit.py'),audit=pin(__file__),whole_source_inverse=True,
    reason='Original unit normalization used max(1,norm). This extra check enforces actual relative error on every nonzero reference; original output is preserved.',
    same_fixtures_and_threshold=True,changes=[dict(before=a,after=b) for a,b in edits]))
exec(compile(source,str(HERE/'unit.py'),'exec'),dict(__name__='__main__',__file__=str(HERE/'unit.py')))
