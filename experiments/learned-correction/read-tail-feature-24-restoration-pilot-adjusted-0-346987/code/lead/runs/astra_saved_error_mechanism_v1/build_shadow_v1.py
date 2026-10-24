from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent
P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
s=P.read_text();assert hashlib.sha256(P.read_bytes()).hexdigest()=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
t=ast.parse(s);node=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_fhm_core')
lines=s.splitlines(True);core=''.join(lines[node.lineno-1:node.end_lineno]);new=core
changes=[
('            pair_map = _nonlinear_first_gaussian',"            if _SH_OBSERVER is not None and l in (4, 5, 6):\n                _SH_OBSERVER('prepair', l, dict(mu=mu, Cpre=Cpre, K21=K21, k4=k4, k22=k22, bpre=bpre, radial=radial))\n            pair_map = _nonlinear_first_gaussian"),
('            if l in _GN_COV_LAYERS:',"            if _SH_OBSERVER is not None and l in (4, 5, 6):\n                _SH_OBSERVER('postpair', l, dict(m=m, C=C, target=target, radial=radial_new, sigma=aux['sigma'], alpha=aux['alpha'], pdf=aux['pdf'], gate=aux['gate']))\n            if l in _GN_COV_LAYERS:"),
('            if _S19_LAM is None:',"            if _SH_OBSERVER is not None and l in (4, 5, 6):\n                _SH_OBSERVER('postnp', l, dict(QX=_NP_STATE['QX'], b=b, gq=gq, t=t, s=s, selected=selected))\n            if _S19_LAM is None:")]
for a,b in changes:assert new.count(a)==1;new=new.replace(a,b)
new_s=''.join(lines[:node.lineno-1])+new+''.join(lines[node.end_lineno:])+(D/'fragment_shadow_capture_v1.py').read_text()
ast.parse(new_s)
out=D/'candidate_shadow_capture_v1.py';out.open('x').write(new_s)
(D/'hook_diff_v1.patch').open('x').write(''.join(difflib.unified_diff(core.splitlines(True),new.splitlines(True),fromfile='parent_core',tofile='capture_core'))+(D/'fragment_shadow_capture_v1.py').read_text())
print(json.dumps(dict(candidate=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),hook_diff=str(D/'hook_diff_v1.patch'))))
