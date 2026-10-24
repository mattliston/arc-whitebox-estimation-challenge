"""Same centered channel with all new vector handles retained by existing E2 hold."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'candidate_95c_centered22_HELD_v1.py';s=p.read_text();lines=s.splitlines(keepends=True)
assert hashlib.sha256(s.encode()).hexdigest()=='a43b84775278be8dc1db0584ab2ceb597ee5526eab968acea97a556d0bb1def4'
assert '_o18_scalar_bc = bb - ((2 * m) / s) * aa' in lines[893]
lines[893]="    FT, GT = st['FT'], st['GT']; _o18_scalar_m2 = 2 * m; _e2_H.append(_o18_scalar_m2); _o18_scalar_ratio = _o18_scalar_m2 / s; _e2_H.append(_o18_scalar_ratio); _o18_scalar_product = _o18_scalar_ratio * aa; _e2_H.append(_o18_scalar_product); _o18_scalar_bc = bb - _o18_scalar_product; _e2_H.append(_o18_scalar_bc)  # Center only channel 22, retaining new vector handles in existing batch hold.\n"
out=''.join(lines);assert len(out.splitlines())==len(s.splitlines());ast.parse(out)
f=D/'candidate_95c_centered22_HELD_v2.py'
with f.open('x')as g:g.write(out)
r=dict(parent_sha256='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf',candidate=str(f),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),status='BUILT_STATIC_AND_SYNTHETIC_ONLY_NOT_NATIVE_OR_FIRE_READY',change_from_v1='Retain all four new vector expression outputs in existing E2 hold, preventing three unbatched temporary frees; arithmetic unchanged.',saved_flops_14layers=94968832,estimated_numeric_call_delta_per_layer=-2,banked=False)
with (D/'BUILD_CENTERED22_v2.json').open('x')as g:json.dump(r,g,indent=2);g.write('\n')
print(json.dumps(r,indent=2))
