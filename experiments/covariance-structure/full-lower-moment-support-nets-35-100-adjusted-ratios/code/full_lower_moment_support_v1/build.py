"""Source-only targeted support change with whole-source byte inverse."""
import ast
from common import HERE,PARENT,PARENT_SHA,sha,write
assert sha(PARENT)==PARENT_SHA
source=PARENT.read_text();tree=ast.parse(source)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
old=ast.get_source_segment(source,node);new=old;edits=[]
def replace(before,after):
    global new
    assert new.count(before)==1,before
    new=new.replace(before,after);edits.append((before,after))
replace('v, active_v = _cov_compact_diagonal(w32, C, previous, previous_alive)',
        'v, active_v = (_flms_covariance_diagonal(w32, C) if _FULL_LOWER_MOMENTS\n                                else _cov_compact_diagonal(w32, C, previous, previous_alive))')
replace('k3 = k3 + _star_compact_diagonal(w32, star, previous, previous_alive)',
        'k3 = k3 + (_flms_star_diagonal(w32, star) if _FULL_LOWER_MOMENTS\n                           else _star_compact_diagonal(w32, star, previous, previous_alive))')
replace('Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)',
        'if _FULL_LOWER_MOMENTS:\n                    Cpre = _flms_covariance(w32, C)\n                    Bpre = Cpre\n                else:\n                    Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)')
replace('Bpre = w.T @ (C * previous_alive[:, None] * previous_alive[None, :]) @ w',
        'Bpre = Cpre if _FULL_LOWER_MOMENTS else w.T @ (C * previous_alive[:, None] * previous_alive[None, :]) @ w')
replace('star_pre = fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else _star_compact(w32, star, previous, previous_alive, live_indices)',
        'star_pre = (fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else\n                        (_flms_star(w32, star) if _FULL_LOWER_MOMENTS else\n                         _star_compact(w32, star, previous, previous_alive, live_indices)))')
replace('b = (g * bpre + radial * d * vv / 6) * alive',
        'b = ((g * bpre + radial * d * vv / 6) if _FULL_LOWER_MOMENTS\n                 else (g * bpre + radial * d * vv / 6) * alive)')
mask='''            cv = fnp.diag(C).copy()
            C = C * live
            fnp.fill_diagonal(C, cv)
            td = fnp.diag(target).copy()
            target = target * live
            fnp.fill_diagonal(target, td)'''
replace(mask,'            if not _FULL_LOWER_MOMENTS:\n'+'\n'.join('    '+line for line in mask.splitlines()))
replace('fnp.fill_diagonal(Bpost, fnp.diag(C) * alive)',
        'if not _FULL_LOWER_MOMENTS:\n                fnp.fill_diagonal(Bpost, fnp.diag(C) * alive)')
inverse=new
for before,after in reversed(edits):assert inverse.count(after)==1;inverse=inverse.replace(after,before)
assert inverse==old and source.count(old)==1
helpers=(HERE/'helpers.py').read_text();ast.parse(helpers)
records=[]
for name,flag in [('null.py',False),('candidate.py',True)]:
    suffix='\n\n# FLMS fixed coupled lower-moment support switch.\n_FULL_LOWER_MOMENTS = '+str(flag)+'\n\n'+helpers
    candidate=source.replace(old,new)+suffix;ast.parse(candidate)
    assert candidate[:-len(suffix)].replace(new,old)==source
    path=HERE/name
    with path.open('x') as f:f.write(candidate)
    records.append(dict(path=str(path),sha256=sha(path),enabled=flag,whole_source_inverse_exact=True))
assert (HERE/'candidate.py').read_text().replace('_FULL_LOWER_MOMENTS = True','_FULL_LOWER_MOMENTS = False')==(HERE/'null.py').read_text()
write(HERE/'build.json',dict(parent_sha256=PARENT_SHA,records=records,edited_function='_predict_binary',edited_function_start_line=node.lineno,
    replacements=len(edits),same_parameters_and_cp_source=True,helper_sha256=sha(HERE/'helpers.py'),build_source_sha256=sha(__file__)))
print(records)
