"""Append the qualified consumer to the exact parent; stdlib only."""
import ast
from common import HERE,D,P2,PARENT,read,write,pin,sha,verify,weights_path
def main():
    verify();unit=read(HERE/'unit.json');assert unit['passed']
    guard=read(HERE/'unit_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    gl=read(HERE/'gl32.json');assert sha(HERE/'gl32.json')==unit['gl32']['sha256']
    source=PARENT.read_text();tree=ast.parse(source)
    node=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_nonlinear_pair_batch'][-1]
    original=ast.get_source_segment(source,node)
    cut=original.index('    maxorder =');tail=original.index('    post21 =')
    added='''    pairs = _ec_pairs(a, rho, pdf, g, T21, T22, T31, t3, t4)
    cov = pairs[0] * fnp.outer(s,s)
    pk21 = pairs[1] * fnp.outer(v,s)
    pk22 = pairs[2] * fnp.outer(v,v)
'''
    changed=original[:cut]+added+original[tail:]
    changed=changed.replace('def _nonlinear_pair_batch(','def _ec_nonlinear(',1)
    needle='    v = fnp.diag(C).copy()\n'
    check="    if not bool(fnp.all(fnp.isfinite(v))) or not bool(fnp.all(v >= 1e-10)):\n        raise ValueError('correlation consumer requires positive original prevariance')\n"
    assert changed.count(needle)==1;changed=changed.replace(needle,needle+check,1)
    ast.parse(changed)
    prefix=original[:cut].replace('def _nonlinear_pair_batch(','def _ec_nonlinear(',1)
    assert changed.split(added)[0].replace(check,'')==prefix
    assert changed.split(added)[1]==original[tail:]
    suffix='\n\n# Fixed correlation-resummed consumer; parent prefix unchanged.\n'+(HERE/'kernel.py').read_text()+'\n'
    suffix+='_ec_nodes = '+repr(tuple(gl['nodes']))+'\n_ec_weights = '+repr(tuple(gl['weights']))+'\n'
    suffix+='''_ec_cdf = flops.stats.norm.cdf
_ec_original_pair = nonlinear
_ec_original_first = _nonlinear_first_gaussian
'''+changed+'''
def _ec_pair_dispatch(*args, **kwargs):
    if _ec_enabled:
        return _ec_nonlinear(*args, **kwargs)
    return _ec_original_pair(*args, **kwargs)

def _ec_first_dispatch(*args, **kwargs):
    if _ec_enabled:
        return _ec_nonlinear(*args, **kwargs)
    return _ec_original_first(*args, **kwargs)

nonlinear = _ec_pair_dispatch
_nonlinear_first_gaussian = _ec_first_dispatch
'''
    for name,flag in [('candidate.py',True),('null.py',False)]:
        text=source+suffix+'\n_ec_enabled = '+str(flag)+'\n';ast.parse(text)
        assert text[:len(source)]==source
        with (HERE/name).open('x') as f:f.write(text)
    write(HERE/'build.json',dict(parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),null=pin(HERE/'null.py'),
        parent_byte_prefix_exact=True,marginal_prefix_exact_except_domain_checks=True,post_centering_tail_exact=True,
        original_pair_helper_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),gl32=pin(HERE/'gl32.json')))
    old=read(D/'full_lower_moment_support_v1/inputs.json')
    inputs={k:old[k] for k in ['parent_dev','parent_cold100','truths','expected_parent']}
    write(HERE/'inputs.json',inputs)
    external=[PARENT,D/'run_guarded.py',D/'weight5_reuse_v1/native.py',D/'full_lower_moment_support_v1/inputs.json']
    for key in ['parent_dev','parent_cold100']:
        m=inputs[key];assert sha(m['path'])==m['sha256'];external.append(__import__('pathlib').Path(m['path']))
        for r in read(m['path'])['records']:external.append(__import__('pathlib').Path(r['output']))
    external +=[weights_path(rank) for rank in [35,100]]
    external +=[__import__('pathlib').Path(x['path']) for x in inputs['truths'].values()]
    own=list(HERE.glob('*.py'))+[HERE/x for x in ['PROTOCOL.md','SOURCE_REVIEW.md','sources_sealed.json','unit.json','unit_watch.json','gl32.json','build.json','inputs.json']]
    paths=sorted(set(own+external))
    write(HERE/'native_sources_sealed.json',dict(pins=[pin(p) for p in paths],ordinary_mean_arrays_decoded=False))
    print(__import__('json').dumps(read(HERE/'build.json')),flush=True)
if __name__=='__main__':main()
