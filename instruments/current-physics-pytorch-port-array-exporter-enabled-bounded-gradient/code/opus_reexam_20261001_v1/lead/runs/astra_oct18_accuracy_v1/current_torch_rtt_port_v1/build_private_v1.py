from pathlib import Path
import ast, hashlib, json
D = Path(__file__).resolve().parent
R = D.parent.parent
O = R.parent.parent
E = O / 'engineer/a22_cutsim'
push = R / 'astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_push_current95c_jsk_v1.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(push) == 'ff36fb11a5c69e8aaeb026fc1c979ee6429de64ab27e046240ae70d1cf612785'
def replace(s, a, b):
    assert s.count(a) == 1, (a, s.count(a))
    return s.replace(a, b)
def save(name, s):
    ast.parse(s)
    (D/name).open('x').write(s)

s = push.read_text()
s = replace(s, "        if not old_columns or not context.get('truncated'):\n            return read0(tt, pp, lam, response, live, U, F, old_columns)\n", "        if not old_columns:\n            cd._o18_rtt_base_old = None\n            return read0(tt, pp, lam, response, live, U, F, old_columns)\n")
s = replace(s, "        inners = [uv_inner(t, left[:, b['lo']:b['hi']] ** 2,\n                           lam[b['lo']:b['hi']], b['F'], b['pairs'], ru)\n                  for b in context['blocks']]\n", "        inners = [(uv_inner(t, left[:, b['lo']:b['hi']] ** 2,\n                            lam[b['lo']:b['hi']], b['F'], b['pairs'], ru)\n                   if context['truncated'] else\n                   ((left[:, b['lo']:b['hi']] ** 2) * lam[b['lo']:b['hi']]) @ b['F'].T)\n                  for b in context['blocks']]\n")
s = replace(s, '        old_plane = inner @ U[live].T\n', '        old_plane = inner @ U[live].T\n        cd._o18_rtt_base_old = (int(cd.call_layer), old_plane, lam)\n')
save('torch_push_current95c_rtt_v1.py', s)

s = (E/'n6y_cutsim_v2.py').read_text()
a = '            B = (pp * w[None, :] + pp[:, partner] * x[None, :]) @ right.T\n'
b = '''            left_B = pp * w[None, :] + pp[:, partner] * x[None, :]
            old_B = left_B[:, :old_columns] @ right[:, :old_columns].T
            B = old_B + left_B[:, old_columns:] @ right[:, old_columns:].T
            S['_o18_old_B'] = old_B
'''
s = replace(s, a, b)
a = '                    plane = plane + k31 * Fd[response_indices][:, indices].to(plane.dtype)\n'
b = '''                    plane = plane + k31 * Fd[response_indices][:, indices].to(plane.dtype)
                    factor = -k31 * (S['mu'][response_indices] /
                        (n * t.diagonal(S['C'])[response_indices].clamp_min(1e-12)))
                    cd._o18_rtt_law = dict(layer=int(S['l']),
                        forms=factor[:, None] * (R-B)[response_indices][:, indices],
                        old_fold=factor[:, None] * S['_o18_old_B'][response_indices][:, indices],
                        factor=factor, old_w=S['w'][:old_columns],
                        old_x=S['x'][:old_columns], old_partner=S['partner'][:old_columns])
'''
s = replace(s, a, b)
save('n6y_rtt_private_v1.py', s)

s = (E/'torch_rowcuts_v15.py').read_text()
s = replace(s, "    path = Path(__file__).resolve().with_name('pbase_cutsim_v1.py')\n", '    path = Path('+repr(str(E/'pbase_cutsim_v1.py'))+')\n')
s = replace(s, "    lawpath = Path(__file__).resolve().with_name('n6y_cutsim_v2.py')\n", '    lawpath = Path('+repr(str(D/'n6y_rtt_private_v1.py'))+')\n')
s += '''
_O18_PARENT_INSTALL_M = install_m
_O18_RTT_MODULE = None
_O18_PUSH_MODULE = None
def install_m(cd, original_m_install):
    answer = _O18_PARENT_INSTALL_M(cd, original_m_install)
    assert _O18_RTT_MODULE is not None and _O18_PUSH_MODULE is not None
    return _O18_RTT_MODULE.install_after_m(answer, _O18_PUSH_MODULE)
'''
save('torch_rowcuts_rtt_private_v1.py', s)

s = '''"""Configure private children after original configure_cutsim(spec)."""
from pathlib import Path
import importlib.util
D = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location('_o18_current_'+name, D/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
def configure(runtime_module):
    push=load('torch_push_current95c_rtt_v1')
    row=load('torch_rowcuts_rtt_private_v1')
    port=load('torch_rtt_current95c_v1')
    row._O18_RTT_MODULE=port;row._O18_PUSH_MODULE=push
    runtime_module._CHILDREN['torch_push_native_v2.py']=push
    runtime_module._CHILDREN['torch_rowcuts_v15.py']=row
    return push,row,port
'''
save('configure_private_v1.py', s)
receipt=dict(status='PRIVATE_SOURCE_READY_NO_FULL_FORWARD',
    files={p.name:sha(p) for p in D.glob('*.py')},
    parent_jsk_sha=sha(push), parent_n6y_sha=sha(E/'n6y_cutsim_v2.py'),
    numerical_scope='RTT rank2 signed-prefix power, exact old tail, NP forms after RTT, original RE22; same full K31L young folds.',
    differences='Torch GEMM/QR reductions remain; private n6y B evaluated as old+young products; L7 base old read explicitly factorized; no native byte parity claim.',
    gradient_scope='Existing frozen carrier replay retained; same-step base tensors differentiable, frozen form/old-law records detached; refined Q remains differentiable.',
    no_live_mutation=True, no_training=True, models=0)
(D/'BUILD_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
