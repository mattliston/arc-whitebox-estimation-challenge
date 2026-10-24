"""Build two fixed late-only local common-factor screens; no numerical work."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'candidate_thin_star_route_v1.py'
EXPECTED = '25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
source = BASE.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
tree = ast.parse(source)
lines = source.splitlines()
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
node = functions['_slr_reconstruct']
reconstruction = '\n'.join(lines[node.lineno - 1:node.end_lineno])
assert reconstruction.count("    ctx = _OTT_CONTEXT[0]") == 1
reconstruction = reconstruction.replace(
    "    ctx = _OTT_CONTEXT[0]",
    "    if _UVP_CUR[0] not in _LCR_LAYERS:\n"
    "        return _LCR_PARENT_RECON(U, state)\n"
    "    ctx = _OTT_CONTEXT[0]", 1)
old = """        prefix = _shared_old384_mm(mapped[rows], group['base'])
        recent = _shared_old384_mm(U[rows], group['join'])"""
new = """        uv = _uvp_state(state)
        _slr_require(uv[0] == 'l', 'local common lazy split')
        prefix = _uvp_recon(mapped[rows], uv[1])
        recent = _uvp_recon(U[rows], uv[2])"""
assert reconstruction.count(old) == 1
reconstruction = reconstruction.replace(old, new, 1)

prefix = '''
# Fixed late-only local common-factor approximation. Carried F/core/Q are
# untouched; every difference direction and all signed coefficients remain.
# Existing both-tail trace, diagonal and omitted-radial repair use full F.
_LCR_LAYERS = (10, 11, 12, 13, 14)
_UVP_ONLY_LAYERS = _LCR_LAYERS + (15,)
_LCR_PARENT_RECON = _slr_reconstruct
_LCR_PARENT_UV_STATE = _uvp_state

def _uvp_state(state):
    # A late common192 block must not leak into the incumbent L15 common256.
    global _UVP_RU
    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256
    if state.get('lcr_ru') != ru:
        state['uvp'] = None
    saved = _UVP_RU
    _UVP_RU = ru
    try:
        uv = _LCR_PARENT_UV_STATE(state)
    finally:
        _UVP_RU = saved
    state['lcr_ru'] = ru
    return uv

'''

records = []
for rank in (256, 192):
    name = f'late_common{rank}'
    footer = f'\n_LCR_RU = {rank}\n' + prefix + reconstruction + '\n'
    candidate = source + footer
    compile(candidate, name, 'exec')
    path = HERE / (name + '.py')
    with path.open('x') as f:
        f.write(candidate)
    records.append(dict(name=name, path=str(path), sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                        common_rank=rank, layers=[10,11,12,13,14], terminal_common_rank=256,
                        parent=str(BASE), parent_sha256=EXPECTED,
                        changed_source_prefix=False, numerical_execution=False))
with (HERE / 'candidates.json').open('x') as f:
    json.dump(records, f, indent=2)
print(json.dumps(records, indent=2))
