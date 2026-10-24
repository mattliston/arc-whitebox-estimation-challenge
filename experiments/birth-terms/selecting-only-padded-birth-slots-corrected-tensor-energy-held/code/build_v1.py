"""Freeze one same-count, above-floor-preserving BD birth-padding candidate."""
import ast, hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parents[1] / 'astra_oct20_vm_frontier_v1'
PINS = {'science': '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510',
        'release': 'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def definitions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}

def transform(text):
    defs = definitions(text)
    bd = ast.get_source_segment(text, defs['_bd_update'])
    core = ast.get_source_segment(text, defs['_astra_fhm_core'])
    bd_before = "    weight = (_e2_append_inline"
    assert bd.count(bd_before) == 1
    bd = bd.replace(bd_before, "    births = _o21_birth_refine(new32, a64, t64, s64, births)\n" + bd_before)
    select = 'selected, births = select(_e2_gate_min, gate_floor)'
    assert core.count(select) == 1
    core = core.replace(select, 'selected, births = _o21_birth_prepare(_e2_gate_min, gate_floor)')
    lines = core.splitlines()
    prefixes = ('            _e2_bb = birth * births', '            born = fnp.einsum(', '            _ps_born_diag = (_e2_append_inline')
    moved = []
    for prefix in prefixes:
        found = [line for line in lines if line.startswith(prefix)]
        assert len(found) == 1, prefix
        moved += found
        lines.remove(found[0])
    anchor = '            del _bd_current_c'
    at = lines.index(anchor) + 1
    lines[at:at] = ['            selected, births = _o21_birth_finish(selected, births)'] + moved
    core = '\n'.join(lines)
    return bd, core, moved

def main():
    assert not (HERE / 'CANDIDATES_v1.json').exists()
    parents = {r: V / 'msg8_loaderfix_v1' / ('candidate_msg8_teacher_loaderfix_' + ('sci' if r == 'science' else 'rel') + '_v1.py') for r in PINS}
    texts = {r: p.read_text() for r, p in parents.items()}
    assert all(sha(parents[r]) == PINS[r] for r in PINS)
    bd, core, moved = transform(texts['science'])
    assert (bd, core, moved) == transform(texts['release'])
    tail = '\n\n# Same-count corrected-base birth-padding selection; ' + utc() + '.\n'
    tail += 'assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14\n'
    tail += 'assert _GNN_PARAMS is None and _BD_J1_PHYS is not None\n'
    tail += 'assert _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}\n'
    tail += 'assert _ASTRA_FHM_ACTIVE_CORE.__wrapped__ is _astra_fhm_core\n'
    tail += 'assert dict(_ASTRA_DENSE_CONTEXT)["_bd_update"] is _bd_update\n'
    tail += (HERE / 'selector_v1.py').read_text() + '\n\n' + bd + '\n\n' + core + '\n\n'
    tail += "_ASTRA_DENSE_CONTEXT = tuple((name, _bd_update if name == '_bd_update' else value) for name, value in _ASTRA_DENSE_CONTEXT)\n"
    tail += '_ASTRA_FHM_ACTIVE_CORE = _tr_clearing(_astra_fhm_core)\n_FAST_RESCUE_FRAC = {}\n'
    tail_path = HERE / 'tail_msg8_bd_padding_energy_v1.py'
    tail_path.open('x').write(tail)
    rec = dict(name='msg8_bd_padding_energy', append_tail=str(tail_path), append_tail_sha256=sha(tail_path), fit=False, truth_used=False)
    for role in PINS:
        out = HERE / ('candidate_msg8_bd_padding_energy_' + ('sci' if role == 'science' else 'rel') + '_v1.py')
        out.open('x').write(texts[role] + tail)
        compile(out.read_text(), str(out), 'exec')
        rec[role] = dict(path=str(out), sha256=sha(out), bytes=out.stat().st_size, parent_path=str(parents[role]), parent_sha256=PINS[role])
    receipt = dict(utc=utc(), status='BD_PADDING_SOURCE_FROZEN_NO_NATIVE', reference_submission=337422,
        candidates=[rec], parent_source_sha256=PINS['science'], builder_sha256=sha(__file__),
        selector_sha256=sha(HERE/'selector_v1.py'), moved_born_statements=moved,
        changed_functions=['_bd_update', '_astra_fhm_core'], no_governor=True,
        numerical_change='Only below-floor padding birth choices change. Every above-floor unit, original32-rounded count and finalgate order are preserved. Corrected BDbase uses the exact unchanged learned coefficients. Later propagated rosters may change.',
        old_closure_distinction='Old all-distinct birthenergy selected the entire roster from the uncorrected alive*g*Cpre proxy. This candidate uses the actual BD-corrected base and changes only0–31 padding units.',
        accuracy_gain_unknown=True, tariff_unmeasured=True)
    (HERE/'CANDIDATES_v1.json').open('x').write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'], candidate=rec), indent=2))

if __name__ == '__main__':
    main()
