"""Freeze a current777a direct-vector64/shared-cubic32 candidate; no native run."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1];A=R/'fable_oct19_kc_v1/agent_astra_joint_v1';V=R/'astra_oct20_vm_frontier_v1'
PINS={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510','release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def main():
    assert not (HERE/'CANDIDATES_v1.json').exists()
    parents={r:V/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+('sci'if r=='science'else'rel')+'_v1.py')for r in PINS}
    texts={r:p.read_text()for r,p in parents.items()};assert all(sha(parents[r])==PINS[r]for r in PINS)
    s=texts['science'];nodes={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    old=ast.get_source_segment(s,nodes['_o20_npx_copy_exact_left_v1'])
    begin=old.index('        W, rows_kept = ');end=old.index('        local = (y0, y1, value)',begin)
    replacement='''        assert _NP_K3L_ALL
        own = hold(rows[:, y0:y1])
        cuts, pair_cuts, at, pair_at = [], [], 0, 0
        for born, width, strength in young:
            cuts += [at + width, at + 2 * width]
            at += 2 * width
            pair_at += width
            pair_cuts.append(pair_at)
        full = hold(pp[:, y0:y1])
        pieces = hold(fnp.split(full, cuts[:-1], axis=1))
        squares = hold(fnp.sum(hold(pp2[:, y0:y1]), axis=0))
        halves = hold(fnp.split(squares, cuts[:-1]))
        pair = [hold(fnp.einsum('ik,ik->k', pieces[i], pieces[i + 1], optimize=False)) for i in range(0, len(pieces), 2)]
        lp = hold(fnp.split(hold(lam[y0:y1]), cuts[:-1]))
        quartic = [(hold(S['k4e'][born][:, :width]) if strength == 1.0 else hold(hold(S['k4e'][born][:, :width]) * strength)) for born, width, strength in young]
        def joined(parts, axis=0):
            return parts[0] if len(parts) == 1 else hold(fnp.concatenate(parts, axis=axis))
        common = hold(_o21_k31_common_matrix64(joined(lp[::2]), joined(quartic, 1), joined(halves[::2]), joined(halves[1::2]), joined(pair), n, hold))
        value = hold(_o21_k31_common_polynomial32(own, common, young, hold))
        _O21_K31C_STATS['calls'] += 1
        _O21_K31C_STATS['row_pairs'] += int(rows.shape[0]) * pair_at
        _O21_K31C_STATS['pairs'] += pair_at
'''
    fn=old[:begin]+replacement+old[end:]
    fn=fn.replace('def _o20_npx_copy_exact_left_v1(', 'def _o21_k31c_left_v1(',1)
    coeff=(HERE/'common_coefficients_batched64_v1.py').read_text().replace('_o21_k31_common64','_o21_k31_common_matrix64')
    final='    return hold(fnp.unstack(hold(hold(fnp.concatenate((coeff,lower),axis=0)).astype(fnp.float32)),axis=0))'
    assert final in coeff;coeff=coeff.replace(final,'    return hold(hold(fnp.concatenate((coeff,lower),axis=0)).astype(fnp.float32))')
    frag=(A/'fragment_k31_pair_shared_complete_v2.py').read_text();node=next(n for n in ast.parse(frag).body if isinstance(n,ast.FunctionDef)and n.name=='_astra_k31pair_polynomial');poly=ast.get_source_segment(frag,node)
    poly=poly.replace('_astra_k31pair_polynomial','_o21_k31_common_polynomial32',1)
    poly=poly.replace('coeff_parts = hold(fnp.split(coefficients, cuts[:-1], axis=1))','pair_cuts = []\n    pair_at = 0\n    for _born, width, _strength in young:\n        pair_at += width\n        pair_cuts.append(pair_at)\n    coeff_parts = hold(fnp.split(coefficients, pair_cuts[:-1], axis=1))')
    b=poly.index('        aa, c2, c1, c0, co, cd = ');e=poly.index('        delta = ',b)
    poly=poly[:b]+'        aa, bb, cc, ddcoef, ee, ff, gg = hold(fnp.unstack(coeff_parts[j], axis=0))\n\n'+poly[e:]
    b=poly.index('        row_pairs = ');poly=poly[:b]+'    return value\n'
    tail='\n\n# Direct K31 coefficient64 vectors, shared cubic32; '+utc()+'.\n'
    tail+='assert _NP_K3L_ALL and _O20_NPX_COPY_EXACT_ON\nassert _K3LB_PARENT_LEFT is _o20_npx_copy_exact_left_v1\nassert _GNN_PARAMS is None and _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}\n_O21_K31C_STATS = dict(calls=0,row_pairs=0,pairs=0)\n\n'
    tail+=coeff+'\n\n'+poly+'\n\n'+fn+'\n\n_K3LB_PARENT_LEFT = _o21_k31c_left_v1\n_FAST_RESCUE_FRAC = {}\n'
    tp=HERE/'tail_msg8_k31coeff64_v1.py';tp.open('x').write(tail)
    rec=dict(name='msg8_k31coeff64',append_tail=str(tp),append_tail_sha256=sha(tp),component='_K3LB_PARENT_LEFT',fit=False,truth_used=False)
    for role in PINS:
        out=HERE/('candidate_msg8_k31coeff64_'+('sci'if role=='science'else'rel')+'_v1.py');out.open('x').write(texts[role]+tail);compile(out.read_text(),str(out),'exec')
        rec[role]=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,parent_path=str(parents[role]),parent_sha256=PINS[role])
    receipt=dict(utc=utc(),status='K31_DIRECT_COEFFICIENT64_SOURCE_FROZEN_NO_NATIVE',reference_submission=337422,parent_source_sha256=PINS['science'],candidates=[rec],builder_sha256=sha(__file__),large_polynomial_dtype='float32',coefficient_dtype='float64',original_age_strengths=True,age4_not_added=True,all_nonlocal_NP_and_auxiliary_arithmetic_unchanged=True,prior_shared32_rejected=True,expected_accuracy_gain=None,whole_network_bill_and_runtime_unmeasured=True)
    (HERE/'CANDIDATES_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
