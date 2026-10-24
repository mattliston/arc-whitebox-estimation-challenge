# [opus-jrr] Rayleigh-Ritz join core on S2tr 46e9315d (lead 14:5xZ item 3: near-eigen join refinement with no G multiply).
# The live join keeps a NYSTROM core R K^-1 R' (= Qn' G_nys Qn, which under-states the Gram) for the next join's Y and row energies,
# while the bank it actually carries is the PROJECTION P [old, J] (factors = Qn' columns). The consistent core is the represented
# bank's frame Gram F W F' = Fj W Fj' + overlap core_old overlap', built from Fj and overlap that the join forms anyway (exact at
# the first join, then exact by recursion). Text edits of the LIVE (last) _astra_bcc_push_impl only; joins 7 and 10 store cores.
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
def edit(s, old, new):
    j = s.find(old, i0)
    assert j > i0 and s.find(old, j + 1) == -1, old[:60]
    return s[:j] + new + s[j + len(old):]
SOLVE = """        if _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)
            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.float32), R64.T).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
"""
FJ = """        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        joined_core = None
        if old_columns:
            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)
            factors = _slr_join_factors(layer, overlap, state, Fj)
            del overlap
        else:
            factors = Fj
"""
FJ_RR = """        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        if _need_core:   # [opus-jrr] Rayleigh-Ritz core: F W F' of the represented bank in the new frame (no G multiply)
            core = _shared_old384_mm(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T)))
        joined_core = None
        if old_columns:
            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)
            if _need_core:   # [opus-jrr] + overlap core_old overlap'
                core = core + _shared_old384_mm(overlap, _e2_H(_shared_old384_mm(state['core'], _e2_H(overlap.T))))
            factors = _slr_join_factors(layer, overlap, state, Fj)
            del overlap
        else:
            factors = Fj
        if _need_core:
            core = (core + core.T) * 0.5
"""
out = edit(edit(src, SOLVE, ""), FJ, FJ_RR) + '\n\n# [opus-jrr] ARM jrr: Rayleigh-Ritz join core (text edits of the live join in _astra_bcc_push_impl, S2tr 46e9315d)\n'
p = H + '/cand_s2tr_l2h_jrr_v1.py'
open(p, 'w').write(out)
print('jrr', hashlib.sha256(out.encode()).hexdigest()[:8], p)
