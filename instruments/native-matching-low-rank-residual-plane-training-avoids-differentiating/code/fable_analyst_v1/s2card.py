"""TAG fable-analyst (AN-34): the S2 physics as a torch research card (fired as #331947: e2 v7 + K31 row (.85, .5) at pair layers 1-14 +
gain b-fix gamma 0.1 (partition, mean m) + join schedule every3 {7: 0-2, 10: 3-5, 13: 6-8}). Built from fable_refit_v1/model_bank_bfix.py
(BFIX_GAMMA = 0.1), the K31 row source patch of fable_refit_v1/physics_stack_test.py and the make_push of fable_frame_v1/join_schedule_test.py
(copied here: that script parses argv at import). card(torch, weights, p, a, c, e) -> the patched Card."""
import sys, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent; REFIT = HERE.parent / 'fable_refit_v1'
if str(REFIT) not in sys.path: sys.path.insert(0, str(REFIT))
import model_bank_bfix as model
model.BFIX_GAMMA = 0.1; model.BFIX_MODE = 'partition'; model.BFIX_MEAN = 'm'
EVERY3 = {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]}
_text = model.SOURCE.read_text(); _tree = ast.parse(_text)
_node = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
_src = ast.get_source_segment(_text, _node)
_old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert _src.count(_old) == 1
_src = _src.replace(_old, "    T31 = (t4 * (_ROWF[0](mu, k3, v) if _ROWF[0] is not None else 1.0))[:, None] * rho if k31 else None\n")
def make_push(qr, joins):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :rank]
            Y = J @ (weight[:, None] * (J.T @ Omega))
            Qp = Q[previous_indices, :] if old_columns else None
            if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            Qn = qr(torch, Y)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = Qn.T @ Qp
                F = torch.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return torch.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push
def _ckpt(torch, fn):
    from torch.utils.checkpoint import checkpoint
    def wrapped(*args, **kw):
        if torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, **kw)
        return fn(*args, **kw)
    return wrapped
def card(torch, w, p, a, c, e, k31=True, every3=True, fine_ckpt=False):
    """fine_ckpt: nested non-reentrant checkpoints around the pair map, _bd_update, the star push, the tier push and the plane read
    (memory only: identical values and gradients)."""
    cd = model.Card(torch, w, p, a, c, e)
    cd.env['_ROWF'] = [(lambda mu, k3, v: 0.85 + 0.5 * mu * k3 / (v * v)) if k31 else None]
    t_ = model.Translate().visit(ast.parse(_src)); ast.fix_missing_locations(t_); exec(compile(t_, 's2pair', 'exec'), cd.env)
    if every3: cd.shared['push'] = make_push(cd.shared['_qr'], EVERY3)
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = _ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = _ckpt(torch, cd.shared[k])
    return cd
