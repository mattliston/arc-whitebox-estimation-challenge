#!/usr/bin/env python3
"""TAG fable-analyst (AN-50): no-refit VALID16 test of the engineer's E-48 row-drop physics variants on the torch S6b card (s6bcard.py:
HookCard + v3 knobs + L15 row cut) with the B2 epoch-1 params (fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt).
In the card, the pushed columns' rows of a unit feed only its own CP k3 diagonal (sum lam pp^3) and the omitted-radial source; the plane
read uses response rows and the frame U, and the carried atoms are masked by alive. So each variant is a row mask on p64 (the float64
copy of the pushed columns) at L1-14, inserted by a count-asserted edit of covstep.HookCard.step right after `p64 = pp.to(t.float64)`:
  a  dead-side row cut (E-48 2b): zero every column on rows with a = mu/sigma < -3.5 (kept rows = select(a + 3.5, 0), rounded up to 32);
     aT (e.g. a3.0, a2.75) = the same cut at a < -T. Records rows cut per layer, rounded (as the card's select keeps them) and exact.
  b  old-tier rebuild on response rows only (E-48 1p): zero the old-tier columns on non-response rows (dead + saturated)
  c  all column work on alive rows only (E-48 2): zero every column on non-alive rows
Mode None is the unmodified card (parity checked against s6bcard.card). usage: rowdrop_s6b.py <modes e.g. base,a,b,c> <threads> <out.jsonl>"""
import sys, ast, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard
cs, gn3 = s6bcard.cs, s6bcard.gn3
OLD = "            p64 = pp.to(t.float64)\n            cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)\n"   # pair-layer block only
NEW = ("            p64 = pp.to(t.float64)\n            p64 = self.rowdrop(layer, p64, mu, vv, alive, response_mask, old_columns)\n"
       "            cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)\n")
_text = (P2 / 'fable_covhunter_v1/covstep.py').read_text(); _tree = ast.parse(_text)
_cls = [n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == 'HookCard'][0]
_step = [n for n in _cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step'][0]
_src = '\n'.join(l[4:] if l.startswith('    ') else l for l in ('    ' + ast.get_source_segment(_text, _step)).split('\n'))
o, n_ = '\n'.join(l[4:] for l in OLD.split('\n')), '\n'.join(l[4:] for l in NEW.split('\n'))
assert _src.count(o) == 1; _src = _src.replace(o, n_)
_ns = dict(cs.__dict__); exec(compile(_src, 'rowdrop_step', 'exec'), _ns)
class RowDropCard(cs.HookCard):
    step = _ns['step']
    mode = None
    def rowdrop(self, layer, p64, mu, vv, alive, response_mask, old_columns):
        if self.mode is None or not 1 <= layer <= 14: return p64
        t = self.t
        if self.mode.startswith('a'):
            score = mu / vv.clamp_min(1e-10).sqrt() + (3.5 if self.mode == 'a' else float(self.mode[1:]))
            self.dropped_exact.setdefault(layer, int((score < 0).sum()))
            count = min(1024, ((int((score >= 0).sum()) + 31) // 32) * 32)
            keep = t.zeros_like(score); keep[t.argsort(-score, stable=True)[:count]] = 1.0
        elif self.mode == 'b':
            keep = response_mask
        else:
            keep = alive
        self.dropped.setdefault(layer, int((keep == 0).sum()))
        if self.mode == 'b':
            q = p64.clone(); q[:, :old_columns] = q[:, :old_columns] * keep[:, None]; return q
        return p64 * keep[:, None]
def card(torch_, w, pl, mode):
    cd = RowDropCard(torch_, w, *pl); gn3.configure(cd, s6bcard.V3); cd.hooks['k3_15'] = s6bcard.l15_rowcut(torch_, s6bcard.ZC)
    cd.mode = None if mode == 'base' else mode; cd.dropped = {}; cd.dropped_exact = {}
    return cd
if __name__ == '__main__':
    modes = sys.argv[1].split(','); torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
    B2 = P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt'
    pl = s6bcard.payload(torch, params=B2)
    VALID16 = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:16]
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        r0 = VALID16[0]; w = s6bcard.weights(r0)                      # parity: mode None == s6bcard.card, all 16 layers
        with torch.no_grad():
            d = float((card(torch, w, pl, 'base').run(False) - s6bcard.card(torch, w, *pl).run(False)).abs().max())
        fo.write(json.dumps(dict(check='parity base vs s6bcard', rank=r0, max_abs_diff=d)) + '\n'); print('parity max abs diff', d, flush=True)
        for r in VALID16:
            w, tp = s6bcard.weights(r), truth(r)
            for mode in modes:
                t0 = time.time(); cd = card(torch, w, pl, mode)
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='AN-50 rowdrop (B2 ep1, s6bcard, no refit)', rank=r, mode=mode, mse=float(((pred - tp) ** 2).mean()),
                           dropped_rows={str(k): v for k, v in sorted(cd.dropped.items())},
                           dropped_rows_exact={str(k): v for k, v in sorted(cd.dropped_exact.items())}, wall=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'mode', 'mse', 'wall')}), flush=True)
