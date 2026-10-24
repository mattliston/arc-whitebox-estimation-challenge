"""Autonomous current all-P capped Torch shadow; no fitting or data I/O at import.

Dense physical Torch factors represent the native shared-newborn tuple. Only
ranking is detached. Surviving factors AND discarded-plane star feedback remain
differentiable. This is not a claim of identical native floating GEMM circuits.
"""
import sys
from common import BD, HERE, NEW, PINS, module, sha

for _path in (BD/'model.py', NEW/'centered_bank_autodiff_v1.py',
              NEW/'centered_bank_autodiff_v3.py', NEW/'full_carrier_autodiff_v1.py'):
    if sha(_path) != PINS[_path]:
        raise RuntimeError('Changed frozen Torch source: '+str(_path))
sys.path.insert(0, str(NEW))
import numpy as np
import torch
from joint_prefix_final_data_v1 import PKEYS

base = module(BD/'model.py', 'cap4096_qualified_all_p_bridge')
diagonal = base.diagonal
PAIR_CAP = 4096


def dropped_plane(physical, lam, columns, live_indices, dtype):
    """Same physical owner as native: live matrix product plus ALL exact cubes.

    In production physical/lam are FP32. The CPU64 diagonal transfer is
    explicitly differentiable, including from MPS; it is not a fallback.
    A CPU64 tiny algebra fixture may pass FP64 physical factors deliberately.
    """
    dropped, weights = physical[:, columns], lam[columns]
    sub = dropped[live_indices]
    values = ((sub*sub) @ (sub*weights).T).to(dtype)
    n = int(physical.shape[0])
    out = torch.zeros((n, n), device=physical.device, dtype=dtype)
    out = out.index_put((live_indices[:, None], live_indices[None, :]), values)
    exact = dropped.to(device='cpu', dtype=torch.float64)
    exact_weights = weights.to(device='cpu', dtype=torch.float64)
    cubes = ((exact*exact)*exact*exact_weights).sum(1)
    return diagonal(out, cubes.to(device=physical.device, dtype=dtype))


def retire(physical, lam, positive, negative, new_count, cp_post,
           live_indices, cap=PAIR_CAP, audit=None):
    """Physical-column equivalent of the pinned native retirement constructor."""
    old_columns = physical.shape[1]-2*int(new_count)
    assert isinstance(cap, int) and cap > 0
    assert old_columns == 2*positive.numel() == 2*negative.numel()
    assert physical.shape[1] == lam.numel()
    offsets = torch.arange(new_count, device=physical.device, dtype=torch.long)
    pos = torch.cat((positive, old_columns+offsets))
    neg = torch.cat((negative, old_columns+new_count+offsets))
    pairs = int(pos.numel())
    if pairs <= cap:
        record = dict(pairs_before=pairs, pairs_after=pairs,
            retired=False, chosen=list(range(pairs)), minimum_relative_margin=None)
        if audit is not None:
            audit(record)
        return physical, lam, pos, neg, cp_post, record
    pc, nc = pos.detach().cpu().numpy(), neg.detach().cpu().numpy()
    # The sole detach of numerical factor values: a locally constant selector.
    first = physical.detach()[:, pos].to('cpu', torch.float64).numpy()
    second = physical.detach()[:, neg].to('cpu', torch.float64).numpy()
    weights = lam.detach().to('cpu', torch.float64).numpy()
    assert np.array_equal(np.sort(np.concatenate((pc, nc))), np.arange(len(weights)))
    assert np.array_equal(weights[pc], -weights[nc])
    aa, bb, ab = (first*first).sum(0), (second*second).sum(0), (first*second).sum(0)
    energy = weights[pc]**2*np.maximum(aa**3+bb**3-2*ab**3, 0)
    if not np.isfinite(energy).all():
        raise FloatingPointError('Nonfinite retirement energy')
    order = np.argsort(-energy, kind='stable')
    chosen = np.sort(order[:cap])
    dropped = order[cap:]
    keep_columns = np.sort(np.concatenate((pc[chosen], nc[chosen])))
    drop_columns = np.sort(np.concatenate((pc[dropped], nc[dropped])))
    inv = np.zeros(len(weights), dtype=np.int64)
    inv[keep_columns] = np.arange(len(keep_columns))
    tensor = lambda x: torch.as_tensor(x, device=physical.device, dtype=torch.long)
    boundary = float(energy[order[cap-1]])
    margin = (boundary-float(energy[order[cap]]))/max(abs(boundary), 1e-300)
    record = dict(pairs_before=pairs, pairs_after=cap, retired=True,
        chosen=chosen.tolist(), minimum_relative_margin=margin)
    # Audit before differentiable reads: non-reentrant checkpoint may stop
    # replay immediately after its last saved tensor, before a return hook.
    if audit is not None:
        audit(record)
    removed = dropped_plane(physical, lam, tensor(drop_columns), live_indices, cp_post.dtype)
    return (physical[:, tensor(keep_columns)], lam[tensor(keep_columns)],
            tensor(inv[pc[chosen]]), tensor(inv[nc[chosen]]), cp_post-removed, record)


source = base.source
source = base.replace_once(source,
    '    m, C, p, lam, star, b, previous_alive, radial, hstate = state',
    '    m, C, p, lam, star, b, previous_alive, radial, hstate, positive, negative = state')
source = base.replace_once(source, '    vv = torch.diagonal(Cpre)',
    '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)')
source = base.replace_once(source,
    '    stored_base, plane_increment = birth_action(bp,Cpre,k21,aux,self.layer_coefficients(l),t,s,a,alive,births)',
    '    stored_base = (bp*alive[:,None]).float()')
source = base.replace_once(source, '    born = born+plane_increment\n', '')
source = base.replace_once(source,
    '    rest = target-cp_post-trace_post',
    '    p, lam, positive, negative, cp_post, retirement = retire(\n'
    '        p,lam,positive,negative,len(selected),cp_post,indices,PAIR_CAP,\n'
    '        audit=lambda item: self._record_retirement(l,item))\n'
    '    rest = target-cp_post-trace_post')
old_return = '(m, C, p, lam, star, b, alive, new_radial, hs)'
assert source.count(old_return) == 2
source = source.replace(old_return, '(m, C, p, lam, star, b, alive, new_radial, hs, positive, negative)')
code, count = base.rewrite_clamps(source, str(__file__)+':step')
assert count == 2
env = dict(base.env, retire=retire, PAIR_CAP=PAIR_CAP)
exec(code, env)


class CappedCard(base.BirthDirectionCard):
    def __init__(self, rank, device='mps', weights=None, P=None, audit_selections=True):
        # No zero theta is evaluated and no birth-direction basis is computed.
        super().__init__(rank, coefficients=(), device=device, weights=weights,
                         P=P, audit_selections=audit_selections)
        self.retirement = {}
        self.replay_counts = {}
        self._layer = None
        self._selection_slot = 0
        self.selection_cells = {}
        self.variance_domain = {}

    def initial(self):
        state = super().initial()
        empty = torch.zeros(0, device=self.device, dtype=torch.long)
        return (*state, empty, empty)

    def _record_retirement(self, layer, item):
        if layer in self.retirement and self.retirement[layer]['chosen'] != item['chosen']:
            raise RuntimeError('Checkpoint recomputation changed retirement IDs')
        self.retirement[layer] = item
        self.replay_counts[layer] = self.replay_counts.get(layer, 0)+1

    def _record_variance(self, layer, variance):
        if self.audit_selections:
            value = variance.detach().cpu()
            self.variance_domain[layer] = dict(minimum=float(value.min()),
                below_floor_count=int((value < 1e-10).sum()), finite=bool(torch.isfinite(value).all()))

    def _select(self, score):
        ix, mask = base.centered_source.select(score)
        key = (self._layer, self._selection_slot)
        self._selection_slot += 1
        if self.audit_selections:
            ids = ix.detach().cpu().tolist()
            if key in self.selection_cells and self.selection_cells[key] != ids:
                raise RuntimeError('Checkpoint recomputation changed alive/birth IDs')
            self.selection_cells[key] = ids
        return ix, mask

    def step(self, layer, state, control, P):
        self._layer, self._selection_slot = layer, 0
        return env['step'](self, layer, state, control, P)

    def run(self, *args, **kwargs):
        self.retirement.clear()
        self.replay_counts.clear()
        self.selection_cells.clear()
        self.variance_domain.clear()
        return super().run(*args, **kwargs)

    def trainable_payload(self):
        result = {key: value.detach().clone() for key, value in self.P0.items()}
        for key in PKEYS:
            result[key].requires_grad_(True)
        assert sum(result[key].numel() for key in PKEYS) == 45200
        return result

    def diagnostics(self):
        return dict(retirement={str(k): v for k, v in self.retirement.items()},
            checkpoint_retirement_visits={str(k): v for k, v in self.replay_counts.items()},
            selections={str(k): v for k, v in self.selection_cells.items()},
            variance_domain={str(k): v for k, v in self.variance_domain.items()},
            physical_column_cap=8192, signed_pair_cap=4096)
