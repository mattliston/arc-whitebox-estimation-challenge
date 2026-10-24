#!/usr/bin/env python3
"""TAG fable-covhunter (CH-44): compare an aux refit run with its w=0 control, epoch by epoch, on VALID16 final MSE (per-net paired),
plus the aux run's own L10-14 VALID MSEs against its init. usage: aux_compare.py <aux log.json> <control log.json> [epoch]"""
import sys, json, numpy as np
M, C = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); ep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
m0, c0 = M['valid_before'], C['valid_before']; m1, c1 = M['epochs'][ep]['valid'], C['epochs'][ep]['valid']
nets = list(c1); mean = lambda d: float(np.mean([d[r] for r in nets]))
print(f'VALID16 final MSE: before aux {mean(m0):.6e} control {mean(c0):.6e}; epoch{ep} aux {mean(m1):.6e} ({mean(m1)/mean(m0)-1:+.3%}) '
      f'control {mean(c1):.6e} ({mean(c1)/mean(c0)-1:+.3%})')
rel = np.array([m1[r] / c1[r] - 1 for r in nets])
print(f'aux vs control at epoch{ep}: pooled {mean(m1)/mean(c1)-1:+.3%}, per-net mean {rel.mean():+.3%} sd {rel.std(ddof=1):.3%} '
      f'(se {rel.std(ddof=1)/np.sqrt(len(rel)):.3%}), aux better on {int((rel < 0).sum())}/{len(rel)}')
D0, D1 = M['aux_diag']['valid_before'], M['aux_diag'][f'valid_epoch{ep}']
for l in sorted(next(iter(D0.values()))):
    a0, a1 = np.mean([D0[r][l] for r in nets]), np.mean([D1[r][l] for r in nets])
    print(f'  aux run L{l}: {a0:.4e} -> {a1:.4e} ({a1/a0-1:+.2%}), better on {sum(D1[r][l] < D0[r][l] for r in nets)}/{len(nets)}')
