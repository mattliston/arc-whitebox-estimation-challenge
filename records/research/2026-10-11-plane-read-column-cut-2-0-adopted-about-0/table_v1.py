"""TAG opus_engineer a62 module bill table (lead 10-11 RESTART): the module census of the fired S47e file (sha 01103249) split by
module and band (layer 0 / 1-6 / 7-10 / 11-15), in percent of each network's bill, mean over the census networks.

usage: table_v1.py <census.json>
The rules read the main loop's line and the frames under it; the line numbers are those of stack/rl/s47rsd3x_safe/candidate.py
(the script refuses another file). Inside the packed pair series one product op covers five (X2) or four (X3) term pairs: X2's
share is split 1/5 per pair (pair 3 = the (2,2) plane, pairs 4-5 = the (3,1) plane), X2's row-1 rescale and sums likewise.
"""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
assert d['sha256'].startswith('01103249'), d['sha256']
BANDS = (('L0', 0, 0), ('1-6', 1, 6), ('7-10', 7, 10), ('11-15', 11, 15))


def band(layer):
    for name, lo, hi in BANDS:
        if lo <= layer <= hi:
            return name
    return 'outside'


def names(chain):
    return [c[0] for c in chain]


def series_split(chain):
    """Packed / full pair series: [(module, fraction)]."""
    inner = [c for c in chain if c[0] == '_psp_series']
    outer = [c for c in chain if c[0] == '_psp_pair_series']
    if inner:
        line = inner[0][1]
        if line in (12637, 12638, 12641):          # five term pairs (X2): t21, t12, t22, t31, t13
            return [('pair map: base series', 0.4), ('K22 plane (2,2)', 0.2), ('K31 plane (3,1) row factor', 0.4)]
        if line == 12639:                          # pair 1's rho rescale (t12)
            return [('pair map: base series', 1.0)]
        if line == 12651:
            return [('K22 plane (2,2)', 1.0)]
        if line == 12652:
            return [('K31 plane (3,1) row factor', 1.0)]
        return [('pair map: base series', 1.0)]
    if outer:
        line = outer[0][1]
        if line == 12677:
            return [('K22 plane (2,2)', 1.0)]
        if line in (12678, 12679):
            return [('K31 plane (3,1) row factor', 1.0)]
    return [('pair map: base series', 1.0)]


def classify(layer, line, chain):
    """[(module, fraction)] for one census key."""
    nm = names(chain)
    first = nm[0] if nm else None
    fl = chain[0][1] if chain else 0
    if line == 0:
        return [('outside the layer loop', 1.0)]
    # ---- the bank: pushes and reads
    if first in ('_deadcut_push', '_l15_rowcut_k3'):
        if first == '_l15_rowcut_k3' and '_k4l_local' in nm:
            return [('K4L', 1.0)]
        if '_push_bank_shared_newborn' in nm:
            return [('BASE bank push: young columns', 1.0)]
        if '_shared_old384_push' in nm:
            return [('BASE bank push: old tier', 1.0)]
        if first == '_l15_rowcut_k3':
            return [('BASE bank third order (pp^2, pp^3, diagonals)', 1.0)]
        return [('BASE bank push: young columns', 1.0)]          # row expansion of the pushed columns
    if first == '_cp_read_young_sq':
        if fl in (5118, 5119):
            return [('BASE bank read: old tier', 1.0)]
        return [('BASE bank read: young square', 1.0)]
    if first == '_dcr_right' or line in (3308, 3309, 3310, 3311):
        return [('BASE bank read: young square', 1.0)]
    if first == '_np_left':
        if '_hc_column' in nm:
            return [('HEAD COLUMNS (hf): diag', 1.0)]
        if fl in (11598, 11602, 11608, 11683):
            return [('BASE bank read: young square', 1.0)]  # the read's left factor (response rows of pp^2)
        if 11615 <= fl <= 11663 or 11672 <= fl <= 11674:
            return [('K31L (law k3l: fourth-chaos plane of young pairs)', 1.0)]
        return [('LAW (pair law of the norm coupling, NP fold)', 1.0)]
    if first == '_np_forms':
        return [('LAW (pair law of the norm coupling, NP fold)', 1.0)]
    if first == '_np_update':
        if 11779 <= fl <= 11783:
            return [('K31L (law k3l: fourth-chaos plane of young pairs)', 1.0)]
        return [('LAW (pair law of the norm coupling, NP fold)', 1.0)]
    # ---- covariance sandwich and star
    if first in ('_dc_cov_star', '_flms_cov_star', '_resident_gram512', '_dc_cov_diagonal', '_cov_compact_diagonal') \
            or line in (3266, 3267, 3325, 3326, 3327):
        return [('BASE covariance sandwich + transported star', 1.0)]
    # ---- learned head, its columns, the edge
    if first in ('hook', 'scalar_terminal_hook'):
        if '_hc_bias' in nm:
            return [('HEAD COLUMNS (hf): bias term', 1.0)]
        if '_e2_bdnh_coefficients' in nm:
            return [('HEAD: learned birth coefficients (bdnh)', 1.0)]
        if '_e2_messages' in nm:
            return [('HEAD: messages (shared with EDGE at 8-14)', 1.0)]
        if 'features' in nm or 'final_scalar_features' in nm:
            return [('HEAD: features', 1.0)]
        if 'gnn_step' in nm:
            gl = chain[nm.index('gnn_step')][1]
            if gl in (749, 750):
                return [('HEAD: output projection (delta)', 1.0)]
            return [('HEAD: encoder + message squares', 1.0)]
        return [('HEAD: features', 1.0)]
    if 3380 <= line <= 3400:
        return [('HEAD: rank-2 update of C by the mean correction', 1.0)]
    if first == 'message_covariance_hook' or 3373 <= line <= 3377:
        if '_e2_messages' in nm:
            return [('HEAD: messages (shared with EDGE at 8-14)', 1.0)]
        return [('EDGE (learned covariance edge, 8-14)', 1.0)]
    if first in ('_b6_old', '_b6_qb', '_b6_m2') or line == 3345:
        return [('HEAD COLUMNS (hf): b6 eold / qb / m2, k4_local_rows', 1.0)]
    # ---- physics modules
    if first in ('_f4c_step', '_f4c_birth') or line in (3475, 3476, 3477, 3491, 3492):
        return [('F4C (column shift: rule + exact S2)', 1.0)]
    if first in ('_k4l_local', '_k4l_dk', '_k4l_birth', '_k4l_final') or line in (3343, 3344):
        return [('K4L', 1.0)]
    if first == '_gn_dcov' or line == 3364:
        return [('COVKNOB (GN covariance correction: iso, k21n, k31)', 1.0)]
    if first == '_bsk':
        return [('PBASE (bsk: kink density skew term)', 1.0)]
    if line in (3410, 3411):
        return [('BFIX', 1.0)]
    if line in (3348, 3349, 3350, 3353, 3354):
        return [('K22 plane (2,2)', 1.0)]
    if first in ('_nonlinear_pair_batch', '_nonlinear_first_gaussian', 'nonlinear_scalar') or line in (3357, 3358, 3359, 3360, 3361, 3362):
        if first == '_nonlinear_pair_batch':
            if '_psp_pair_series' in nm or '_psp_series' in nm:
                return series_split(chain)
            if 914 <= fl <= 918:
                return [('K22 plane (2,2)', 1.0)]
            if 919 <= fl <= 925 or '_k31_row_scale' in nm:
                return [('K31 plane (3,1) row factor', 1.0)]
        return [('BASE pair map (moments, Hermite rows, post planes)', 1.0)]
    if first == 'omitted_radial_source' or 3368 <= line <= 3372:
        return [('BASE radial source', 1.0)]
    if first == '_bd_update' or 3450 <= line <= 3458:
        return [('BASE births: bd update', 1.0)]
    if line in (3304, 3314, 3317, 3318, 3319, 3320, 3321, 3322):
        return [('BASE bank third order (pp^2, pp^3, diagonals)', 1.0)]
    return [('BASE vectors, K21 assembly, births, star, lam', 1.0)]


tot = {n['rank']: n['flops'] for n in d['nets']}
nets = len(tot)
share = collections.defaultdict(lambda: collections.Counter())
ops = collections.defaultdict(lambda: collections.Counter())
for rank, layer, line, chain, op, fl, n in d['rows']:
    for module, frac in classify(layer, line, chain):
        share[module][band(layer)] += 100.0 * frac * fl / tot[rank] / nets
        share[module]['all'] += 100.0 * frac * fl / tot[rank] / nets
        ops[module]['all'] += frac * n / nets
cols = [b[0] for b in BANDS] + ['outside', 'all']
order = sorted(share, key=lambda m: (not m.startswith('BASE'), -share[m]['all']))
print('networks', sorted(tot), 'bill', {r: '%.4e' % f for r, f in tot.items()})
print('%-58s ' % 'module (% of the network bill)' + ' '.join('%7s' % c for c in cols) + '   ops/net')
for m in order:
    print('%-58s ' % m[:58] + ' '.join('%7.3f' % share[m][c] for c in cols) + '   %7.0f' % ops[m]['all'])
print('%-58s ' % 'TOTAL' + ' '.join('%7.3f' % sum(share[m][c] for m in share) for c in cols))
json.dump(dict(census=sys.argv[1], nets=d['nets'], share={m: dict(share[m]) for m in share}, ops={m: ops[m]['all'] for m in ops}),
          open(sys.argv[1].replace('.json', '_table_v1.json'), 'x'), indent=1)
