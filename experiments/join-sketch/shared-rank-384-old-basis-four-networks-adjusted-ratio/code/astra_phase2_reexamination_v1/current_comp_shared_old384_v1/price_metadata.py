"""Stdlib-only integer tariff replay; never import an estimator or open an array.

Explicit execution prints a conditional geometry price, not a candidate bill.
It creates no files/stages. The checked-in PRICE.json is a source-planning record.
"""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
PARENT = APUSH/'build_v1/estimator.py'
PARENT_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
RESULT = APUSH/'native40_terminal_v1/result.json'
RESULT_SHA = '23b88f12449442cad00eb352e2aca1d90d277120b7ed348f800a9bcff64684a7'
RANK = 384


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dense(m, k, n):
    return m * n * (2 * k - 1) if m and k and n else 0


def packed(m, k, n, depth):
    unit = 2 ** depth
    assert all(v % unit == 0 for v in (m, k, n))
    q = Fraction(7, 4)
    value = (7 ** depth * (m // unit) * (n // unit) * (2 * k // unit - 1)
             + q * (m * k + k * n + m * n) * sum(q ** j for j in range(depth)))
    assert value.denominator == 1
    return int(value)


def matrix_price(m, k, n, tables, row_tiles=False):
    if not m or not k or not n:
        return 0
    if m > 1024:
        assert row_tiles and m <= 2048
        return m * n + sum(matrix_price(min(1024, m - j), k, n, tables)
                           for j in range(0, m, 1024))
    if n > 8192:
        return m * n + sum(matrix_price(m, k, min(8192, n - j), tables)
                           for j in range(0, n, 8192))
    if k > 8192:
        pieces = list(range(0, k, 8192))
        return ((len(pieces) - 1) * m * n
                + sum(matrix_price(m, min(8192, k - j), n, tables) for j in pieces))
    if m % 32 or k % 32 or n % 32:
        return dense(m, k, n)
    if max(m, k, n) <= 1024:
        if m == n:
            depth = tables['_SQUARE_EQUAL_DEPTHS'][m // 32 - 1][k // 32 - 1]
        elif (m == k and n == 1024) or (k == n and m == 1024):
            depth = tables['_SQUARE_CROSS_DEPTHS'][k // 32 - 1]
        else:
            depth = min((0, 3, 4, 5), key=lambda d: dense(m, k, n) if not d else packed(m, k, n, d))
        price = dense(m, k, n) if not depth else packed(m, k, n, depth)
        # Independently check the frozen table agrees with its exact source rule
        # on every pricing shape. Do not substitute an optimized depth silently.
        assert price == min(dense(m, k, n), *(packed(m, k, n, d) for d in (3, 4, 5)))
        return price
    depth = (6 if m == 1024 and k <= 1024 and n > 1024
             and k % 64 == 0 and n % 64 == 0 else 4)
    return packed(m, k, n, depth)


def calculate(tick=lambda: None):
    assert sha(PARENT) == PARENT_SHA and sha(RESULT) == RESULT_SHA
    tick()
    # Evaluate only the two integer metadata tables, never any parameter literal.
    tree = ast.parse(PARENT.read_text())
    tables = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('_SQUARE_EQUAL_DEPTHS', '_SQUARE_CROSS_DEPTHS')):
            tables[node.targets[0].id] = ast.literal_eval(node.value)
    assert len(tables) == 2
    del tree
    tick()
    record = json.loads(RESULT.read_text())
    assert record['passed'] and record['completed'] and record['full_model_calls'] == 41
    assert record['candidate_sha256'] == PARENT_SHA and record['initialization_flops'] == 1944970298
    rows, actual_dispatch_count = [], 0
    for call, trial in enumerate(record['trials'][:40]):
        packet = record['diagnostic_records'][call]
        path = Path(packet['path'])
        assert path.parent == RESULT.parent and path.name == f'call_{call:02d}_diagnostics.json'
        assert sha(path) == packet['sha256']
        diagnostic = json.loads(path.read_text())
        assert diagnostic['call'] == call and diagnostic['rank'] == trial['rank']
        for item in diagnostic['dispatches']:
            assert packed(*item['shape'], 6) == item['flops']
            actual_dispatch_count += 1
        cells = diagnostic['domains']['selection_cells']
        counts = [[cells[f'{layer}:0']['count'], cells[f'{layer}:1']['count'],
                   cells[f'{layer}:2']['count'] if layer < 15 else 0]
                  for layer in range(16)]
        old, layers = 0, []
        mm = lambda m, k, n: matrix_price(m, k, n, tables, row_tiles=True)
        for layer in range(5, 16):
            a, v, s = counts[layer - 1][0], counts[layer][0], counts[layer][1]
            total = 2 * sum(z[2] for z in counts[:layer])
            inherited = 2 * sum(z[2] for z in counts[:layer - 1])
            joining = 2 * counts[layer - 5][2] if layer < 15 else 0
            previous_old = old
            old += joining
            r, n = RANK, 1024
            assert a >= r and old <= inherited
            prices = dict(new_join_gram=0, old_gram_apply=0, qr=0,
                          join_projection=0, basis_overlap=0, old_factor_rotation=0,
                          core_rotation=0, new_core=0)
            if joining:
                prices['new_join_gram'] = mm(joining, a, r) + mm(a, joining, r)
                prices['qr'] = 4 * a * r * r - 4 * r ** 3 // 3
                prices['join_projection'] = mm(r, a, joining)
                prices['new_core'] = mm(r, joining, r)
                if previous_old:
                    prices['old_gram_apply'] = mm(r, a, r) + mm(r, r, r) + mm(a, r, r)
                    prices['basis_overlap'] = mm(r, a, r)
                    prices['old_factor_rotation'] = mm(r, r, previous_old)
                    prices['core_rotation'] = 2 * mm(r, r, r)
            prices.update(old_push=mm(n, a, inherited), old_read=mm(s, total, v),
                          young_push=mm(n, a, inherited - old), young_read=mm(s, total - old, v),
                          basis_push=mm(n, a, r), reconstruction=mm(n, r, old),
                          inner_read=mm(s, old, r), trailing_read=mm(s, r, v))
            # Conservative additional finite scans, casts, coefficient weights,
            # factor concatenates, extra young concatenation, gathers and copies.
            # No original p/diagonal/radial work is subtracted or declared free.
            prices['extra_scalar_copy_reserve'] = (16 * n * old + 2 * n * r + s * v
                + (12 * a * joining + 16 * a * r + 8 * r * r + 8 * joining if joining else 0))
            before = prices['old_push'] + prices['old_read']
            delta = sum(prices.values()) - 2 * before
            layers.append(dict(layer=layer, previous_active=a, active=v, response=s,
                               joining_signed_columns=joining, old_signed_columns=old,
                               total_signed_columns=total, prices=prices, delta=delta))
        # No new persistent pool is implemented. This deliberately unreclaimed
        # first-call allowance is a conservative planning reserve, not a billed op.
        extra_cold_reserve = 64000000 if trial['includes_initialization'] else 0
        projected = trial['flops'] + sum(z['delta'] for z in layers) + extra_cold_reserve
        rows.append(dict(call=call, rank=trial['rank'], counts=counts,
                         diagnostic_path=str(path), diagnostic_sha256=packet['sha256'],
                         actual_parent_flops=trial['flops'], extra_cold_reserve=extra_cold_reserve,
                         conditional_complete_projection=projected, layers=layers))
        tick()
    return dict(schema='current_comp_shared_old384_price_v1',
                numerical_arrays_read=False, estimator_executed=False, candidate_bill=False,
                conditional_on_parent_geometry=True, extra_incumbent_plane_repair=False,
                parent_sha256=PARENT_SHA, native40_result_sha256=RESULT_SHA,
                actual_primary_D6_dispatches_reconciled=actual_dispatch_count,
                actual_parent_mean_flops=sum(z['actual_parent_flops'] for z in rows) / 40,
                conditional_mean_flops=sum(z['conditional_complete_projection'] for z in rows) / 40,
                rows=rows)


if __name__ == '__main__':
    print(json.dumps(calculate(), sort_keys=True, indent=2, allow_nan=False))
