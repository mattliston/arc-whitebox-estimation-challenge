"""Stdlib-only prospective union policy; preserves the sealed fixed54 result."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PRICE = H2 / 'teammate_s32_rank15_dispatch_v1/price.py'
PRIOR = PRICE.with_name('result.json')
CERT = H2.parent / 'astra_phase2_reexamination_v1/catalog_executable_cse_certificate_v1.json'


def read(path):
    return json.loads(path.read_text())


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def compact_keys(table, used):
    keys = set()
    for key in used:
        q = table[key]
        d = q['depth']; a, b, c = q['child']; u = 2**d
        ca = 15*7**d*(a//u)*(b//u)
        cb = 15*7**d*(b//u)*(c//u)
        keys.update(((0,d,a,b,0), (1,d,b,c,ca), (2,d,a,c,ca+cb)))
    return keys


def main():
    prior = read(PRIOR)
    price_pin = next(p for p in prior['pins'] if Path(p['path']) == PRICE)
    assert pin(PRICE) == price_pin
    assert pin(PRIOR)['sha256'] == 'a771c76ddf8fc7de30c13c7d9b5fa65c6a28826ba1210b52292959fa2c674eac'
    assert pin(HERE/'assess.py')['sha256'] == '9979e639b513ecb1c2db14496eaaa15d0e05611f4e08bab14a0beb7d9f5fe7e2'
    assert pin(CERT) == next(p for p in prior['pins'] if Path(p['path']) == CERT)
    # Load only the two unchanged arithmetic function definitions, never the
    # old driver's top-level census processing or exclusive result writer.
    syntax = ast.parse(PRICE.read_text())
    definitions = [n for n in syntax.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('binary', 'alternative')]
    assert [n.name for n in definitions] == ['binary', 'alternative']
    arithmetic = {'Fraction': Fraction}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PRICE), 'exec'), arithmetic)
    spec = importlib.util.spec_from_file_location('frozen_hk_owner_assessor', HERE/'assess.py')
    assessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assessor)
    programs = [r for r in read(CERT)['records'] if r['rank'] == 15]
    assert len(programs) == 6 and all(r['passed'] for r in programs)
    old = {tuple([r['depth']]+r['shape']): r['alternative']
           for r in prior['one_algorithm_shape_dispatch']['positive']}
    assert len(old) == 54
    traces = {}
    observed = set()
    for rank in (35, 100):
        receipt = read(HERE/f'row_{rank}.json')
        assert receipt['passed'] and receipt['all16_bytes_exact'] and receipt['full_bill_exact']
        assert pin(HERE/f'trace_{rank}.jsonl') == receipt['trace']
        with (HERE/f'trace_{rank}.jsonl').open() as stream:
            events = [json.loads(line) for line in stream]
        traces[rank] = events
        observed.update(tuple([e['depth']]+e['shape']) for e in events
                        if e['kind'] == 'product' and e['supported'])
    table = {}
    for key in sorted(observed):
        depth, m, k, n = key
        choices = [q for p in programs for d in range(7)
                   if (q := arithmetic['alternative'](p,d,m,k,n)) is not None]
        best = min(choices, key=lambda q: (q['flops'], q['name'], q['depth']))
        if best['flops'] < sum(arithmetic['binary'](depth,m,k,n)):
            table[key] = best
    assert set(old).issubset(observed)
    assert all(table[k] == q for k, q in old.items())
    fixed = read(HERE/'assessment.json')
    rows = []; used_all = set(); fixed_used_all = set()
    for rank in (35, 100):
        row = assessor.assess_row(read(HERE/f'row_{rank}.json'), table)
        old_row = next(r for r in fixed['rows'] if r['rank'] == rank)
        selected = {r['root'] for r in row['proposals']}
        old_selected = {r['root'] for r in old_row['proposals']}
        used = set(); count = 0; new_events = []
        for e in traces[rank]:
            if e['kind'] != 'product' or not e['supported']:
                continue
            key = tuple([e['depth']]+e['shape'])
            if e['root'] in old_selected and key in old:
                fixed_used_all.add(key)
            if e['root'] in selected and key in table:
                used.add(key); count += 1
                if key not in old:
                    new_events.append(dict(event=e['event'],root=e['root'],key=key))
        used_all.update(used)
        row.update(HK_product_calls=count, used_positive_keys=sorted(used),
                   newly_covered_product_events=new_events,
                   prior_fixed54_net=old_row['conditional_warm_saving'],
                   incremental_net=row['conditional_warm_saving']-old_row['conditional_warm_saving'])
        rows.append(row)
    keys = compact_keys(table, used_all)
    fixed_keys = compact_keys(old, fixed_used_all)
    all_table_keys = compact_keys(table, set(table))
    cold = sum(15*(7**d+7*4**(d-1))*(x//2**d)*(y//2**d)
               for _,d,x,y,offset in keys if d)
    output = dict(
        status='SEPARATE_PROSPECTIVE_SHAPE_UNION_PRICE_NOT_A_KERNEL_OR_MEASURED_GAIN',
        prior_fixed54_unchanged=True, metadata_only=True, numerical_jobs=0,
        algorithm='Same certified Hopcroft-Kerr rank15, same six orientations',
        policy='Union of supported binary shape/depth metadata in untouched35/100 traces; exact old cost rule, depths0..6, leaf>=4; ties(flops,name,depth). Complete-owner eligibility and loss simulation unchanged.',
        observed_binary_shape_keys=len(observed), positive_shape_keys=len(table),
        positive_shape_keys_added=len(set(table)-set(old)),
        table=[dict(key=k,alternative=q) for k,q in sorted(table.items())],
        rows=rows, mean_conditional_warm_pct=sum(r['conditional_warm_pct'] for r in rows)/2,
        required_actual_compact_keys=sorted(keys), required_actual_compact_key_count=len(keys),
        required_actual_side_counts=dict(Counter(k[0] for k in keys)),
        prior_fixed54_actual_compact_key_count=len(fixed_keys),
        compact_keys_added_vs_actual_fixed54=sorted(keys-fixed_keys),
        all_positive_table_compact_key_count=len(all_table_keys),
        unused_W5_only_positive_keys=sorted(set(table)-used_all),
        cold_two_reshape_flops_if_required_keys_unbound=cold,
        capacity636_remaining_old_slots=636-len(keys),
        setup_selection_constructed=False, UBC_overlap_rechecked=False,
        limitations=[
            'This prospective policy does not overwrite or reinterpret the sealed fixed54 policy/assessment.',
            'Shape selection uses no targets or prediction values and no per-layer exceptions.',
            'Negative complete-owner contributions remain included when the fixed shape rule selects any internal product.',
            'Cold/setup bind transfers, evictions, extra fused/DAG metadata, and old-key replacement debit are not included in warm pricing.',
            'The206-side-key count is for actual selected non-W5 owners; all-table allocation additionally includes excluded-only shapes.',
            'UBC/prebind component transfer needs overlap rechecking for this distinct policy; no bundle credit added.',
            'Native FP behavior, downstream routing, runtime, RSS, handles and setup completion remain unqualified.'
        ],
        pins=[pin(p) for p in (Path(__file__), PRICE, PRIOR, CERT, HERE/'assess.py',
              HERE/'assessment.json', HERE/'guard.json', HERE/'trace_35.jsonl', HERE/'trace_100.jsonl',
              HERE/'row_35.json', HERE/'row_100.json')])
    dest = HERE/'prospective_union_price_v1.json'
    with dest.open('x') as stream:
        json.dump(output, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(output=pin(dest),mean_pct=output['mean_conditional_warm_pct'],
                         keys=len(keys),all_table_keys=len(all_table_keys),cold_unbound=cold)))


if __name__ == '__main__':
    main()
