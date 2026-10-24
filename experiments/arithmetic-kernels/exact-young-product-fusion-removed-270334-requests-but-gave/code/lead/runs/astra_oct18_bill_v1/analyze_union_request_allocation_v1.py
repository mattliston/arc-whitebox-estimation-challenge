"""Pure-Python tariff rollout; no SDK, numerical model, or endpoint selection."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

D = Path(__file__).resolve().parent
R = D.parent
ROOT = R / 'astra_oct18_lead_v1'
RUN = ROOT / 'final_union32_endtrim095_v1'
OUT = D / 'request_allocation_v1'
OUT.mkdir(exist_ok=True)
def literals(path, wanted):
    answer = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try: answer[target.id] = ast.literal_eval(node.value)
                    except ValueError: pass
    return answer
def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value, (tuple, list)) else value
parent = R / 'astra_oct18_accuracy_v1/endtrim095_v1/candidate_union_sidebatch_endtrim095_HELD_v1.py'
current = literals(parent, {'_O18_UNION_RULES'})['_O18_UNION_RULES']
seed = literals(D / 'candidate_v61j_union_shared_HELD_v2.py', {'_O18_UNION_RULES', '_O18_UNION_NEW_FAMILIES'})
families = {f[0]: f for f in seed['_O18_UNION_NEW_FAMILIES']}
families['peel6_p10_5_5_26'] = ('peel6_p10_5_5_26', 6, 10, 5, 5, 26)
canonical = {
    (5, 20): ('o18union_L5P20_9_9_44', 5, 20, 9, 9, 44),
    (5, 40): ('o18union_L5P40_9_9_41', 5, 40, 9, 9, 41),
    (6, 10): ('oct18u_L6P10_5_5_29', 6, 10, 5, 5, 29),
    (6, 20): ('oct18u_L6P20_5_5_21', 6, 20, 5, 5, 21),
}
def remap(plan):
    f = families[plan[1]]
    target = canonical.get(f[1:3])
    if target is None or any(a > b for a, b in zip(f[3:], target[3:])):
        return None
    return (plan[0], target[0]) + plan[2:]
priced = defaultdict(list)
for key, options in seed['_O18_UNION_RULES'].items():
    for gain, cost, plan in options:
        p = remap(plan)
        if p: priced[key, p].append((gain, cost))
build = json.loads((D / 'final_union32_v1/BUILD_v2.json').read_text())
for path, sha in build['price_files'].items():
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    table = json.loads(path.read_text())
    tasks = json.loads((path.parent / 'TASKS_v1.json').read_text())['tasks']
    assert len(table['rows']) == len(tasks)
    for row, task in zip(table['rows'], tasks):
        family = freeze(task['family'])
        families[family[0]] = family
        plan = remap(freeze(row['plan']))
        if plan:
            m, k, n, q = row['shape']
            priced[(m, k, n + q), plan].append((int(row['gain']), max(0, int(row['request_delta']))))
groups = defaultdict(set)
for (key, plan), prices in priced.items():
    gain = min(g for g, c in prices)
    cost = max(c for g, c in prices)
    if gain > 0:
        groups[key].add((gain, cost, plan))
all_rules = {}
for key, options in groups.items():
    all_rules[key] = tuple(sorted((v for v in options if not any(
        w[0] >= v[0] and w[1] <= v[1] and (w[0] > v[0] or w[1] < v[1])
        for w in options)), key=lambda v: (-v[0], v[1], v[2])))
controls = {r['rank']: r for r in map(json.loads, (RUN / 'control32/rows.jsonl').read_text().splitlines())}
candidates = list(map(json.loads, (RUN / 'candidate32/rows.jsonl').read_text().splitlines()))
assert len(candidates) == len(controls) == 32
data = []
for row in candidates:
    shapes = row['jun_stats']['shapes'][-15:]
    keys = [(m, k, n + q) for m, k, n, q in shapes]
    data.append((row, controls[row['rank']], keys))

def greedy(rules, keys, cap, threshold=0, penalty=False):
    left = cap
    selected = []
    for layer, key in enumerate(keys, 1):
        options = [v for v in rules.get(key, ()) if v[1] <= left and v[0] >= threshold * v[1]]
        if penalty:
            options.sort(key=lambda v: (-(v[0] - threshold * v[1]), v[1], -v[0]))
        if options:
            chosen = options[0]
            left -= chosen[1]
            selected.append((layer, key, chosen))
    return sum(v[2][0] for v in selected), cap - left, selected

def oracle(rules, keys, cap):
    # Upper bound only: this knows all15 future shapes, which the online
    # estimator does not necessarily know before spending its first call.
    states = {0: 0}
    for key in keys:
        next_states = dict(states)
        for cost, gain in states.items():
            for dg, dc, plan in rules.get(key, ()):
                if cost + dc <= cap:
                    next_states[cost + dc] = max(next_states.get(cost + dc, -1), gain + dg)
        states = next_states
    return max((gain, -cost) for cost, gain in states.items())

baseline = [greedy(current, keys, 500) for row, control, keys in data]
for (row, control, keys), (gain, cost, selected) in zip(data, baseline):
    assert cost == 500 - row['oct18_peel_state']['left']
    assert len(selected) == row['oct18_peel_state']['push']
projected14 = 1.4417177802437146e-9
best = 1.4515052706908603e-9
def summarize(values):
    increments = [(v[0] - old[0]) / row['flops'] for (row, control, keys), old, v in zip(data, baseline, values)]
    absolute = projected14 * (1 - statistics.mean(increments))
    return dict(union_bill_credit_pct=100 * statistics.mean(v[0] / control['flops'] for (row, control, keys), v in zip(data, values)),
                incremental_bill_gain_vs14ce_pct=100 * statistics.mean(increments),
                conditional_adjusted_score=absolute,
                conditional_adjusted_change_vs336867_pct=100 * (absolute / best - 1),
                mean_call_debit=statistics.mean(v[1] for v in values),
                max_call_debit=max(v[1] for v in values),
                count_higher_credit=sum(v[0] > old[0] for v, old in zip(values, baseline)))
results = {'baseline500': summarize(baseline)}
for cap in (500, 750, 1000, 10000):
    label = 'unlimited' if cap == 10000 else str(cap)
    values = [(gain, -cost) for row, control, keys in data for gain, cost in [oracle(all_rules, keys, cap)]]
    results['hindsight_oracle_' + label] = summarize(values)
    results['greedy_' + label] = summarize([greedy(all_rules, keys, cap) for row, control, keys in data])
fixed = []
for cap in (500, 750, 1000):
    for threshold in (0, 500000, 1000000, 2000000, 3000000, 4000000, 5000000, 10000000):
        for penalty in (False, True):
            values = [greedy(all_rules, keys, cap, threshold, penalty) for row, control, keys in data]
            fixed.append(dict(cap=cap, tariff_penalty_per_call=threshold, choose_net_credit=penalty, **summarize(values)))
best_fixed = [max((v for v in fixed if v['cap'] == cap), key=lambda v: v['incremental_bill_gain_vs14ce_pct']) for cap in (500,750,1000)]
receipt = dict(status='TARIFF_ONLY_NO_ACCURACY_OR_WALL_CLAIM', models=0, SDK_calls=0,
               source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               geometry_source='Exact14ce candidate32; baseline500 selections reproduce observed hit count/debit on every row.',
               rule_keys=len(all_rules), plan_count=len({v[2] for opts in all_rules.values() for v in opts}),
               results=results, best_fixed_online_policies=best_fixed, fixed_online_sweep=fixed,
               notes=['Every candidate is an already-priced route within the existing four canonical capacities.',
                      'Hindsight oracle is a ceiling, not an implementable online policy.',
                      'Conditional projections keep14ce accuracy and all other bill costs unchanged; changed kernels may change rounding.',
                      'Higher cap adds prediction requests; current setup-only request savings do not fund prediction residual.',
                      'Additional retained plans may require setup inventory and own-prebind updates.'])
(OUT / 'RESULT_v1.json').open('x').write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'results': results, 'best_fixed_online_policies': best_fixed}, indent=2))
