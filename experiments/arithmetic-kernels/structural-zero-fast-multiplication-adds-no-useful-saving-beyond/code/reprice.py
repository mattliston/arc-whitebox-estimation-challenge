"""Bounded scalar search on pinned historical censuses; no model calls."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import engine

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())


def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    start = time.monotonic()
    seal = read(HERE / 'inputs_sealed.json')
    for expected in seal['pins']:
        assert pin(expected['path']) == expected, expected['path']
    assert not (HERE / 'result.json').exists()
    fs = os.statvfs(HERE)
    assert fs.f_bavail * fs.f_frsize > 470_000_000_000
    qualified = read(HERE / 'qualification.json')
    guard = read(HERE / 'qualification_guard.json')
    assert qualified['passed'] and guard['completed'] and guard['whole_group_retired']
    assert guard['max_swap_growth_bytes'] == 0
    prior = read(seal['prior_result'])
    inputs = read(seal['census_inputs'])
    programs = read(inputs['smirnov_certificate']['path'])['records'] + [engine.winograd()]
    simplified = [dict(name=p['name'], shape=p['shape'], rank=p['rank'],
                       arithmetic=p['packed_linear_operations']) for p in programs]
    assert simplified == prior['families'][0]['programs']
    search = engine.Search(programs)
    records, roots = [], []
    for census in inputs['censuses']:
        data = read(census['path'])
        calls = data['outermost_mm_calls']
        assert len(calls) == census['expected_outermost_mm_calls'] == 295
        assert data['total_flops'] == census['expected_total_flops']
        assert sum(c['flops'] for c in calls) == data['outermost_mm_sum']
        rank = census['rank']
        old_base = next(r for r in prior['families'][0]['records'] if r['rank'] == rank)
        old_extended = next(r for r in prior['families'][1]['records'] if r['rank'] == rank)
        old_by_index = {r['index']: r for r in old_base['details']}
        details, saved = [], 0
        for i, call in enumerate(calls):
            assert call['input_dtypes'] == ['float32', 'float32']
            m, k = call['left']
            k2, n = call['right']
            assert k == k2
            shape = (m, k, n)
            if min(shape):
                row = search.best(shape)
                roots.append(shape)
                old_cost = old_by_index[i]['optimistic_best'][0]
                assert row['cost'] <= old_cost, (rank, i, shape, row['cost'], old_cost)
            else:
                row = dict(cost=call['flops'], leaf_bytes=0, kind='unchanged_empty')
            gain = max(0, call['flops'] - row['cost'])
            saved += gain
            details.append(dict(index=i, shape=shape, caller=call['caller'],
                                actual_call_flops=call['flops'], optimistic_cost=row['cost'],
                                selected_kind=row['kind'], selected_leaf_bytes=row['leaf_bytes'],
                                optimistic_saved=gain))
            if i % 50 == 0:
                print(json.dumps(dict(rank=rank, call=i, states=len(search.memo),
                                      elapsed_s=time.monotonic() - start)), flush=True)
        total = data['total_flops']
        incremental = saved - old_extended['optimistic_saved']
        records.append(dict(rank=rank, complete_cold_flops=total,
                            outermost_mm_calls=len(calls), optimistic_saved=saved,
                            optimistic_new_total=total - saved, optimistic_fraction=saved / total,
                            previous_base_saved=old_base['optimistic_saved'],
                            previous_extended_saved=old_extended['optimistic_saved'],
                            incremental_fraction_vs_base=(saved - old_base['optimistic_saved']) / total,
                            incremental_saved_vs_extended=incremental,
                            incremental_fraction_vs_extended=incremental / total,
                            passes_three_percent_incremental_gate=100 * incremental >= 3 * total,
                            above_arena_selected_calls=sum(x['selected_leaf_bytes'] > 1168900096 for x in details),
                            maximum_selected_leaf_bytes=max(x['selected_leaf_bytes'] for x in details),
                            details=details))
        print(json.dumps({k:v for k,v in records[-1].items() if k != 'details'}), flush=True)
    graph = search.selected_graph(roots)
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   completed=True, programs=simplified, records=records, selected_graph=graph,
                   cached_shapes=len(search.memo), selected_shapes=len(graph),
                   side_geometry_cache=search.geometry.side.cache_info()._asdict(),
                   opportunity_gate_passed=all(r['passes_three_percent_incremental_gate'] for r in records),
                   qualification='Optimistic arithmetic only, old censuses, omitted additional assembly/copy/setup/dispatch. No native candidate bill, accuracy, runtime or total workspace certificate.',
                   source_seal=pin(HERE / 'inputs_sealed.json'),
                   wall_s=time.monotonic() - start,
                   process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   numerical_imports=False, network_arrays_read=0, target_arrays_read=0,
                   estimator_calls=0, candidates_created=0, uploads=0)
    data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    retained = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    assert retained + len(data) <= 30_000_000
    with (HERE / 'result.json').open('xb') as out:
        out.write(data)
    print(json.dumps(dict(completed=True, gate=payload['opportunity_gate_passed'],
                         cached_shapes=len(search.memo), selected_shapes=len(graph),
                         wall_s=payload['wall_s'], result=pin(HERE / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
