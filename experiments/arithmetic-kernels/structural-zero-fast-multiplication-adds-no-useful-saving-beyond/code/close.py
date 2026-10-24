"""Close the completed symbolic screen, preserve failures and retire ownership."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[4]
A = D.parent / 'astra_phase2_reexamination_v1'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve(); b = path.read_bytes()
    return dict(path=str(path), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def check(item):
    assert pin(item['path']) == {k:item[k] for k in ('path', 'bytes', 'sha256')}


def write(path, data):
    with Path(path).open('x') as out:
        json.dump(data, out, indent=2, allow_nan=False); out.write('\n')


def main():
    assert not (HERE / 'closeout.json').exists()
    for item in read(HERE / 'inputs_sealed.json')['pins']:
        check(item)
    result = read(HERE / 'result.json'); replay = read(HERE / 'replay.json')
    assert result['completed'] and not result['opportunity_gate_passed']
    assert replay['passed'] and replay['selected_nodes_replayed'] == result['selected_shapes'] == 1753
    assert replay['census_calls_replayed'] == 590 and not replay['opportunity_gate_passed']
    assert replay['result'] == pin(HERE / 'result.json')
    names = ['qualification_guard.json.before_uniform_fallback', 'qualification_guard.json',
             'search_guard.json', 'replay_guard.json']
    guards = [read(HERE / name) for name in names]
    for g in guards:
        assert g['completed'] and g['exit_code'] == 0 and g['error'] is None
        assert g['whole_group_retired'] and g['remaining_rows'] == []
        assert g['max_swap_growth_bytes'] == 0
    pids = {g['pid'] for g in guards}
    process_rows = []
    for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,command='], text=True, timeout=5).splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) == 5:
            process_rows.append(dict(pid=int(fields[0]), ppid=int(fields[1]), pgid=int(fields[2]),
                                     rss_bytes=1024 * int(fields[3]), command=fields[4]))
    owned = [r for r in process_rows if r['pid'] in pids or r['pgid'] in pids]
    assert owned == []
    bank_path = D / 'official_best_331824.json'; bank = read(bank_path)
    assert bank['completed'] == 100 and bank['failed'] == 0 and bank['submission_id'] == 331824
    assert bank['adjusted'] == 3.5070948899311718e-9
    for name in ('candidate', 'archive', 'evaluation', 'completion_audit'):
        check(bank[name])
    grouping_path = D / 'smirnov_operand_grouping_audit_v1/result.json'
    grouping = read(grouping_path)
    assert grouping['all_twelve_operand_sides_have_40_distinct_projective_forms']
    assert grouping['exact_tensor_coefficients_checked'] == 17496
    for name in ('source', 'audit_source', 'protocol'):
        check(grouping[name])
    graph = result['selected_graph']; fringe_occurrences = []
    for shape, row in graph.items():
        if row['kind'] == 'recursive_peel':
            for s in row['fringes']:
                key = ','.join(map(str, s))
                fringe_occurrences.append((key, graph[key]['kind']))
    diagnostic = dict(selected_graph_fringe_occurrences=len(fringe_occurrences),
                      unique_fringe_shapes=len(set(x[0] for x in fringe_occurrences)),
                      fringe_kind_counts=dict(Counter(x[1] for x in fringe_occurrences)),
                      whole_zero_products_skipped_in_selected_graph=replay['selected_graph_zero_products_skipped'],
                      scope='Counts on the stored selected DAG, not invocation-weighted counts or universal bounds.')
    assert diagnostic['fringe_kind_counts'] == {'gemm': 889}
    assert diagnostic['unique_fringe_shapes'] == 708
    assert diagnostic['whole_zero_products_skipped_in_selected_graph'] == 0
    write(HERE / 'selected_graph_diagnostic.json', diagnostic)
    fs = os.statvfs(HERE); free = fs.f_bavail * fs.f_frsize
    assert free > 470_000_000_000
    peak = max(result['process_high_water_rss_bytes'], *(g['max_rss_bytes'] for g in guards))
    assert peak < 2_000_000_000
    lines = []
    for row in result['records']:
        lines.append(f"| {row['rank']} | {row['optimistic_new_total']:,} | "
                     f"{100*row['optimistic_fraction']:.6f}% | "
                     f"{100*row['incremental_fraction_vs_base']:.6f}% | "
                     f"{100*row['incremental_fraction_vs_extended']:.6f}% |")
    report = '''# Structural-zero and recursive-fringe screen: closed negative

The fixed arithmetic opportunity gate failed on both archived Phase 2
censuses. The new family adds only 0.023312% / 0.018239% of the original
complete bills beyond the prior seven-program frontier. Its optimistic
totals remain worse than the existing extended45 frontier. No native kernel,
estimator, training, target read, candidate archive or upload follows.

| Trace | Optimistic remaining F | Saving vs original complete bill | Additional vs old seven-program family | Additional vs old extended45 family |
| --- | ---: | ---: | ---: | ---: |
''' + '\n'.join(lines) + f'''

The preregistered engineering gate required at least another3% of each
original complete bill beyond the known extended45 unrestricted result.
This is a source/cost screen, not an adjusted-score test. Both traces predate
the current331824 bank; they are not fresh candidate bills or a prediction
of the official score. New padding/assembly/copy/dispatch/setup costs,
finite-precision effects and runtime remain unqualified.

The search used112,875 cached shapes and selected1,753 distinct nodes. An
independent accounting program replayed those nodes and all590 original
calls. All889 selected fringe occurrences (708 unique shapes) use ordinary
matrix multiplication; no entire product is skipped by a zero operand in
the selected DAG. These are scoped observations, not impossibility claims.
The chosen minimum-cost plans also have24 /23 calls whose simultaneous-leaf
diagnostic exceeds the old1,168,900,096-byte arena; its maximum is2,531,065,600
bytes. This diagnostic omits total workspace and does not optimize under a
memory constraint. There is no feasible implementation claim.

All seven circuit tensor identities passed17,560 exact coefficient checks.
Ninety-one small, asymmetric two-level integer products matched ordinary
multiplication exactly, including every propagated support bound. Full-block
costs reproduce the source packed counts. Before whole-census execution an
explicit unpruned option was added to retain every old uniform path without
assuming cost monotonicity in dimensions. The first successful qualification
and pre-change sources are preserved; the final sources were requalified
and sealed before search. No observed cost result or scientific gate changed.

Four guard groups retired, zero added swap; maximum observed/self-reported
RSS{peak:,} bytes. The search took{result['wall_s']:.3f}s. Tool sessions
32364,62862,40593 are terminal0; replay completed in its initial tool call.
Do not poll or restart those handles. All records and original failed
studies remain preserved. Free disk at closeout{free/1e9:.3f}GB.

The separate exact operand audit confirms that each of the six rank40
circuits has40 distinct projective forms on BOTH input sides. It independently
rechecked17,496 tensor coefficients. Its fixed uniform-level recursion proof
rules out grouping equal or proportional generic operands at deeper levels;
padding and other algorithms are explicitly outside that proof. The present
cost study addresses only its own dimension-based bounding-rectangle family.

Official331824 remains3.5070948899311718e-9 adjusted,100/100 complete,
zero failures. The exact approved368577-byte archive still has SHA256
93f075402ae30c2a7996d53176081fb080ef6be8ab3f626516759b7f755c2496.
Standing user approval persists for future validated Phase2 improvements.
The4e-9 milestone is banked;3e-9,2e-9 and approximately1e-9 remain unmet.
The goal is active. No next numerical study is selected.
'''
    with (HERE / 'RESULT.md').open('x') as out:
        out.write(report)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase=2,
                   completed=True, integrity_passed=True, opportunity_gate_passed=False,
                   status='closed_negative_at_original_arithmetic_opportunity_gate',
                   selected_graph_diagnostic=pin(HERE / 'selected_graph_diagnostic.json'),
                   inputs=pin(HERE / 'inputs_sealed.json'), result=pin(HERE / 'result.json'),
                   replay=pin(HERE / 'replay.json'), report=pin(HERE / 'RESULT.md'),
                   guards=[pin(HERE / name) for name in names],
                   grouping_audit=pin(grouping_path), official_best=pin(bank_path),
                   candidate=bank['candidate'], approved_archive=bank['archive'],
                   source=pin(Path(__file__).resolve()), all_owned_groups_retired=True,
                   retired_pids=sorted(pids), owned_rows=owned, zero_added_swap=True,
                   peak_rss_bytes=peak, free_disk_bytes=free,
                   terminal_sessions=[32364,62862,40593],
                   numerical_imports=False, model_calls=0, target_arrays_read=0,
                   new_candidate=False, upload_attempted=False,
                   goal_active=True, goal_complete=False, next_numerical_study_selected=False)
    assert not {'numpy', 'torch', 'flopscope'} & set(sys.modules)
    write(HERE / 'closeout.json', receipt)
    print(json.dumps(dict(closed_negative=True, no_owned_workers=True,
                         closeout=pin(HERE / 'closeout.json'), peak_rss_bytes=peak,
                         free_disk_bytes=free)))


if __name__ == '__main__':
    main()
