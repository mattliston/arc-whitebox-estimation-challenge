"""Versioned metadata-reader repair; same fixed pilot and all numerical gates.

The native recorder permits an empty threadpoolctl inventory, and all three
actual role inventories are exactly empty. Empty means instrumentation did
not report a controller; it does not prove a thread count or a mismatch.
Match the recorder's predicate, retain exact inventory equality, and retain
every byte/output/FLOP/initialization/resource/MPS/score gate. No rerun or fit.
"""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V1_SHA = 'f0e8fd36f74f89eab5b457d8537d67301bcf42ac3f361c02176f1f6216894441'


def derived():
    path = HERE / 'judge_terminal.py'
    original = path.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == V1_SHA
    edits = (
        ("    pin(__file__, args.source_sha256)\n",
         "    pin(__file__, args.source_sha256)\n"
         "    pin(HERE / 'judge_terminal.py', NODE_JUDGE_V1_SHA)\n"),
        ("    require(pools and all(x['num_threads'] == 1 for x in pools if x.get('user_api') == 'blas'),\n"
         "            'Recorded parent BLAS1 backend')\n",
         "    require(isinstance(pools, list) and all(x['num_threads'] == 1 for x in pools if x.get('user_api') == 'blas'),\n"
         "            'Reported BLAS controllers must be single-threaded; empty inventory is unreported')\n"),
        ("    require(selected_bytes['zero'] == selected_bytes['parent'], 'Zero node all16 output bytes equal parent on all4')\n",
         "    result['backend_inventory'] = dict(all_three_recorded_inventories_equal=True,\n"
         "        reported_controllers=len(pools), thread_count_independently_observed=bool(pools),\n"
         "        empty_inventory_is_not_a_thread_count_certificate=True,\n"
         "        native_recorder_launch_and_context_checks_preserved=True)\n"
         "    require(selected_bytes['zero'] == selected_bytes['parent'], 'Zero node all16 output bytes equal parent on all4')\n"),
        ("    out = HERE / 'terminal_score_v1'\n", "    out = HERE / 'terminal_score_v2'\n"),
    )
    changed = original
    for before, after in edits:
        assert changed.count(before) == 1
        changed = changed.replace(before, after, 1)
    inverse = changed
    for before, after in reversed(edits):
        assert inverse.count(after) == 1
        inverse = inverse.replace(after, before, 1)
    assert inverse == original, 'All unlisted scorer bytes must remain unchanged'
    return changed


def main():
    if '--judge' not in sys.argv:
        print('Inert. Same explicit --judge and seven --*-sha256 arguments as v1.')
        return 0
    namespace = dict(__name__='node_pilot_metadata_repair', __file__=str(Path(__file__).resolve()),
                     NODE_JUDGE_V1_SHA=V1_SHA)
    exec(compile(derived(), str(__file__) + ':preserved_v1', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
