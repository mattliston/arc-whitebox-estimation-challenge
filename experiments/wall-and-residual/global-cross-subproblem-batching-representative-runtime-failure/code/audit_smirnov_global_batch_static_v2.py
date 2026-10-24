"""Versioned correction: public FNP zeros cost0; price concrete placement."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'audit_smirnov_global_batch_static_v1.py').read_text()
source = source.replace("'smirnov_global_batch_static_v1.json'", "'smirnov_global_batch_static_v2.json'")
before = "        self.charge('zeros', prod(shape))"
assert source.count(before) == 1
source = source.replace(before, "        self.charge('zeros', 0)")
before = '            record.update(largest_first_arena_bytes=4*high,'
after = '''            first_keys = {(r['first_fit_offset'], tuple(r['shape'])): r['size']
                          for r in kernel.arena.intervals}
            placed_keys = {(r['offset'], tuple(r['shape'])): r['size'] for r in placement}
            binding_delta = sum(placed_keys.values())-sum(first_keys.values())
            call_delta = len(placed_keys)-len(first_keys)
            record.update(largest_first_binding_flops=sum(placed_keys.values()),
                          first_fit_binding_flops=sum(first_keys.values()),
                          concrete_placement_flops=record['flops']+(binding_delta if repetition=='cold' else 0),
                          concrete_placement_counted_calls=record['counted_calls']+(call_delta if repetition=='cold' else 0),
                          largest_first_arena_bytes=4*high,'''
assert source.count(before) == 1
source = source.replace(before,after)
exec(compile(source,str(HERE/'audit_smirnov_global_batch_static_v1.py'),'exec'),
     dict(__file__=str(Path(__file__).resolve()),__name__=__name__))
