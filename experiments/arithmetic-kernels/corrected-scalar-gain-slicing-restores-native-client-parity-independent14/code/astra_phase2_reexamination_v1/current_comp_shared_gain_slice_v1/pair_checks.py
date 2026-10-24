"""Small stdlib-only actual-output/bill checks; no estimator or truth loading."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct

RANKS = (35, 35)
BUDGET = 2**41
VECTOR_TOLERANCE = 1e-5


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2,
            'Regular bounded metadata')
    return json.loads(path.read_text())


def raw_f32(path, npy=False):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 100_000,
            'Bounded actual all16 vector, never a target or weight file')
    raw = path.read_bytes()
    if npy:
        require(raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00'),
                'Actual uncompressed NPY v1/v2')
        wide = raw[6] == 2
        width = 4 if wide else 2
        size = struct.unpack('<I' if wide else '<H', raw[8:8+width])[0]
        header = ast.literal_eval(raw[8+width:8+width+size].decode('latin1'))
        require(header == dict(descr='<f4', fortran_order=False, shape=(16,1024)),
                'Exact native FP32 all16 row-major output')
        raw = raw[8+width+size:]
    require(len(raw) == 65_536, 'Exact all16 FP32 payload length')
    values = struct.unpack('<16384f', raw)
    require(all(math.isfinite(x) and x >= 0 for x in values),
            'All16 actual outputs finite and nonnegative')
    return raw, values


def check_bill(row, index):
    """Unchanged production safety gates, specialized to the fixed pair."""
    require(type(index) is int and 0 <= index < 2 and row['index'] == index
            and row['rank'] == RANKS[index] and row['output_shape'] == [16,1024]
            and row['output_dtype'] == 'float32', 'Exact pair context/output identities')
    f, timing = row['flops'], row['timing']
    require(type(f) is int and 0 < f <= BUDGET and row['util'] == f/BUDGET,
            'Own complete context FLOPs, no subtracted preparation or startup')
    require(isinstance(timing, dict) and timing and all(type(v) in (int,float)
            and math.isfinite(v) and v >= 0 for v in timing.values())
            and 0 < timing['wall_time_s'] < 60 and timing['residual_wall_time_s'] < .4,
            'Original production wall<60s/residual<.4s gates')
    require(re.fullmatch('[0-9a-f]{64}', row['output_sha256'] or ''),
            'Actual returned output SHA')


def compare(native, rows, remote_paths, tick):
    """Called only after both complete pairs and processes have retired."""
    require(native['full_model_calls'] == native['retired_call_count'] == 2
            and native['ranks'] == [35,35] and len(native['trials']) == len(rows) == 2,
            'Both actual fixed pairs, not cached replacement calls')
    checks, native_raw, remote_raw = [], [], []
    old_production = native['control_seals']['old_production_controls']
    require(len(old_production) == 2 and [x['call'] for x in old_production] == [0,4]
            and all(x['rank'] == 35 for x in old_production),
            'Actual saved original cold35/repeat35 records, not inferred old prices')
    for index, (trial, row, path) in enumerate(zip(native['trials'], rows, remote_paths, strict=True)):
        check_bill(row, index)
        a, av = raw_f32(trial['output'], npy=True)
        b, bv = raw_f32(path)
        require(hashlib.sha256(b).hexdigest() == row['output_sha256']
                and hashlib.sha256(a).hexdigest() == trial['output_bytes_sha256'],
                'Actual sealed source-owned output payloads')
        native_raw.append(a); remote_raw.append(b)
        differences = [x-y for x,y in zip(av,bv,strict=True)]
        layers = [max(abs(x) for x in differences[i*1024:(i+1)*1024]) for i in range(16)]
        checks.append(dict(call=index, rank=35, all16_max_abs=max(layers),
            layer_max_abs=layers, output_byte_exact=a == b,
            final_prediction_difference_mse=sum(x*x for x in differences[-1024:])/1024,
            native_flops=trial['flops'], production_flops=row['flops'],
            complete_flops_delta=trial['flops']-row['flops'],
            saved_old_production_flops=old_production[index]['flops'],
            new_minus_old_production_flops=row['flops']-old_production[index]['flops'],
            predicted_new_minus_old_production_flops=15090,
            original_all16_vector_gate_passed=max(layers) <= VECTOR_TOLERANCE))
        tick()
    require(native_raw[0] == native_raw[1] and remote_raw[0] == remote_raw[1],
            'Byte-exact complete native and production own repeats')
    return dict(checks=checks, all16_vector_gate_passed=all(
        x['original_all16_vector_gate_passed'] for x in checks),
        native_repeat_byte_exact=True, production_repeat_byte_exact=True,
        predicted_production_tariff_shift_observed=all(
            x['new_minus_old_production_flops'] == 15090 for x in checks),
        native_initialization_flops=native['trials'][0]['flops']-native['trials'][1]['flops'],
        production_initialization_flops=rows[0]['flops']-rows[1]['flops'],
        raw_error_measured=False, adjusted_score_measured=False,
        truth_accessed=False, full_five_network_fidelity_qualified=False)
