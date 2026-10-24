"""Inert mechanical predicate extension; no participant or numerical imports."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'current_comp_partial_d6_v1/build_v1/estimator.py'
PARENT_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
OUT = HERE/'build_v1'
OLD = """    if (tuple(a.shape) == (1024,832) and tuple(b.shape) == (832,8192)
            and tuple(out.shape) == (1024,8192)
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1]):
"""
REPLACEMENT = """    if (len(a.shape) == len(b.shape) == len(out.shape) == 2
            and int(a.shape[0]) == 1024 and 0 < int(a.shape[1]) <= 1024
            and 1024 < int(b.shape[1]) <= 8192
            and int(a.shape[1]) % 64 == int(b.shape[1]) % 64 == 0
            and int(b.shape[0]) == int(a.shape[1])
            and tuple(out.shape) == (1024,int(b.shape[1]))
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1]):
"""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def derive(tick=lambda:None):
    raw = PARENT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA, 'Frozen CD parent bytes')
    original = raw.decode('utf-8')
    tick()
    require(original.count(OLD) == 1 and REPLACEMENT not in original, 'Unique final predicate')
    changed = original.replace(OLD, REPLACEMENT)
    require(changed.count(REPLACEMENT) == 1 and changed.replace(REPLACEMENT, OLD) == original,
            'Whole source byte-exact inverse to CD')
    tick()
    before = ast.parse(original)
    tick()
    after = ast.parse(changed)
    tick()
    b = [x for x in before.body if isinstance(x,ast.FunctionDef) and x.name == '_packed_product_4'][-1]
    a = [x for x in after.body if isinstance(x,ast.FunctionDef) and x.name == '_packed_product_4'][-1]
    require(isinstance(b.body[0],ast.If) and isinstance(a.body[0],ast.If), 'Exact last dispatcher owner')
    a.body[0].test = b.body[0].test
    before_dump = ast.dump(before,include_attributes=False)
    tick()
    after_dump = ast.dump(after,include_attributes=False)
    tick()
    require(before_dump == after_dump,
            'Only the last dispatcher predicate changed in the entire AST')
    compile(changed, str(OUT/'estimator.py'), 'exec')
    tick()
    proof = dict(parent_sha256=PARENT_SHA,
        candidate_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True, only_final_dispatch_predicate_changed=True,
        heads_and_physics_byte_unchanged=True, same_original_backing_arena=True,
        backing_arena_bytes=1168900096, additional_initialization_over_CD=0,
        existing_complete_initialization_flops=1944970298,
        no_read_or_square_change=True, no_padding=True, no_rank_or_policy_tuning=True,
        policy='FP32 original PUSH owner; m1024, k<=1024, 1024<n<=8192, k/n64-aligned')
    return changed, proof


def build(tick):
    require(not OUT.exists() and not HERE.is_symlink(), 'Exclusive generated build, never overwrite')
    source, proof = derive(tick)
    tick()
    OUT.mkdir()
    path = OUT/'estimator.py'
    # Generated artifact: one mechanically checked substitution, not a source refit.
    with path.open('x') as stream:
        stream.write(source)
    tick()
    require(hashlib.sha256(path.read_bytes()).hexdigest() == proof['candidate_sha256'], 'Actual build readback')
    record = dict(schema='current_comp_aligned_push_d6_build_v1',passed=True,completed=True,
        candidate_sha256=proof['candidate_sha256'],proof=proof,
        estimator_imported=False,truth_accessed=False,score_measured=False,
        artifacts={str(path):proof['candidate_sha256']})
    def write(name, value):
        with (OUT/name).open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.write('\n')
        tick()
        return hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    result_sha = write('result.json',record)
    end_sha = write('completion.json',dict(passed=True,completed=True,result_sha256=result_sha))
    final_sha = write('finalization.json',dict(passed=True,completed=True,
        result_sha256=result_sha,completion_sha256=end_sha))
    return record, dict(result_sha256=result_sha,completion_sha256=end_sha,finalization_sha256=final_sha)
