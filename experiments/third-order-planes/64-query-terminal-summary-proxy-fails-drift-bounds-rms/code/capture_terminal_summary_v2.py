"""Pre-execution, observation-only nesting fix; preserves the unexecuted v1."""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "capture_terminal_summary_v1.py"
V1_SHA = "be38599c6dd21654ccd7b108675c7090e1a4532a6de4ffa0d1a80e43bffed054"
PROTOCOL_V1_SHA = "9dd305c930270792fb8d98f6c181cef541787bdfeb1b4267aa36e9f3d36a3b59"


def main():
    raw = V1.read_text()
    if hashlib.sha256(raw.encode()).hexdigest() != V1_SHA:
        raise RuntimeError("Frozen v1 capture source changed")
    old_state = 'state = dict(layer=-1, final_captured=False)'
    new_state = 'state = dict(layer=-1, final_captured=False, mm_depth=0)'
    old_observer = '''        def observe_mm(left, right, *args, **kwargs):
            before = int(ctx_ref[0].flops_used)
            value = original_mm(left, right, *args, **kwargs)
            if state["layer"] == 15:
                kernels.append(dict(left_shape=list(left.shape), right_shape=list(right.shape),
                                    flops=int(ctx_ref[0].flops_used)-before,
                                    after_input_capture=state["final_captured"]))
            return value
'''
    new_observer = '''        def observe_mm(left, right, *args, **kwargs):
            depth = state["mm_depth"]
            state["mm_depth"] = depth + 1
            before = int(ctx_ref[0].flops_used)
            try:
                value = original_mm(left, right, *args, **kwargs)
            finally:
                state["mm_depth"] = depth
            if state["layer"] == 15 and depth == 0:
                kernels.append(dict(left_shape=list(left.shape), right_shape=list(right.shape),
                                    flops=int(ctx_ref[0].flops_used)-before,
                                    after_input_capture=state["final_captured"],
                                    inclusive_recursive_bill=True))
            return value
'''
    if raw.count(old_state) != 1 or raw.count(old_observer) != 1:
        raise RuntimeError("Ambiguous technical observation splice")
    changed = raw.replace(old_state, new_state).replace(old_observer, new_observer)
    if changed.replace(new_state, old_state).replace(new_observer, old_observer) != raw:
        raise RuntimeError("Technical source inverse failed")
    ast.parse(changed)
    namespace = dict(__name__="terminal_summary_v2_inert_body", __file__=str(Path(__file__).resolve()))
    exec(compile(changed, "<terminal-capture-v2-observation-only>", "exec"), namespace)
    namespace["OUT"] = HERE / "terminal_summary_capture_v2"
    namespace["PROTOCOL"] = "TERMINAL_SUMMARY_CAPTURE_NESTING_V2.md"
    namespace["PINS"].update({
        V1.name: V1_SHA,
        "TERMINAL_SUMMARY_CAPTURE_PROTOCOL_V1.md": PROTOCOL_V1_SHA,
    })
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
