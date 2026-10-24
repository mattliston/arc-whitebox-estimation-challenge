"""Same unfinished fixed trial: functional FNP assembly and stricter byte control."""

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "probe_terminal_summary64_v1.py"
V1_SHA = "964514c4e6287a22050adde7aba3c3dfc7b24d8432c7909d463539bf606a342a"
V2 = HERE / "probe_terminal_summary64_v2.py"
V2_SHA = "beffa6bb22280fb81b7209a29527b7a2aed21c8f6813f1f925147ca4e2b5c4a4"
CAPTURE = HERE / "terminal_summary_capture_v2/result.json"
CAPTURE_SHA = "5b30d63806115d1686e4e6dac9a556fda0815c580ca77140daeddb6ded15ddbd"


def main():
    raw = V1.read_text()
    if hashlib.sha256(raw.encode()).hexdigest() != V1_SHA:
        raise RuntimeError("Preserved v1 source changed")
    saved = CAPTURE.read_bytes()
    if hashlib.sha256(saved).hexdigest() != CAPTURE_SHA:
        raise RuntimeError("Preserved current-parent capture changed")
    old_assembly = '''                x = a["features"].copy()
                x[:, 22] = fnp.sqrt(row2).astype(fnp.float32)
                x[:, 23] = fnp.sqrt(col2).astype(fnp.float32)
                x[:, 27] = fnp.sqrt(post2).astype(fnp.float32)
                x[:, 28] = post1.astype(fnp.float32)
'''
    new_assembly = '''                replacements = {22: fnp.sqrt(row2).astype(fnp.float32),
                                23: fnp.sqrt(col2).astype(fnp.float32),
                                27: fnp.sqrt(post2).astype(fnp.float32),
                                28: post1.astype(fnp.float32)}
                x = fnp.stack([replacements[i] if i in replacements else a["features"][:, i]
                               for i in range(30)], axis=1)
'''
    old_control = '''        h.require(np.array_equal(np.asarray(original_delta), inputs["delta"])
                  and np.array_equal(np.asarray(original_output), inputs["output"][15]),
                  "Original current-P head/served-output control failed")
'''
    new_control = '''        od, oo = np.asarray(original_delta), np.asarray(original_output)
        h.require(od.shape == inputs["delta"].shape and od.dtype == inputs["delta"].dtype
                  and oo.shape == inputs["output"][15].shape and oo.dtype == inputs["output"].dtype
                  and od.tobytes() == inputs["delta"].tobytes()
                  and oo.tobytes() == inputs["output"][15].tobytes(),
                  "Original current-P head/served-output byte control failed")
'''
    if raw.count(old_assembly) != 1 or raw.count(old_control) != 1:
        raise RuntimeError("Ambiguous technical source splice")
    changed = raw.replace(old_assembly, new_assembly).replace(old_control, new_control)
    if changed.replace(new_assembly, old_assembly).replace(new_control, old_control) != raw:
        raise RuntimeError("Technical source inverse failed")
    ast.parse(changed)
    namespace = dict(__name__="terminal_summary64_v3_inert_body", __file__=str(Path(__file__).resolve()))
    exec(compile(changed, "<terminal-summary64-v3-functional-repair>", "exec"), namespace)
    seed_binding = json.loads(saved)["fixture_bindings"]["row_35"]
    namespace["PINS"].update({V1: V1_SHA, V2: V2_SHA,
        Path(seed_binding["path"]): seed_binding["sha256"],
        HERE / "TERMINAL_SUMMARY64_PROXY_PROTOCOL_V1.md": "ea0f643adfda1398f7e8a1561d2399481d81a6ee4428926161bad11d1b9b3018"})
    namespace["PROTOCOL"] = HERE / "TERMINAL_SUMMARY64_FUNCTIONAL_REPAIR_V3.md"
    namespace["OUT"] = HERE / "terminal_summary64_proxy_v3"
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
