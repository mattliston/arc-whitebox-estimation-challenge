"""Additive seed-fixture seal before the single, still-unexecuted v1 probe."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "probe_terminal_summary64_v1.py"
V1_SHA = "964514c4e6287a22050adde7aba3c3dfc7b24d8432c7909d463539bf606a342a"
CAPTURE = HERE / "terminal_summary_capture_v2/result.json"
CAPTURE_SHA = "5b30d63806115d1686e4e6dac9a556fda0815c580ca77140daeddb6ded15ddbd"


def main():
    raw = V1.read_bytes()
    if hashlib.sha256(raw).hexdigest() != V1_SHA:
        raise RuntimeError("Preserved v1 source changed")
    saved = CAPTURE.read_bytes()
    if hashlib.sha256(saved).hexdigest() != CAPTURE_SHA:
        raise RuntimeError("Preserved current-parent capture changed")
    namespace = dict(__name__="terminal_summary64_v2_inert_body", __file__=str(Path(__file__).resolve()))
    exec(compile(raw, "<terminal-summary64-v2-seed-seal>", "exec"), namespace)
    seed_binding = json.loads(saved)["fixture_bindings"]["row_35"]
    namespace["PINS"].update({V1: V1_SHA, Path(seed_binding["path"]): seed_binding["sha256"]})
    namespace["OUT"] = HERE / "terminal_summary64_proxy_v2"
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
