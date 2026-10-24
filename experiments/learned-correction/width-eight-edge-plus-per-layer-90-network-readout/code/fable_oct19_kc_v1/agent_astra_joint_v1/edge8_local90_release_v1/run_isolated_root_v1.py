"""Root-only quiet/release isolation for the exact authorized edge8+local90 source.
Uses frozen command arrays and original guards. Does not submit.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST_v1.json"
MANIFEST_SHA = "125ebd4e66c7f1339a6fe8b15a4b5f7260b9d58624273b2b896df6c627de23cb"
HELPER = HERE.parent / "qualify_runtime_companion_isolated_v3.py"
HELPER_SHA = "cc6f457db1297877a4b2c09b0d95489d5cd40d26e749ca5151b098d4f8911b84"
ROOT_PID = 77740
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
stamp = lambda: subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("quiet", "release"))
    args = parser.parse_args()
    assert sha(MANIFEST) == MANIFEST_SHA
    assert sha(HELPER) == HELPER_SHA
    plan = json.loads(MANIFEST.read_text())
    for path, digest in plan["source_pins"].items():
        assert sha(path) == digest, ("source pin changed", path)
    spec = importlib.util.spec_from_file_location("frozen_isolation_helpers", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    out = Path(plan["quiet_output"])
    record = HERE / ("ISOLATION_" + args.phase + "_v1.json")
    assert not record.exists(), record
    if args.phase == "quiet":
        assert not out.exists(), out
        out.mkdir()
        (out / "bank").mkdir()
        (out / "candidate").mkdir()
        jobs = [("bank", out/"bank/run.log", 140),
                ("candidate", out/"candidate/run.log", 140),
                ("analyze", out/"analyze.log", 30)]
    else:
        analysis = out/"QUIET_PAIR_ANALYSIS_v1.json"
        assert json.loads(analysis.read_text())["status"] == "QUIET_THREE_RANK_OWN_SCIENCE_BYTES_AND_TARIFF_PASS"
        review = json.loads((HERE/"ROOT_RUNTIME_REVIEW_v1.json").read_text())
        assert review["status"] == "ACCEPT_RUNTIME_FOR_ORIGINAL_RELEASE_GATES"
        assert review["analysis_sha256"] == sha(analysis)
        assert review["candidate_release_sha256"] == plan["sources"]["release_sha256"]
        assert not Path(plan["qualification"]).exists()
        assert not (helper.O/"release/runs"/plan["release_tag"]).exists()
        jobs = [("release_min", out/"release_min.log", 180)]
    initial = helper.processes()
    assert ROOT_PID in initial and "run_queue_astra_v6.sh" in initial[ROOT_PID]["command"]
    helper.deny_external_heavy(initial, helper.descendants(initial, ROOT_PID))
    rec = dict(start_utc=stamp(), phase=args.phase, status="ISOLATING",
               manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST),
               runner_sha256=sha(__file__), helper_sha256=sha(HELPER),
               reference_submission=337422, candidate_release_sha256=plan["sources"]["release_sha256"],
               no_auto_fire=True, no_capacity_guard_queue_source_mutation=True, runs=[])
    paused = {}
    active = None
    began = time.monotonic()
    exitcode = 1
    def save():
        record.write_text(json.dumps(rec, indent=2)+"\n")
    def interrupt(signum, frame):
        raise KeyboardInterrupt("signal %s" % signum)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    def run(command, log, timeout):
        nonlocal active
        with log.open("x") as stream:
            active = subprocess.Popen(command, cwd=helper.O, stdout=stream, stderr=subprocess.STDOUT)
            try:
                return active.wait(timeout=timeout)
            finally:
                if active.poll() is None:
                    active.send_signal(signal.SIGINT)
                    active.wait(timeout=30)
                active = None
    try:
        os.kill(ROOT_PID, signal.SIGSTOP)
        paused[ROOT_PID] = initial[ROOT_PID]
        table = helper.processes()
        judges = [pid for pid, r in table.items() if r["ppid"] == ROOT_PID
                  and "run_current32_" in r["command"] and "_fable_v1.py " in r["command"]]
        for pid in judges:
            os.kill(pid, signal.SIGSTOP)
            paused[pid] = table[pid]
        rec["paused"] = paused
        save()
        print(stamp(), "DRAINING", len(judges), flush=True)
        deadline = time.monotonic()+210
        while True:
            table = helper.processes()
            tree = helper.descendants(table, ROOT_PID)
            helper.deny_external_heavy(table, tree)
            remaining = {pid for pid in tree-set(paused) if not table[pid]["state"].startswith("Z")}
            if not remaining:
                break
            assert time.monotonic() < deadline, remaining
            time.sleep(2)
        rec.update(drained_utc=stamp(), free_pct_before=helper.pressure())
        save()
        assert rec["free_pct_before"] >= 60, rec["free_pct_before"]
        for label, log, timeout in jobs:
            for path, digest in plan["source_pins"].items():
                assert sha(path) == digest, ("pin changed", path)
            table = helper.processes()
            helper.deny_external_heavy(table, helper.descendants(table, ROOT_PID))
            command = plan["commands"][label]
            print(stamp(), "START", label, flush=True)
            rc = run(command, log, timeout)
            rec["runs"].append(dict(label=label, command=command, returncode=rc, log=str(log), end_utc=stamp()))
            save()
            assert rc == 0, (label, rc, str(log))
            print(stamp(), "DONE", label, flush=True)
        rec["status"] = "QUIET_PHASE_PASS_REVIEW_REQUIRED" if args.phase == "quiet" else "ORIGINAL_RELEASE_GATES_PASS_NO_FIRE"
        exitcode = 0
    except BaseException as error:
        rec.update(status="FAILED", error=repr(error))
        raise
    finally:
        table = helper.processes()
        resumed = []
        for pid, before in reversed(list(paused.items())):
            now = table.get(pid)
            if now and now["started"] == before["started"] and now["command"] == before["command"]:
                try:
                    os.kill(pid, signal.SIGCONT)
                    resumed.append(pid)
                except ProcessLookupError:
                    pass
        rec.update(end_utc=stamp(), pause_elapsed_s=time.monotonic()-began,
                   resumed=resumed, exitcode=exitcode)
        try:
            rec["free_pct_after"] = helper.pressure()
        except Exception as error:
            rec["free_pct_after_error"] = repr(error)
        save()
        print(rec["end_utc"], "RESEARCH_RESUMED", resumed, "exitcode", exitcode, flush=True)
    return exitcode

if __name__ == "__main__":
    raise SystemExit(main())

