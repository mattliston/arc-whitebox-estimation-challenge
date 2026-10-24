"""Inert exact-source local package, conditional on native14 and remote5 pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

HERE = Path(__file__).resolve().parent
ENV = HERE.parent.parent / "data_gate/.venv"
SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remote-result-sha256")
    args = parser.parse_args()
    if not args.execute:
        parser.print_help(); return 0
    require(args.remote_result_sha256 is not None and len(args.remote_result_sha256) == 64,
            "Completed runtime receipt SHA required")
    require(Path(sys.prefix).resolve() == ENV.resolve() and importlib.metadata.version("whestbench") == "0.16.0",
            "Use OFFPY Whest0.16")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and os.getpriority(os.PRIO_PROCESS, 0) >= 10,
            "Use nice10/no bytecode")
    source, remote_path = HERE / "estimator.py", HERE / "remote/result.json"
    require(digest(source) == SHA and digest(remote_path) == args.remote_result_sha256, "Source/runtime pin differs")
    remote, complete = load(remote_path), load(HERE / "remote/completion.json")
    require(remote["passed"] and remote["hashes_unchanged"] and remote["candidate_sha256"] == SHA
            and remote["runtime_gates_passed"] and remote["strict_complete_cost_saving_all5"]
            and remote["backend_subset_adjusted_gate_passed"] and remote["prior_native14_pass_verified"]
            and complete["passed"] and complete["result_sha256"] == args.remote_result_sha256,
            "No qualified runtime and actual backend adjusted pass")
    bindings = {item["path"]: item["sha256"] for item in remote["input_bindings"].values()}
    bindings.update({item["path"]: item["sha256"] for item in remote["truth_bindings"].values()})
    bindings.update({str(source): SHA, str(remote_path): args.remote_result_sha256,
                     str(HERE / "remote/completion.json"): digest(HERE / "remote/completion.json"),
                     str(Path(__file__).resolve()): digest(__file__)})
    require(all(digest(path) == sha for path, sha in bindings.items()), "Qualification input changed")
    stage = HERE / "package_stage"
    archive = HERE / "resident_gram512_0c21db7a_whest016.tar.gz"
    receipt = HERE / "package_result.json"
    require(not stage.exists() and not archive.exists() and not receipt.exists(), "Refusing existing outputs")
    stage.mkdir()
    # Mechanical isolated build artifact; no source rewriting or extra members.
    with (stage / "estimator.py").open("xb") as stream:
        stream.write(source.read_bytes())
    result = dict(passed=False, candidate_sha256=SHA, remote_result_sha256=args.remote_result_sha256,
                  input_sha256=bindings, commands=[], uploaded=False, official_score=False)
    try:
        cli = str(ENV / "bin/whest")
        for command in (
            [cli, "validate", "--estimator", str(stage / "estimator.py"), "--json"],
            [cli, "package", "--estimator", str(stage / "estimator.py"), "--output", str(archive), "--yes", "--json"],
            [cli, "validate-package", str(archive), "--json"],
        ):
            run = subprocess.run(command, cwd=stage, env=dict(os.environ), capture_output=True,
                                 text=True, timeout=120)
            result["commands"].append(dict(command=command, returncode=run.returncode,
                                            stdout=run.stdout, stderr=run.stderr))
            print(run.stdout, flush=True)
            require(run.returncode == 0, "Whest package/validation failed")
            value = json.loads(run.stdout)
            require(value.get("ok") is True, "Whest did not return ok=true")
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            require(sorted(m.name for m in members) == ["estimator.py", "manifest.json"]
                    and all(m.isfile() for m in members), "Unexpected package members")
            contents = {m.name: tar.extractfile(m).read() for m in members}
            require(hashlib.sha256(contents["estimator.py"]).hexdigest() == SHA, "Packaged source differs")
            require(isinstance(json.loads(contents["manifest.json"]), dict), "Invalid manifest")
        require(all(digest(path) == sha for path, sha in bindings.items())
                and digest(stage / "estimator.py") == SHA, "Inputs changed during packaging")
        result.update(passed=True, archive=str(archive), archive_sha256=digest(archive),
                      archive_bytes=archive.stat().st_size, hashes_unchanged=True,
                      members={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                               for name, data in contents.items()},
                      validation_scope="Whest generic guard-path and package checks; native14 and actual remote5 evidence remain separate.")
    except BaseException:
        result["error"] = traceback.format_exc()
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256", "commands")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
