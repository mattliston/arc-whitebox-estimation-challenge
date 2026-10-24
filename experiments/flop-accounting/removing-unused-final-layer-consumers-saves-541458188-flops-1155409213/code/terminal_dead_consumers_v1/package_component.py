"""Inert exact-source local package; requires completed runtime qualification."""
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
SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"


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
        parser.print_help()
        return
    require(args.remote_result_sha256 is not None and len(args.remote_result_sha256) == 64,
            "A completed runtime receipt SHA is required")
    require(Path(sys.prefix).resolve() == ENV.resolve(), "Use OFFPY")
    require(importlib.metadata.version("whestbench") == "0.16.0", "Use Whest0.16")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Use nice10/no bytecode")
    source = HERE / "estimator.py"
    remote_path = HERE / "remote/result.json"
    require(digest(source) == SHA and digest(remote_path) == args.remote_result_sha256,
            "Source/runtime receipt differs")
    remote = load(remote_path)
    complete = load(HERE / "remote/completion.json")
    require(remote["passed"] and remote["hashes_unchanged"] and remote["candidate_sha256"] == SHA
            and remote["parent_all5_byte_exact"] and remote["strict_complete_cost_saving_all5"]
            and remote["prior_native14_pass_verified"] and complete["passed"]
            and complete["result_sha256"] == args.remote_result_sha256, "No qualified runtime result")
    bindings = {item["path"]: item["sha256"] for item in remote["input_bindings"].values()}
    bindings.update({str(source): SHA, str(remote_path): args.remote_result_sha256,
                     str(HERE / "remote/completion.json"): digest(HERE / "remote/completion.json"),
                     str(Path(__file__).resolve()): digest(__file__)})
    require(all(digest(path) == sha for path, sha in bindings.items()), "Qualification inputs changed")
    stage = HERE / "package_stage"
    archive = HERE / "terminal_dead_consumers_07b71a03_whest016.tar.gz"
    receipt = HERE / "package_result.json"
    require(not stage.exists() and not archive.exists() and not receipt.exists(), "Refusing existing package outputs")
    stage.mkdir()
    # Mechanical, isolated build artifact; exclude all scripts/receipts/weights.
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
            check = subprocess.run(command, cwd=stage, env=dict(os.environ), capture_output=True,
                                   text=True, timeout=120)
            result["commands"].append(dict(command=command, returncode=check.returncode,
                                            stdout=check.stdout, stderr=check.stderr))
            print(check.stdout, flush=True)
            require(check.returncode == 0, "Whest validation/package command failed")
            value = json.loads(check.stdout)
            if "ok" in value:
                require(value["ok"] is True, "Whest returned ok=false")
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            require(sorted(m.name for m in members) == ["estimator.py", "manifest.json"]
                    and all(m.isfile() for m in members), "Unexpected package members")
            contents = {m.name: tar.extractfile(m).read() for m in members}
            require(hashlib.sha256(contents["estimator.py"]).hexdigest() == SHA,
                    "Packaged estimator differs from qualified source")
            manifest = json.loads(contents["manifest.json"])
            require(isinstance(manifest, dict), "Invalid manifest")
        require(all(digest(path) == sha for path, sha in bindings.items()), "Inputs changed while packaging")
        require(digest(stage / "estimator.py") == SHA, "Staged estimator changed")
        result.update(passed=True, archive=str(archive), archive_sha256=digest(archive),
                      archive_bytes=archive.stat().st_size, hashes_unchanged=True,
                      members={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                               for name, data in contents.items()},
                      validation_scope="Generic Whest guard-path contract plus package checks; numerical evidence is separate native14/remote5.")
    except BaseException:
        result["error"] = traceback.format_exc()
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("input_sha256", "commands")}, sort_keys=True))
    require(result["passed"], "Local package qualification failed; preserved receipt")


if __name__ == "__main__":
    main()
