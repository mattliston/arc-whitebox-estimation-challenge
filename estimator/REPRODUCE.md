# Reproduce the designated estimators

Run these commands from the repository root. They rebuild both submission sources from the archived parent and fitted coefficients, then execute #337641 on one fresh width-1024, depth-16 network. Inference needs no training data or GPU. Use Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 _build/reproduce-venv
uv pip install --python _build/reproduce-venv/bin/python -r tools/requirements-reproduce.txt
_build/reproduce-venv/bin/python -B tools/reproduce_estimator.py
_build/reproduce-venv/bin/python -B tools/run_estimator.py \
  --submission 337641 --estimator _build/reproduced/337641/estimator.py
```

The reconstruction writes `_build/reproduced/{337619,337641}/estimator.py` and `verification.json`. Inference writes `_build/inference-337641/prediction.npy` and `result.json`. Output directories must be new; pass `--output <new-directory>` to repeat a command. To run #337619, change both occurrences of `337641` in the inference command. To execute the preserved artifact directly, omit `--estimator`.

The environment pins **NumPy 2.2.6, flopscope 0.12.3 and WhestBench 0.16.2**. The SDK pins follow the [October 17 organizer update](https://discourse.aicrowd.com/t/phase-2-update-flopscope-v0-12-3-deadline-extension-and-final-submission-nomination/18251). Earlier Phase 1 instructions and versions in the archived packaging manifests describe different environments.

## What the reconstruction verifies

[reproduce_estimator.py](../tools/reproduce_estimator.py) invokes the preserved [readout exporter](../release/common/readout/export_blocks_v1.py) with relocated parent paths. It uses the [fitted coefficient record](../release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json), strength `1.5` for #337619 and `1.35` for #337641, and the archived grouped-multiplication/gather tail. For #337641 it also reproduces the three retired helper-body replacements and the associated comment edit.

Input hashes are pinned before export. The verifier independently reapplies the coefficient matrix, checks the 22 decoded float32 tensors in the final `_GNN_LITERAL` against the preserved estimator, and requires the complete generated source to match byte for byte:

| Submission | Source SHA-256 |
|---|---|
| #337619 | `f66cb28baf0c018f25bb749d14ce9e9aedbb47b8f2acd408680cad169485d557` |
| #337641 | `5a4c15405c4462703aeead1bb6ccbc9b98ccf6ba2b5974ce23713303c87d594d` |

The exporter is executed from `release/common/`; estimator inputs are only parsed during reconstruction. The script writes a new output tree and never modifies the preserved inputs. A source-hash failure is an error even if the tensor values match; compression/runtime differences are recorded rather than silently accepted. The full mathematics and active override chain are described in [ANATOMY.md](ANATOMY.md).

## What the inference test verifies

[run_estimator.py](../tools/run_estimator.py) uses the official `sample_mlp` generator, a fixed seed (default `20261022`), `setup()`, and a metered `predict()` call. The network is **width 1024, depth 16** and the prediction budget is **2,199,023,255,552 FLOPs (`2**41`)**. Smaller shapes or budgets make these submitted estimators return zeros.

The test requires a finite, nonnegative, nonzero `float32` array of shape `(16, 1024)` and checks its first layer against the analytic Gaussian ReLU mean. It records source/weight/output hashes, package versions, FLOPs and timings. It uses one BLAS thread by default; `--threads`, `--seed`, and `--wall-time-limit` are configurable.

This is an in-process numerical test. Its default prediction cap is 600 seconds, and it does not enforce the hosted residual-time or setup-time limits. The hosted limits are 120 seconds per prediction, 0.4 seconds residual time and 5 seconds for setup. Output values and timing can vary with the numerical backend and machine; source identity does not imply cross-platform bitwise inference identity. See the [official CLI contract](https://github.com/AIcrowd/whestbench/blob/main/docs/reference/cli-reference.md).

## Validation recorded on 2026-10-22

Both full source hashes and all 22 tensor payloads matched on CPython 3.11.5 / macOS 26.5 arm64. The fresh-network test at seed `20261022` produced valid `(16, 1024)` float32 outputs for both sources:

| Submission | Metered FLOPs | Prediction wall | Residual wall |
|---|---:|---:|---:|
| #337619 | 234,802,513,819 | 43.34 s | 0.95 s |
| #337641 | 234,802,513,819 | 40.68 s | 0.88 s |

The first-layer Gaussian check for #337619 had maximum absolute error `7.11e-7`. These local residual times exceed the hosted 0.4-second gate; they establish successful numerical execution, not a hosted timing pass.

## Evaluate against public targets

The public Phase 2 dataset is pinned at `v2-phase2`, with `mini` and `full` splits. Its metadata supplies width 1024 and depth 16; `whest run` does not accept shape flags. This diagnostic command streams one `mini` network and computes accuracy against its published target; it relaxes local timing gates while retaining the Phase 2 FLOP budget:

```bash
HF_HOME="$PWD/_build/huggingface" PYTHONDONTWRITEBYTECODE=1 \
  _build/reproduce-venv/bin/whest run \
  --estimator estimator/submission_337641.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
  --split mini --streaming --n-mlps 1 \
  --flop-budget 2199023255552 \
  --runner local --max-threads 1 --setup-timeout 60 \
  --wall-time-limit 600 --no-residual-wall-time-limit --json \
  > _build/public-337641.json
```

The command was tested on the first `mini` network (`logan-fitzgerald`): zero failures, final-layer MSE `1.2468886367855703e-8`, and `247,437,803,321` metered FLOPs. This is a one-network check with relaxed timing gates.

Increase `--n-mlps` to evaluate more public networks. To rehearse the official timing gates, replace the last timing options with `--setup-timeout 5 --wall-time-limit 120 --residual-wall-time-limit 0.4` and remove `--no-residual-wall-time-limit`. Inspect the report's failure/exhaustion fields: a completed command can still contain predictions zeroed by a resource limit. Local public evaluation does not reproduce the archived hosted score or the private evaluation networks. Dataset provenance and split definitions are in the [official dataset card](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2).

## Retraining boundary

The commands above reconstruct the submitted artifacts from already fitted coefficients. Repeating the learning process additionally needs the parent-training data, 90-network on-policy captures, per-layer feature exports, truth targets, seeds, fitting dependencies and the original numerical environment. The retained research scripts contain VM paths and import assumptions that need adaptation. Several large assets remain unpublished or were not recovered; their status is in [DATA_INDEX.md](../data/DATA_INDEX.md). Inference and exact source reconstruction do not require those assets.
