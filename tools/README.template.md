# ARC White-Box Estimation Challenge 2026: Phase 2

Estimating mean activations of width-1024, depth-16 ReLU networks under standard-normal input. This release contains the submitted estimators, experiment programs, measured results and research dependencies.

| Designated estimator | Archived public score | Source |
|---|---:|---|
| #337641 | {score337641} | [Python](estimator/submission_337641.py) · [grade](submissions/337641/README.md) |
| #337619 | {score337619} | [Python](estimator/submission_337619.py) · [grade](submissions/337619/README.md) |

Both embed their inference coefficients. They combine moment and cumulant propagation, learned corrections and an on-policy fitted readout. The [Phase 2 paper (download PDF)](https://github.com/mattliston/arc-whitebox-estimation-challenge/raw/refs/heads/main/WRITEUP.pdf) describes #337641, the research decisions, measured improvements and open mechanistic questions. [LaTeX source](WRITEUP.tex) · [Web-readable text](WRITEUP.md). The [source-linked anatomy](estimator/ANATOMY.md) covers both designated implementations.

For clickable citations and source links, open the downloaded PDF in a PDF reader. [GitHub's PDF preview ignores embedded links](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files#rendering-pdf-documents).

## Run and reproduce

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
uv venv --python 3.11 _build/reproduce-venv
uv pip install --python _build/reproduce-venv/bin/python -r tools/requirements-reproduce.txt
_build/reproduce-venv/bin/python -B tools/reproduce_estimator.py
_build/reproduce-venv/bin/python -B tools/run_estimator.py --submission 337641 --estimator _build/reproduced/337641/estimator.py
```

The first Python command reconstructs both submitted source files and checks their exact hashes. The second runs the rebuilt #337641 on a seeded width-1024, depth-16 network and writes predictions and metered-computation diagnostics under `_build/inference-337641/`. It is a numerical smoke test, not a leaderboard score or a hosted timing pass.

For **both designated estimators (#337619 and #337641)**, the [reproduction guide](estimator/REPRODUCE.md) covers exact source reconstruction, expected outputs, inference, public-target evaluation and retraining inputs. The data needed to refit the learned model is not all distributed in Git; [availability is explicit](data/DATA_INDEX.md).

## Explore the work

- **[Interactive research graph](https://mattliston.github.io/arc-whitebox-estimation-challenge/):** choose **Start here**. Read the problem, then follow six approaches into the individual attempts, their results and the questions explored beneath them. **Estimator development** follows the recorded changes from analytic moments through learned corrections, the rebuilt core and the final readouts. The complete map stays visible, with unsuccessful versions beside the continuing work. **Open work folder** takes you to the related code and results.
- **[Experiments](experiments/README.md):** actual Python programs and recovered local dependencies, grouped by mathematical component, with results and concise READMEs.
- **[Submissions](submissions/SUBMISSIONS.md):** all {submissions} submitted sources, grade records and {archives} recovered archives.
- **[Research records](records/README.md):** findings, decisions, lessons and explicitly identified result-only records. [Glossary](graph/GLOSSARY.md).
- **[Release package](release/README.md):** the two designated archives, reconstruction inputs and packaging records.

![Archived public submission scores](graph/score.svg)

Lower scores are better. The official score averages each network's final-layer MSE multiplied by `max(0.1, metered FLOPs / 2^41)`; failed evaluations are marked separately. These are archived public results, not private re-evaluation results. [Rules](records/OFFICIAL_RULES_VERBATIM_PASTE_20261021.md#L163).

## Verify the release

```sh
python3 -I graph/validate.py --strict --checksums
node tools/check_viewer.cjs
```

[Provenance](PROVENANCE.json) records the origin and SHA-256 of every copied artifact. Submitted estimator sources and experiment programs retain their original bytes. The frozen research repository was read only. [Validation coverage and remaining limitations](graph/release_audit.json). The viewer also works offline: open `graph/index.html` in a browser. Offline file links default to GitHub and can be switched to local files.

To rebuild the paper, install [Tectonic](https://tectonic-typesetting.github.io/) and run `tectonic WRITEUP.tex` from the repository root. The LaTeX source contains its bibliography. To republish the graph and PDF, run `python3 -I tools/publish_pages.py`; this updates only the `gh-pages` branch.

LLM coding agents assisted the research, code and documentation under human direction. [Archived disclosure](release/common/writeup/WRITEUP_DISCLOSURE_DRAFT_v1.md). [MIT license](LICENSE).
