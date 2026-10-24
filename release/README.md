# Designated release packages

Start with [run and reproduction instructions](../estimator/REPRODUCE.md): install the pinned runtime, rebuild the exact estimator files, run a seeded network, then evaluate on the public dataset.

| Submission | Inference source | Original archive | Recorded grade |
|---|---|---|---|
| #337641 | [Python](../estimator/submission_337641.py) | [archive](../submissions/337641/submission.tar.gz) | [record](../submissions/337641/README.md) |
| #337619 | [Python](../estimator/submission_337619.py) | [archive](../submissions/337619/submission.tar.gz) | [record](../submissions/337619/README.md) |

`common/parents/` and `common/readout/` contain the parent sources, fitted coefficients and original export programs used by `tools/reproduce_estimator.py`. The designated files embed all inference weights.

The [original packaging README](ARCHIVE_README.md), [dependency manifest](DEPENDENCY_MANIFEST.md) and subdirectory READMEs are byte-preserved historical records. Use the reproduction guide for the tested installation and run commands; it explains the later SDK versions and the distinction between local execution and hosted timing.
