# Terminal covariance action D C D with 65 directions: covariance-only SSE 0.912/0.909 of parent, fails the 20% gates

2026-09-20 · experiment · rejected · measured

A final-read-only covariance action D C D with D = diag(exp(0.005*tanh([H14,1] q))), one 65-vector q, unchanged upstream state and final CP/star terms was given a privileged per-network least-squares oracle on the two reused development networks. Covariance alone improves the final sum of squared errors by 8.84 to 9.10% and the unique covariance improvement over the existing affine head is 10.2 to 10.5%, below the required 20% on both networks, so the fixed screen closed without a fit or candidate.

Recorded measurement:

    covariance/parent SSE 0.911650704 (net 35) and 0.908981202 (net 100); affine/parent 0.889154351/0.877527446; combined/parent 0.795778967/0.787664235; combined/affine 0.894984055/0.897594985; gates <=0.80 failed; derivative error 2.54e-9; three parent calls 1,164,520,978,510 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_covariance_information_v1/capture.py](code/terminal_covariance_information_v1/capture.py)
- [code/terminal_covariance_information_v1/information.py](code/terminal_covariance_information_v1/information.py)
- [code/terminal_covariance_information_v1/qualify.py](code/terminal_covariance_information_v1/qualify.py)
- [code/terminal_covariance_information_v1/unit.py](code/terminal_covariance_information_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/terminal_covariance_information_v1/tangent.py](code/terminal_covariance_information_v1/tangent.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)
- [NEGATIVE_RESULT.md#L17](NEGATIVE_RESULT.md#L17)
- [NEGATIVE_RESULT.md#L19](NEGATIVE_RESULT.md#L19)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-terminal-covariance-action-d-c-d-65-directions-covariance) · [Copy provenance](../../../PROVENANCE.json)

