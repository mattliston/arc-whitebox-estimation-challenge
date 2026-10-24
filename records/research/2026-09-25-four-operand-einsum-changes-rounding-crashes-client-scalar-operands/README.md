# Four-operand einsum changes rounding and crashes on client scalar operands

2026-09-25T10:25Z · incident · failed · measured

The request-reduction candidate used a client scalar where the server expected an array, and the library evaluation order differed from raw NumPy. Client/server census became mandatory for new operation forms.

Recorded measurement:

    - 2026-09-25 03:25 lead: GROUP 5 route B DROPPED from S26 (engineer2): NOT bit-identical on the S26 pre-base (max |diff| 1.0-1.1e-5, MSE ratio 0.9966-0.9994), AND the numpy-free client census CRASHED in gnn_step. In client mode an integer index on a 1-D array returns a Python scalar, not a 0-d array, so the server einsum raised "'float' object has no attribute 'shape'"; in-process parity can't see this.
      FIRE-CHECK RULE (new): any new op form or operand-shape change needs the numpy-free CLIENT census (setup + >= 2 nets, all outputs finite) before it ships; in-process parity alone is insufficient. S26 = S24 + j3 + k-split v3 + treuse, and the chunk keys stay in the prebind list (engineer). The porter launches on the engineer's all-pass.
    2026-09-25 03:25 porter: group 5 OUT of S26 (engineer2: setup-first parity NOT bit-identical 4/4, max diff ~1e-5; client-mode crash in gnn_step: P['rms'][l] reaches the server as a bare float). S26 = S24 > j3cut > k-split v3 (02870e60) > key list (census of that file) > treuse LAST, via fire_chain_S26v3.sh -0.6 .../ksplit_fragment_v3.py 02870e60c012c413 none none none none. Launch waits only for the engineer's k-split all-pass (lead's conditional GO). Note for all in-process gates: they cannot see client-mode scalar/array differences; the client/server smoke in every chain is what catches them.
    - 2026-09-25 03:25 [engineer2] GROUP 5 (route B) FAILED ITS GATES, DROPPED FROM S26 (lead's rule; the porter uses k-split v3). Built by build_g5_B_v1.py (28701070) on the exact S26 pre-base a986fdc4 = S25_s24_j3.py + ksplit_fragment_v2.py: output 02d2126f, fragment ed5806df; porter's einsum gate PASS.
      (1) Parity (setup first, flopscope 0.12, one candidate per process, 35/100/37/39): NOT bit-identical on 4/4. max |diff| 1.0-1.1e-5, MSE ratio 0.9966-0.9994, bill -69M/net. The S23 four-function einsum prototype WAS bitwise, so the break is in something route B adds: _gn_dcov reading the pair state's FT/P16 buffers, or the new gnn/omitted/edge chains. Post-mortem off the fire path.
      (2) CLIENT CRASH (numpy-free client census): gnn_step's einsum got c['rms'] = P['rms'][l]. In client mode an INTEGER index on a 1-D RemoteArray returns a scalar VALUE, sent as a bare float, so the server einsum raised "'float' object has no attribute 'shape'". In process it is a 0-d array, so in-process parity cannot see this.
      LESSON: a new op form needs the client-server census before any claim; in-process parity is not enough.
    - 2026-09-25 03:26 [engineer2] GROUP 5 POST-MORTEM (off the fire path): flopscope's einsum(optimize=False) equals the left-deep chain bitwise only for <= 3 operands. With 4+ it runs pairwise along a contraction path (1-ulp differences in f64 and f32; raw numpy's 4-operand einsum is left to right, and that is what my earlier certificate tested). Route B's ONLY 4-operand chain is gnn_step's delta, which is also the client-crash site (RemoteScalar operand). Every other route-B einsum has <= 3 operands (certified through flopscope). Confirmation queued behind the fire path: route B minus the gnn_step edit (setup-first parity vs the saved base npz + client census). Nothing ships without the lead's GO.

## Programs

- [fire_chain_S26v3.sh](fire_chain_S26v3.sh)
- [build_g5_B_v1.py](build_g5_B_v1.py)
- [S25_s24_j3.py](S25_s24_j3.py)
- [../../../experiments/arithmetic-kernels/summing-split-contractions-leaf-space-removes-repeated-unwind-work/code/fable_engineer_v1/ksplit_fragment_v2.py](../../../experiments/arithmetic-kernels/summing-split-contractions-leaf-space-removes-repeated-unwind-work/code/fable_engineer_v1/ksplit_fragment_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L989](../../TEAM_LOG_2026-09-22_to_09-26.md#L989)
- [TEAM_LOG_2026-09-22_to_09-26.md#L990](../../TEAM_LOG_2026-09-22_to_09-26.md#L990)
- [TEAM_LOG_2026-09-22_to_09-26.md#L991](../../TEAM_LOG_2026-09-22_to_09-26.md#L991)
- [TEAM_LOG_2026-09-22_to_09-26.md#L992](../../TEAM_LOG_2026-09-22_to_09-26.md#L992)
- [TEAM_LOG_2026-09-22_to_09-26.md#L993](../../TEAM_LOG_2026-09-22_to_09-26.md#L993)
- [TEAM_LOG_2026-09-22_to_09-26.md#L994](../../TEAM_LOG_2026-09-22_to_09-26.md#L994)
- [TEAM_LOG_2026-09-22_to_09-26.md#L995](../../TEAM_LOG_2026-09-22_to_09-26.md#L995)
- [TEAM_LOG_2026-09-22_to_09-26.md#L996](../../TEAM_LOG_2026-09-22_to_09-26.md#L996)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-four-operand-einsum-changes-rounding-crashes-client-scalar-operands) · [Copy provenance](../../../PROVENANCE.json)

