# Selecting only padded birth slots by corrected tensor energy: held-out error worsens on both hosts

2026-10-21T03:04:54Z · experiment · rejected · measured

The selector preserved all above-floor births and only changed already-paid padding slots. Mac32 adjusted -.085019526% had positive held-out error, and Linux32 adjusted +.02903465% confirmed no usable gain.

Recorded measurement:

    BD-padding SCI74335f92 originalMac32 COMPLETE03:01:10Z: -0.085019526%adjusted/held +0.031654421%, SE0.284877,14/32lower,raw -0.114086525%,bill +0.029053563%. Wrapper resumed77740; no bank/stack. VM92528 still evaluating and will finish normally, scorer92823.
    Bill BDpadding SCI74335f92/RELe83342fe VM32PID92528 launched02:52:01, planc14da9c1/independentreview79606bdd. Allabovefloorbirths/currentcount preserved, only0–31padding selected byactualBDcorrectedtensorenergy. Accuracyhypothesis costsmaxmean+.032477%bill, not computesaving. Tiny14selectors/6actualBD/7tensororacles/importsPASS; Mac32 running. Newcompute routecutscontractedOLD CP-pair dimension whilekeeping384framecoordinates/alloutputrows/fullbank, distinctfromclosed384coordinatecompression. Existingcapturescovervlegonly: quartercutgrossceiling~.8583%bill (earlierbothleg1.455%notinitialscope), beforegathers/actualtariff/accuracy. ReuseTRAIN35/150physicalreads, completepairs/signedcolumns plusoneexact-rowmean controlvariate; no nativecandidateyet.
    Birth-consumer SCI37760350 CLOSED: Mac32 -0.0248472%/held -0.0030486%; Linux32 +0.0392067%/held +0.1174926%, bill0,16/32lower. BD-padding CLOSED: Linux +0.02903465%/held +0.24342797%, Mac -0.0850195%/held +0.0316544%. Edge8+innovation old stack CLOSED as prior record. None increases bank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/bd_padding_energy_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_v1.py](code/build_v1.py)
- [code/candidate_msg8_bd_padding_energy_sci_v1.py](code/candidate_msg8_bd_padding_energy_sci_v1.py)
- [code/price_current32_v1.py](code/price_current32_v1.py)
- [code/selector_v1.py](code/selector_v1.py)
- [code/tail_msg8_bd_padding_energy_v1.py](code/tail_msg8_bd_padding_energy_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6487](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6487)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6479](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6502](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6502)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-selecting-only-padded-birth-slots-corrected-tensor-energy-held) · [Copy provenance](../../../PROVENANCE.json)

