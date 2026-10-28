# Project video index

This is the single video catalogue: **11 unique files** across two evaluation protocols. Start with **V01 → V02 → V03 → V04 → V05 → V06**. V01 is shared by both early comparisons, so it appears only once. V07–V11 are supplementary finishing diagnostics. These selected cases are explanations, not a success-rate sample.

## Access and provenance

All 11 files have verified Drive `anyone / reader` permission metadata as of 28 October 2026. The owner reports anonymous playback for the six recommended clips. Playback of the five supplementary clips has not been independently tested. A permission check is not a full streaming test. Observations for later clips are the owner’s historical descriptions, not fresh frame-level review.

The [CSV](video_index.csv) is the single machine-readable inventory. [Trace reconciliation](../tables/video_protocol_links.csv) is a derived eight-row diagnostic table for notebook 15, not a second video catalogue.

## Early comparisons

V01 versus V02 illustrates the recorded clean/masked difference. V01 versus V03 illustrates a first-round regression. These episodes stop at first benchmark goal or the 280-action cap; they cannot establish stable behavior after termination. Video metadata in the verified early evaluator uses 80 fps.

| ID | Model and case | Video | Recorded goal | Interpretation |
|---|---|---|---|---|
| V01 | Original / 001 clean | [Watch](https://drive.google.com/file/d/1uSGf66QPzuvQOiVKFiHBWyMUU5iBTDSr/view?usp=drivesdk) | True | Post-termination finishing unknown |
| V02 | Original / 001 occluded | [Watch](https://drive.google.com/file/d/1kYz9No2MNcJsLitBkEWXa29zlxJl72Sg/view?usp=drivesdk) | False | Post-termination finishing unknown |
| V03 | Round 1 / 001 clean | [Watch](https://drive.google.com/file/d/1HmLRBi5faQiXnzQ63XBIbwhCtGpCOIYH/view?usp=drivesdk) | False | Post-termination finishing unknown |

## Extended finishing diagnosis

These are separate notebook-15 rollouts with 440 policy actions and 20-fps video; execution continues after first goal attainment. Goal within 280, goal at action 440 and automatic complete finish are separate labels. Do not compare early and extended playback seconds directly, or treat identical state IDs as identical rollout files.

| ID | Model and case | Video | Goal ≤280 | Goal at end | Complete finish | Historical observation |
|---|---|---|---|---|---|---|
| V04 | Original / 001 clean | [Watch](https://drive.google.com/file/d/12HMXHck83fMeR5eneK7vghL7SQrd3atm/view?usp=drivesdk) | True | True | False | Placement followed by repeated bowl contact. |
| V05 | Finish-augmented / 001 clean | [Watch](https://drive.google.com/file/d/1tD9_Py9H1H78Q-yonYjX8L9PdP34n6aj/view?usp=drivesdk) | True | False | False | Regrasp by spreading fingers inside bowl; drop leaves bowl partly on table. |
| V06 | Finish-augmented / 020 clean | [Watch](https://drive.google.com/file/d/18_EAQMSGMyMfmWGHNBPbme_lmLF_JsYb/view?usp=drivesdk) | True | True | False | Single grasp/place; later jitter without further bowl contact. |
| V07 | Original / 008 clean | [Watch](https://drive.google.com/file/d/1Mpb6i2Q1yuyUfML51YPErsXqVG9233JJ/view?usp=drivesdk) | True | True | False | Delayed separation, then jitter and repeated contact; bowl remains on plate. |
| V08 | Original-demonstration training control / 008 clean | [Watch](https://drive.google.com/file/d/1jCF1y4d_BTY92Po08UjdgWwZxRRIdn4q/view?usp=drivesdk) | True | True | False | Delayed separation and jitter; contacts around seconds 9 and 12. |
| V09 | Finish-augmented / 008 clean | [Watch](https://drive.google.com/file/d/1dYWywdDAzQlkuXxNgUjj33pXCciaLU9T/view?usp=drivesdk) | False | False | False | Failed lift; bowl only tilts right. |
| V10 | Original / 020 clean | [Watch](https://drive.google.com/file/d/1fFDklRt-Z79fwMlR3LJykt74Rsy8u4nz/view?usp=drivesdk) | True | False | False | Regrasp after placement; bowl ends entirely on plate, slightly tilted. |
| V11 | Finish-augmented / 020 occluded | [Watch](https://drive.google.com/file/d/1AHt9_WYI2K9ftNmqUQRdum0lEncC3ry9/view?usp=drivesdk) | True | True | False | Similar single grasp/place and residual jitter without re-contact. |

## Selection rationale

- V04–V05: original and finish-augmented 001 clean, illustrating post-goal interaction and a later goal loss.
- V06: finish-augmented 020 clean, illustrating that no reported further contact does not establish stable finishing.
- V07–V09: 008 clean across three models, illustrating delayed release, jitter and a failed lift.
- V10–V11: supplementary 020 outcomes; manual physical placement and the geometric goal predicate must remain separate.

Read [protocol definitions](../docs/EVALUATION_PROTOCOLS.md) and [unresolved case interpretation](../docs/CASE_RECONCILIATION.md). “True” in the early table means benchmark goal attainment, not verified complete finishing.
