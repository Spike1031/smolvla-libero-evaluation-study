# Project demonstration selection

Six clips selected to explain the experimental progression, not to estimate population success or show a new successful policy. All six files were checked on 28 October 2026: Drive permission metadata includes `type=anyone`, `role=reader`, `allowFileDiscovery=false` (anyone with the link can view). This verifies sharing permissions, not a completed anonymous playback test. Links identify historical Drive files; no sharing permission was changed.

| ID | Clip | Purpose |
|---|---|---|
| 1 | [Original 001 clean](https://drive.google.com/file/d/1uSGf66QPzuvQOiVKFiHBWyMUU5iBTDSr/view?usp=drivesdk) | Shared baseline for comparisons with 2 and 3; early goal attainment, unknown post-stop behavior. |
| 2 | [Original 001 occluded](https://drive.google.com/file/d/1kYz9No2MNcJsLitBkEWXa29zlxJl72Sg/view?usp=drivesdk) | Compare with 1: same initial state, central image mask, recorded goal failure. |
| 3 | [Round 1 001 clean](https://drive.google.com/file/d/1HmLRBi5faQiXnzQ63XBIbwhCtGpCOIYH/view?usp=drivesdk) | Compare with 1: same state and clean input, first adaptation regressed to recorded goal failure. |
| 4 | [Notebook 15 original 001 clean](https://drive.google.com/file/d/12HMXHck83fMeR5eneK7vghL7SQrd3atm/view?usp=drivesdk) | Extended original policy: recorded goal within 280 and at end, but complete finish false; recorded user observation includes post-placement contact. |
| 5 | [Notebook 15 with_finish 001 clean](https://drive.google.com/file/d/1tD9_Py9H1H78Q-yonYjX8L9PdP34n6aj/view?usp=drivesdk) | Extended adapted policy: goal reached and subsequently lost; user reports regrasp/drop. |
| 6 | [Notebook 15 with_finish 020 clean](https://drive.google.com/file/d/18_EAQMSGMyMfmWGHNBPbme_lmLF_JsYb/view?usp=drivesdk) | Extended adapted policy: user reports residual jitter without later bowl contact; automatic complete finish remains false. |

## How to interpret the comparisons

Clips 1-3 end on first benchmark success or the 280-step cap and use the early video encoding protocol. Clips 4-6 belong to a separate 440-action experiment with 20-fps video. Equal case IDs do not identify the same rollout across protocols. Do not align clips by playback seconds, mix early and extended success rates, or call clip 1 a verified stable finish.

The first three files were resolved on 28 October 2026 by traversing the exact attempt directories named in committed results to `videos/eval_episode_0.mp4`. This establishes path identity, not a fresh decoded-video inspection. Later observations remain the owner’s historical annotations. See `public_demo_selection.csv` for record paths.

The eight-clip archive remains in `README.md`; it is the later finishing diagnosis collection. This six-clip selection adds the early experimental context.
