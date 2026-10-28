# Case and video reconciliation

The eight indexed videos all belong to notebook 15, not to the initial benchmark or the first two fine-tuning rounds. Their role is later finishing diagnosis. The index is not a representative estimate of the entire project's success rate.

## Checks completed on 28 October 2026

Each of eight index rows was matched uniquely by model, state and condition to a packaged 440-step trace. Both goal-within-280 and goal-at-end values agree with the trace scores. See `tables/video_protocol_links.csv` for historical attempt paths, links and every goal true-to-false transition. This is a record linkage check; video binaries were not re-downloaded, frame-synchronized or independently visually reviewed in this audit.

| Case | Goal within 280 | Goal at 440 | Explanation |
|---|---|---|---|
| Original, 020 clean | true | false | The trace contains several true-to-false transitions. The user's observation that the bowl ends on the plate does not prove the geometric predicate is true. Why this exact final geometry fails remains unverified. |
| Finish-augmented, 001 clean | true | false | The goal was reached and later lost; this can happen because the extended protocol does not stop at first goal. The user reports regrasp and drop. Temporal causal linkage to the individual contacts still requires synchronized video/trace review. |
| Original, 008 clean | true | true | The goal turns false at step 267, then is regained. A true final goal does not imply continuously stable finishing. |
| Original-only, 008 clean | true | true | Goal losses occur at steps 236 and 245; the goal is true at the end. |

All eight have automatic complete finish false. Do not infer that every one fails for the same physical reason.

## Open historical questions

- Round-1 019 occluded was manually described as placed and released but automatically failed; round-2 008 occluded had similar visible post-placement contact and was automatically successful. The early records do not contain synchronized per-step predicate/geometry traces. These remain visually similar cases with unresolved label interpretation, not proof of an evaluator bug.
- Early approximately one-second successes have unknown post-termination behavior. Asking the owner to rewatch cannot recover unrecorded frames.
- Missing binary identity checks must not be replaced with assumed matches based on state number alone.

## Presentation plan

Keep these eight clips as the later-diagnosis collection. A project-wide demo should separately include an early clean/masked pair, a verified original/adapted repair or regression pair, and an extended finishing pair. The early pairs are not yet selected and verified. Do not publish an arbitrary clip as a full-finish success. No Drive sharing change is required to complete this record audit.
