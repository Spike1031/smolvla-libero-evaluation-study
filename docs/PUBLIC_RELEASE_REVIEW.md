# Public release review

Current review: **28 October 2026**. Repository: `Spike1031/smolvla-libero-evaluation-study`.

## Current decision

The owner authorized publication after final checks. The repository is suitable for public review as a single-task evaluation and failure-analysis study, with the limitations below. Live GitHub visibility is authoritative; this document does not itself change access.

## Final checks

- Compared all 264 pre-release repository blobs against local copies. One video CSV differed only in line-ending serialization; retained the exact remote bytes and rebuilt the SHA-256 inventory. No experimental result was changed.
- Recalculated all 160 early episode records: passed. Original 28/40, round 1 23/40, round 2 28/40; paired repairs/regressions retained.
- Verified 38 later trace records, representing 34 distinct rollouts and four reused baselines: passed.
- Verified one video catalogue with 11 unique files: three early and eight extended clips. Record identities and stored labels match; this is not fresh video decoding or a frame-level review.
- All 11 video files have verified anyone-with-link reader permission metadata. The owner confirms anonymous playback of the six recommended clips; supplementary streaming has not been independently confirmed.
- All local Markdown file destinations resolve. All 13 packaged Python sources parse. The final hash inventory is checked after editing.
- A targeted current-text credential scan found no common GitHub/Hugging Face/AWS access-key patterns or private-key headers. The previous historical review inspected 15 reachable commits and 237 unique historical blobs; these covered the 235 then-current files and two older README versions. This is a targeted scan, not a guarantee against every secret format.
- The owner accepts disclosure of the RWTH author email in existing Git history. No history rewrite was performed.
- Retained LeRobot Apache-2.0 and LIBERO MIT license notices and upstream model/dataset declarations. No blanket license is asserted for the mixed repository.

## Result and source boundaries

Early benchmark goal attainment does not establish release, withdrawal or stability after termination. Later complete-finish results use a defined conservative automatic rule, complemented by historical owner observations. No tested adaptation produced a verified better autonomous policy. Development cases are not an independent generalization test.

Six evaluation runners match recorded historical hashes. Training reports and protocols are packaged, but historical training-source byte identity and every historical external dependency are not fully pinned. The evidence calculations are reproducible on CPU; a fresh end-to-end training/simulation environment, model weights and demonstration images are not bundled. No GPU inference, training or simulation was performed for release.

The report PDF remains the dated 26 October synthesis. Its delivery-status passages are historical; current source, protocol, access and case-reconciliation notes supplement it. Original records and dated audits are preserved as evidence rather than rewritten to erase uncertainty.

## Earlier access checks resolved

The 27 October anonymous Drive HTTP 429 responses were inconclusive. They did not establish a permission failure. Subsequent permission metadata and the owner's six-clip playback check supersede that access uncertainty for the recommended selection. The previously overlapping six/eight clip lists have been replaced by one 11-file catalogue.
