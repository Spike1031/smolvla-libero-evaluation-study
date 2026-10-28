# Public release preparation review

Review date: 27 October 2026. Repository: Spike1031/smolvla-libero-evaluation-study.

## Completed checks

- Inspected the main branch and all 15 reachable commits before this preparation change. Their trees contain 237 unique historical blobs: 235 current files and two earlier README versions. No additional historical file versions were hidden by deletion in this reachable history.
- Verified the pre-change packaged files against the included SHA-256 inventory. The owner's most recent README edit was preserved as the starting point for this preparation.
- Scanned 198 current text/archive payloads and both historical README versions for common GitHub/Hugging Face credential formats and private-key headers: no matches. This is a targeted scan, not a guarantee against every secret format.
- No email addresses or Windows/macOS home-directory patterns were found in the current text-file scan. Historical Colab/project paths remain as provenance; they are not usernames or portable execution paths.
- Read the upstream LeRobot Apache-2.0 and LIBERO MIT license texts and the three Hugging Face model/dataset card declarations. Retained upstream notices and documented the licensing boundary.
- Updated the project landing page to distinguish recorded-evidence reproduction from new training or simulation. No new inference, training or rollout was performed.

## Items requiring an owner decision or separate verification

1. All 15 existing commits expose the author's RWTH email address in Git metadata. Publishing this history exposes that address. The preparation does not rewrite history or claim it is anonymized.
2. All eight anonymous Drive requests returned HTTP 429. This is inconclusive, not evidence of either public access or a sign-in wall. The video index now labels playback as unverified and optional. There is no guaranteed public video demo.
3. Historical experiment source revisions and line-level derivation of the finishing scorer are not fully pinned. Do not describe this repository as a fully licensed upstream source redistribution or an end-to-end training reproduction.

## Release boundary

The report and packaged calculation evidence can be reviewed independently of video access. A public evidence repository can disclose the optional-demo limitation, but the owner must decide whether to expose the existing author-email history. Repository visibility and Drive sharing remain unchanged by this preparation. No force-push or history rewrite was performed.

## Owner decision recorded 28 October 2026

The owner explicitly accepts disclosure of the RWTH author email in the existing commit history. This privacy choice is resolved; history remains unchanged. It does not itself verify video playback or change repository visibility.

Six specific video files are listed in `videos/README.md` for owner-side sharing and anonymous playback verification. No permission change is claimed.

## Selected video permission verification 28 October 2026

Drive permission metadata confirms anyone-with-link reader access for all six files in `videos/README.md`. This resolves the permission-setting question for that selection, not for all eight archival clips. Anonymous playback remains a separate check; no end-to-end playback test is asserted.

## Unified catalogue update 28 October 2026

All 11 distinct video files now have verified anyone-with-link reader permissions. The owner confirms anonymous playback of the six recommended clips; supplementary playback is not independently confirmed. One CSV inventory replaces the overlapping six/eight catalogues. Historical review statements retain their dates.
