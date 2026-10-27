# Third-party materials and attribution

This evaluation study uses the upstream projects below. License declarations were inspected at their linked source locations on 27 October 2026; these checks do not pin the historical experimental dependencies to exact source revisions.

| Upstream artifact | Source | Declared license |
|---|---|---|
| LeRobot / SmolVLA implementation | https://github.com/huggingface/lerobot | Apache-2.0 |
| LIBERO simulation benchmark | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT |
| Evaluated policy checkpoint | https://huggingface.co/HuggingFaceVLA/smolvla_libero | Apache-2.0 (model card) |
| VLM checkpoint | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct | Apache-2.0 (redirected model card) |
| Original demonstration dataset | https://huggingface.co/datasets/HuggingFaceVLA/libero | Apache-2.0 (dataset card) |

Upstream license texts are retained in `licenses/LeRobot-Apache-2.0.txt` and `licenses/LIBERO-MIT.txt`, including their copyright notices. Model weights, original demonstration images, upstream library source trees and video binaries are not bundled. Listing a dependency does not mean its source code was authored in this project.

`analysis/reproduce_results.py` recalculates the study tables; `analysis/verify_later_traces.py` loads and verifies packaged evidence; `analysis/finish_scoring.py` preserves the historical experiment scoring logic. These are study analysis artifacts, not replacements for the upstream policy or simulator. Exact derivation of every historical scoring line from the original notebooks has not been independently established; no blanket license is asserted for the mixed repository. Inclusion of the upstream license texts does not relicense study artifacts or establish every third-party redistribution right.

Reports and derived records retain their provenance. Source and video links may require the original Drive access. Public video playback was not verified, and Drive sharing was not modified. See `docs/PUBLIC_RELEASE_REVIEW.md` for the review boundary.
