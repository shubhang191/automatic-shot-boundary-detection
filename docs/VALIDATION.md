# Public packaging validation — 2026-11-03

This records software validation, not new benchmark accuracy. Windows 11 x64, Python 3.13.5; unchanged CPU dependency pins. Staging fresh venv: **92 passed, 0 failed, 0 skipped (164.89 s)**. Separate Git clean clone, independently created venv: **92 passed, 0 failed, 0 skipped (263.11 s)**. No global/system-site dependency reuse. pip check and environment/model/native validation passed in each.

All eleven supported configurations ran on the licensed six-second/150-frame Sintel sample, including actual neural inference, Python fitted models and native C++ filtering. Scientific fields (shot geometry/times, candidates, scores/decisions, native labels and model identity) match the known-good faculty sample outputs across all eleven. The clean-clone results also match staging. Every result ZIP passes CRC and has matching CSV/JSON shots, a timeline and one JPEG keyframe per shot. UI was launched through run.bat, visually inspected and used for actual upload/inference; README screenshots show this genuine demonstration. The existing synthetic/numeric fixtures also pass unchanged. Unsupported TinyCut Python is covered by the retained regression suite.

Both environments fetched official pinned Omni models and the pinned FFmpeg archive independently. AutoShot used the explicit checksum-matched user-supplied file through the documented downloads/import step. **The authors' Baidu login/download acquisition flow was NOT TESTED**; access to that service is a manual prerequisite. No permission to rehost that checkpoint is assumed. No all-200 live neural rerun, independent source-disjoint study, CUDA test, Linux/macOS test or separate clean OS VM test is claimed.

Security review covered the exact committed source tree before public publication: no credentials/secrets, .env, private notebook/research/media archives, runtime environments/caches/logs, pretrained .pth files, FFmpeg binaries or personal absolute paths. Bundled sample is attributed CC BY 3.0 Sintel; synthetic/numeric fixtures contain no SHOT images. Original backend Python, original tests, model/decoder/fitted-bundle bytes, C++ source/binaries, dependency pins and startup scripts remain unchanged. Public packaging adds prepare_models.py, curated documentation/attribution, Git ignores and byte-preserving .gitattributes; an unused machine-path validation manifest and private historical reports were omitted.

Original faculty source preservation: **431 distributable files, zero added/missing/modified** in the recorded inventory (runtime caches/environments and an existing nested duplicate excluded from the inventory; none was used as a writable test location). No original Git state was touched.

The only application warning was the pre-existing upstream AutoShot/PyTorch indexing deprecation. One external export-audit helper initially expected PNG thumbnails, then was corrected to the actual JPEG format; no application or existing test was changed to force success.

See README for supported configurations, MODELS.md for legal acquisition and exact identities, and RESEARCH_RESULTS.md for the separate historical figures. Full machine-local logs are intentionally not published because they contain development paths.

## Published-repository check

After publication, a new anonymous Git clone from the public GitHub URL independently obtained the official Omni models and FFmpeg again, imported the explicit user-supplied AutoShot file, passed environment/model verification and Gradio startup, and ran **all eleven configurations** successfully. Scientific outputs match the pre-publication validated staging run, and all result ZIPs pass CRC with matching JSON shots. This final smoke reused the separately fresh-installed clean-clone dependency environment; it does not claim a third fresh package installation. Public accessibility, rendered README, both actual screenshot images and the published file tree were checked.

## Final Windows source synchronization — 2026-11-04

The finishing source changes are deployment diagnostics/build/packaging fixes only: fresh deterministic native output discovery (including GNU-style DLL names, never import libraries), Windows x64 checks and naming, explicit ZIP safety exceptions under optimized Python, and a deterministic DLL/CLI installation operation. Existing precompiled binaries, model/checkpoint bytes, original tests, inference modules and dependency pins are unchanged. The curated public allowlist is preserved; it declares required locally provisioned model/tool paths without publishing those files. Private faculty media and audit artifacts are excluded.

Newly executed on the reviewed Git checkout using the prior audit's isolated Python 3.13.5 x64 environment (no new installation or second-machine claim):

- Focused deployment tests: **21 passed**, 0 failed/errors/skipped.
- Full regression suite: **113 passed** (92 original + 21 added), 0 failed/errors/skipped.
- Installation diagnostics: **PASS**, including unchanged checkpoint hashes and deterministic native DLL/CLI agreement with compiler directories absent from PATH.
- Optimized-Python negative safety checks: **22 rejected invalid cases**, including unsafe/reserved paths, case collisions, duplicate names/manifest keys, missing files, manifest tampering, symlinks, CRC corruption and output overwrite attempts. These checks used disposable fixtures; no faculty ZIP was generated or modified.
- Licensed Sintel real inference: **all 11 supported configurations PASS**. Boundaries/intervals, timestamps, raw proposals, KEEP/DROP probabilities/decisions, native labels, rescues/snaps and model identities match the verified faculty baseline. Deterministic CSV/image exports are byte-identical and scientific JSON fields match; runtime timing fields are excluded. Equal shot counts alone are not used as parity evidence.

The tested checkout was provisioned with checksum-matched local model/tool files that remain ignored and unpublished. This synchronization did not repeat external download acquisition, clean dependency installation or browser interaction. Earlier final-release evidence covers fresh extraction/setup, repeated setup, invalid-DLL rejection, paths with spaces, Gradio upload/export workflows and all 11 modes; its source identities were reconciled read-only against the unchanged archived/tested application. A prior Ninja + LLVM-MinGW x64 source build passed; actual GNU GCC x64 and MSVC builds were not performed. The shipped native binaries were not rebuilt or replaced.

The expected test warnings are the intentional duplicate-ZIP fixture and upstream AutoShot indexing deprecation. No original test was changed or disabled. Local machine paths, logs and recordings remain private; this public summary records outcomes rather than uploading the audit directory.

**INDEPENDENT_SECOND_PC_TEST: NOT TESTED.** Other Windows versions, GPU deployment and new manual visual inspection were not tested in this synchronization. The public source checkout is not the complete faculty archive: follow README and MODELS.md provisioning instructions. Historical benchmark scores are separate and unchanged. No automatic merge or claim of GitHub CI success is implied by these local results.
