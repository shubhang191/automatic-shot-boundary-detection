# Ownership and redistribution

Project-origin TinyCut, fitted verifiers, SBDV2 and integration are published by the project owner for inspection/reproducibility. No blanket open-source license has been selected for these components; all rights remain with their respective owners. No license is invented for pretrained weights.

AutoShot source: Wentao Zhu et al., MIT; third_party/autoshot/LICENSE. The external checkpoint is NOT redistributed; obtain from authors as documented in docs/MODELS.md. AutoShot uses TransNet V2 lineage (Soucek and Lokoc); acknowledge both original works.

OmniShotCut source and official weights: Boyang Wang et al., UVA, MIT. Licenses/cards included. Pretrained weights are downloaded directly from official immutable Hugging Face releases, not committed. DETR-derived code retains Facebook copyright and Apache 2.0 notices under third_party/notices/DETR_LICENSE.txt.

C++ runtime: LLVM/MinGW notices in third_party/notices, Apache 2.0 with LLVM exceptions and MinGW runtime terms. Bundled Windows verifier DLL/EXE are project-built; compiler not bundled.

FFmpeg: not committed. scripts/bootstrap_ffmpeg.py obtains Gyan's pinned 9.0.2 archive, verifies SHA and retains upstream LICENSE/README. Observe the downloaded distribution's GPL and source/build terms.

Sintel sample and derived application screenshots: copyright Blender Foundation, CC BY 3.0, https://durian.blender.org/sharing/ and https://creativecommons.org/licenses/by/3.0/. Edits described in examples/README.md. No endorsement. No SHOT200/BBC/ad footage is included.

Python packages retain their upstream licenses and are installed from original distributions. No package wheels or virtual environment are republished.
