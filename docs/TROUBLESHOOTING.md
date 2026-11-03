# Troubleshooting

- Use Windows x64 and Python 3.13 (tested 3.13.5). CPU pins are intentional; CUDA is not certified.
- If PowerShell activation is blocked, invoke .venv\Scripts\python.exe directly without changing system execution policy.
- Missing AutoShot: follow MODELS.md; obtain the exact file from the original authors, then run prepare_models.py --autoshot downloads/ckpt_0_200_0.pth. Their Baidu service may require an account/client. No substitute or silent fallback is used.
- Checksum mismatch: stop and reacquire the pinned artifact. Never disable integrity checks. Only remove your own failed .download file, not research models.
- Missing FFmpeg/FFprobe: run scripts/bootstrap_ffmpeg.py. The official Windows archive is downloaded, SHA-verified and its notices retained. Different PATH versions may change preprocessing and cache keys.
- Native DLL failure: retain third_party/sbdv2/bin and use x64 Python. No compiler is required for bundled binaries. Rebuilding from supplied C++ source needs a compatible toolchain and is optional.
- Port busy: run.bat prints the selected localhost port (7860–7879). Keep its terminal open and stop with Ctrl+C.
- Permission/path errors: clone to a short writable path outside system/protected/synchronized directories.
- Joblib/sklearn incompatibility: use the pinned sklearn 1.6.1, joblib 1.4.2 and dependencies; do not upgrade individually.
- TinyCut Python is unsupported; choose Raw or C++. Both Omni variants and AutoShot support Python.
- The preserved AutoShot/PyTorch indexing deprecation warning is known; do not rewrite inference to suppress it.
- Clip limits: 500 MiB, 300 s, 3840x2160, 1–60 fps; Omni decoded input has a 256 MiB cap. These are enforced limits, not minimum hardware measurements.
