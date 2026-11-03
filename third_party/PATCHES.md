# Runtime portability patches

Original user research/checkpoints remain unchanged outside the new repository. The original V40 ZIP and original notebooks preserve source evidence. Deployment copies have these explicit changes:

- OmniShotCut `engine.py`: `load_model(..., device="cpu")` and `_run_on_numpy` uses the model's actual device. Strict state loading, transforms, window merging and predictions are unchanged. The unused original file-based CUDA routine remains historical code.
- OmniShotCut `architecture/backbone.py`: `weights=None` prevents an unnecessary online ImageNet download before loading the complete supplied state dict.
- SBDV2 `python/sbdv.py`: full `ctypes` argtypes for verify/features protect 64-bit model handles; Windows binary discovery includes `.exe`.
- SBDV2 `src/sbdv.hpp`: C++17 `filesystem::u8path` opens UTF-8 model paths on Windows. Feature/model/decision math is unchanged.
- SBDV2 `CMakeLists.txt`: fix invalid single-line conditional syntax, select the Windows CLI transport, link shell32.
- New `src/sbdv_cli_windows.cpp`: raw RGB input transport and UTF-8 arguments, invoking original `run_pipeline` or `Streamer` directly. It supports `info`, `verify --model --frames --cuts`, `--ram`, `--stream`, `--out`. The original Linux CLI is preserved. Video decoding for the Windows route is in `scripts/verify_video_cli.py` and uses safe argument lists.
- AutoShot architecture/linear/utils and the TinyCut helper are byte-for-byte copies. AutoShot's exact sigmoid/logit and 100/50 window behavior is implemented in its adapter; native official scene ranges are retained. TinyCut skips only the four unused CLAHE feature columns; exact reference parity is tested.

The original top-level easy_sbd_verifier.py from V40 is preserved as a reference; the app uses the explicit patched ABI bridge and single default model path. Original research/training modules in third_party are not imported as application entry points.
